# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Keccak-f[1600] — the permutation under SHA-3, SHAKE, and Keccak-256.

Implements `Permutation` over `uint32` lane halves: `width = 50` elements
carrying 25 lanes of 64 bits, `dtype = uint32`. The seam takes any dtype a
construction can allocate and index, so this needs no field arithmetic and no
seam of its own; `lane.py` explains why the halves are not one `uint64`.

The state is carried as a `(5, 5)` grid indexed `[y][x]`, per half, and every
step is element-wise over the whole grid. Doing it lane by lane instead is what
a first cut does and it does not survive: one scalar op per lane per round
compiles to ~9,800 HLO lines for a *single* round, so the 24-round body never
finishes compiling. Grid-shaped steps keep the same schedule in a few dozen ops
per round.

The single-kernel rule is what shapes the rest. The 24 rounds unroll at trace
time (a Python loop over a fixed count, so nothing traced), theta's column
parity is an unrolled XOR fold rather than a reduction, rho's per-lane offsets
become constant arrays rather than branches, and pi is a static reorder built
from slices — a runtime lane permutation would lower to a gather and split the
kernel.

`has_dedicated_fusion` is False: there is no `hash_frx.keccak_f` emitter yet
(#21), so `permute` marks itself with the generic `zorch.fused_region` rather
than a name-routed one. False means *generic marker*, not *no marker* — the
contract's guarantee that a permutation call is one device unit does not wait
for a dedicated emitter.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array

from hash_frx.fusion import FUSED_REGION_MARKER, fused_region
from hash_frx.keccak import lane as lanes
from hash_frx.keccak.lane import Lane
from hash_frx.keccak.params import (
    ROTATION_OFFSETS,
    ROUND_CONSTANTS,
    ROUNDS,
    WIDTH,
)
from hash_frx.word import roll

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation

# rho's offsets as a (5, 5) [y][x] constant, matching the grid.
_ROT = np.asarray(ROTATION_OFFSETS, dtype=np.uint32).reshape(5, 5)


def _unpack(state: Array) -> Lane:
    """`(50,)` interleaved -> a `(5, 5)` grid per half.

    The strided slices are static, so they lower to `slice` rather than a
    gather; `[y][x]` falls out of the flat lane index being `x + 5*y`.
    """
    return state[0::2].reshape(5, 5), state[1::2].reshape(5, 5)


def _pack(a: Lane) -> Array:
    """A `(5, 5)` grid per half back to the `(50,)` interleaved state."""
    lo, hi = a
    return fnp.stack([lo.reshape(25), hi.reshape(25)], axis=1).reshape(WIDTH)


def _theta(a: Lane) -> Lane:
    """C[x] = parity of column x; D[x] = C[x-1] ^ rotl(C[x+1], 1); A ^= D[x].

    The parity folds the five rows with four XORs rather than reducing over the
    axis: a reduction is the `kInput` fusion boundary that would split the round
    body. `roll` is a static shift, so it stays slices and a concatenate.
    """
    lo, hi = a
    c_lo, c_hi = lo[0], hi[0]
    for y in range(1, 5):
        c_lo, c_hi = c_lo ^ lo[y], c_hi ^ hi[y]

    rot_lo, rot_hi = lanes.rotl((c_lo, c_hi), 1)
    d_lo = roll(c_lo, 1) ^ roll(rot_lo, -1)
    d_hi = roll(c_hi, 1) ^ roll(rot_hi, -1)
    return lo ^ d_lo, hi ^ d_hi


def _pi(g: Array) -> Array:
    """pi's lane movement, as a shear then a transpose then a row permutation.

    FIPS 202 states pi as "the lane at (x, y) moves to (y, 2x + 3y)". Read by
    destination on the `[y][x]` grid that is `out[Y][X] = g[X][(X + 3Y) % 5]`.
    Rolling row `X` left by `X`
    puts the wanted element of every row in a common column, the transpose turns
    those columns into rows, and a static row reorder finishes it.

    The obvious spelling — 25 unit slices into one concatenate — is a static
    reorder too and emits no gather, but it fans the whole rotation chain out
    into 25 consumers, and XLA re-materialises that chain into each one: 1197
    fusion computations against 285 for this form, and 6.6x the runtime. A
    reorder being gather-free is not the same as it being cheap.
    """
    sheared = fnp.concatenate([roll(g[x : x + 1], -x, axis=1) for x in range(5)])
    columns = sheared.T
    return fnp.concatenate([columns[(3 * y) % 5 : (3 * y) % 5 + 1] for y in range(5)])


def _rho_pi(a: Lane) -> Lane:
    """Rotate every lane by its own offset, then move it to its pi destination.

    Both steps are static: the offsets are a constant grid, and the movement is
    a fixed reorder built from slices, so no index reaches the device.
    """
    rot_lo, rot_hi = lanes.rotl_each(a, _ROT)
    return _pi(rot_lo), _pi(rot_hi)


def _chi(a: Lane) -> Lane:
    """A[y][x] ^= (~A[y][x+1]) & A[y][x+2] — the only non-linear step.

    The neighbours are the row rolled by one and two, so the whole grid updates
    element-wise.
    """
    lo, hi = a
    lo1, hi1 = roll(lo, -1, axis=1), roll(hi, -1, axis=1)
    lo2, hi2 = roll(lo, -2, axis=1), roll(hi, -2, axis=1)
    return lo ^ ((~lo1) & lo2), hi ^ ((~hi1) & hi2)


def _patch_lane_zero(g: Array, value: int) -> Array:
    """XOR `value` into lane (0, 0) only, leaving the rest of the grid untouched.

    XORing a whole grid that is zero elsewhere is simpler to write but not
    equivalent in cost: XLA drops `x ^ 0` only when the constant is *entirely*
    zero, so a grid with one live lane is 25 lanes of work, of which 24 are
    wasted. Slicing the one lane out and concatenating it back is four more
    instructions and ~24x less element work.
    """
    if value == 0:
        return g
    head = g[0:1, 0:1] ^ fnp.uint32(value)
    row0 = fnp.concatenate([head, g[0:1, 1:5]], axis=1)
    return fnp.concatenate([row0, g[1:5]])


def _iota(a: Lane, rnd: int) -> Lane:
    """Lane (0, 0) takes this round's constant. The halves are split on the
    host, so no 64-bit literal is ever materialised — and a half that is zero
    (eleven of the twenty-four high halves) costs nothing at all."""
    lo, hi = a
    rc_lo, rc_hi = ROUND_CONSTANTS[rnd]
    return _patch_lane_zero(lo, rc_lo), _patch_lane_zero(hi, rc_hi)


def _rounds(state: Array) -> Array:
    """The 24 rounds, unrolled — the decomposition the marked region runs."""
    a = _unpack(state)
    for rnd in range(ROUNDS):
        a = _iota(_chi(_rho_pi(_theta(a))), rnd)
    return _pack(a)


# Module-level jit zone so the permutation body traces once per state aval
# process-wide: `lax.composite` re-traces its decomposition on every emission,
# and one sponge absorb emits a permute per block. `inline=True` splices the
# cached jaxpr into the enclosing trace, so the emitted module is unchanged.
# Keccak-f is parameterless, so unlike its siblings there is no static key.
@partial(frx.jit, inline=True)
def _permute_body(state: Array) -> Array:
    """`permute` as ONE `zorch.fused_region`.

    The marker is the contract rather than an optimisation: a permutation call
    *is* one marked region by construction, so an unmarked body leaves nothing
    naming the unit. `has_dedicated_fusion = False` selects *which* marker, not
    whether there is one — the same thing `SparsePoseidon` does on its
    non-dedicated path.

    The generic marker is also what obliges the body to be straight-line and
    element-wise, since the generic rewriter accepts nothing else.
    """
    return fused_region(_rounds, state, name=FUSED_REGION_MARKER)


class KeccakF1600:
    """Keccak-f[1600] as a `Permutation` over `uint32` lane halves.

    Stateless and parameterless — the standard fixes width, rounds, offsets, and
    constants — so every instance equals every other, which keeps it a stable
    static jit-zone key the way the `Permutation` contract requires.
    """

    width = WIDTH
    dtype = fnp.uint32
    # No `hash_frx.keccak_f` emitter exists yet; consumers fall back to the
    # generic region marker, which carries no version.
    fused_region_marker = (FUSED_REGION_MARKER, 0)
    has_dedicated_fusion = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KeccakF1600):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(KeccakF1600)

    def permute(self, state: Array) -> Array:
        """Apply Keccak-f[1600]: `(50,)` uint32 -> `(50,)`.

        Batch with `frx.vmap(permute)`; the body is element-wise over the lane
        grid, so a batched call lowers to the same straight-line graph.
        """
        if state.ndim != 1 or state.shape[0] != WIDTH:
            raise ValueError(
                f"state must be a 1-D array of shape ({WIDTH},), got {state.shape}"
            )
        if state.dtype != self.dtype:
            raise TypeError(
                f"state dtype {state.dtype} must be {self.dtype} — a lane is two "
                "uint32 halves, not one 64-bit word"
            )
        return _permute_body(state)

    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """Inert: the round constants are inlined literals rather than ABI
        operands, so there is no dedicated layout to hand out until an emitter
        exists (#21)."""
        return (leading,), (lambda state, *_ops: _permute_body(state)), {}


if TYPE_CHECKING:
    _perm: type[Permutation] = KeccakF1600

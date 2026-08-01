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

`has_dedicated_fusion` is False: there is no `hash_frx.keccak_f` emitter yet, so
a consumer wrapping a region over this permutation gets the generic marker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import frx.numpy as fnp
import numpy as np
from frx import Array

from hash_frx.keccak import lane as lanes
from hash_frx.keccak.lane import Lane
from hash_frx.keccak.params import (
    PI_SOURCE,
    ROTATION_OFFSETS,
    ROUND_CONSTANTS,
    ROUNDS,
    WIDTH,
)

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation

# rho's offsets as a (5, 5) [y][x] constant, matching the grid.
_ROT = np.asarray(ROTATION_OFFSETS, dtype=np.uint32).reshape(5, 5)
# iota touches lane (0, 0) only, so each round's constant is a full grid that is
# zero everywhere else — one XOR instead of an indexed update.
_RC_LO = np.zeros((ROUNDS, 5, 5), dtype=np.uint32)
_RC_HI = np.zeros((ROUNDS, 5, 5), dtype=np.uint32)
for _r, (_lo, _hi) in enumerate(ROUND_CONSTANTS):
    _RC_LO[_r, 0, 0] = _lo
    _RC_HI[_r, 0, 0] = _hi


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
    d_lo = fnp.roll(c_lo, 1) ^ fnp.roll(rot_lo, -1)
    d_hi = fnp.roll(c_hi, 1) ^ fnp.roll(rot_hi, -1)
    return lo ^ d_lo, hi ^ d_hi


def _rho_pi(a: Lane) -> Lane:
    """Rotate every lane by its own offset, then move it to its pi destination.

    Both halves of the step are static: the offsets are a constant grid, and pi
    is a fixed reorder expressed as `PI_SOURCE` gathered *on the host* into a
    concatenate of unit slices, so no index reaches the device.
    """
    rot_lo, rot_hi = lanes.rotl_each(a, _ROT)
    flat_lo, flat_hi = rot_lo.reshape(25), rot_hi.reshape(25)
    out_lo = fnp.concatenate([flat_lo[s : s + 1] for s in PI_SOURCE])
    out_hi = fnp.concatenate([flat_hi[s : s + 1] for s in PI_SOURCE])
    return out_lo.reshape(5, 5), out_hi.reshape(5, 5)


def _chi(a: Lane) -> Lane:
    """A[y][x] ^= (~A[y][x+1]) & A[y][x+2] — the only non-linear step.

    The neighbours are the row rolled by one and two, so the whole grid updates
    element-wise.
    """
    lo, hi = a
    lo1, hi1 = fnp.roll(lo, -1, axis=1), fnp.roll(hi, -1, axis=1)
    lo2, hi2 = fnp.roll(lo, -2, axis=1), fnp.roll(hi, -2, axis=1)
    return lo ^ ((~lo1) & lo2), hi ^ ((~hi1) & hi2)


def _iota(a: Lane, rnd: int) -> Lane:
    """Lane (0, 0) takes this round's constant, as a whole-grid XOR against a
    constant that is zero elsewhere. The halves are split on the host, so no
    64-bit literal is ever materialised."""
    lo, hi = a
    return lo ^ fnp.asarray(_RC_LO[rnd]), hi ^ fnp.asarray(_RC_HI[rnd])


def _permute_body(state: Array) -> Array:
    """The 24 rounds, unrolled — the decomposition a fused region would run."""
    a = _unpack(state)
    for rnd in range(ROUNDS):
        a = _iota(_chi(_rho_pi(_theta(a))), rnd)
    return _pack(a)


class KeccakF1600:
    """Keccak-f[1600] as a `Permutation` over `uint32` lane halves.

    Stateless and parameterless — the standard fixes width, rounds, offsets, and
    constants — so every instance equals every other, which keeps it a stable
    static jit-zone key the way the `Permutation` contract requires.
    """

    width = WIDTH
    dtype = fnp.uint32
    # No `hash_frx.keccak_f` emitter exists yet; consumers fall back to the
    # generic region marker.
    has_dedicated_fusion = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KeccakF1600):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self).__name__)

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

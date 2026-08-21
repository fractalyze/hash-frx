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

`permute` marks itself with `hash_frx.perm.keccak_f` where that emitter can actually
be reached and the generic `zorch.fused_region` where it cannot — which takes
both `_DEDICATED_EMITTER_AVAILABLE` (does the pinned plugin carry it) and
`_EMITTER_BACKENDS` (does this backend), since the Keccak emitters are GPU-only.
`fusion_path` reports which of the two ran (`DEDICATED` vs `GENERIC`), and the
two are not a fast and a slow kernel: **only the dedicated marker is one device
unit.**

A bare `zorch.fused_region` carries no live-width operand, and the rewriter
declines those on purpose — routing one to the generic loop fusion would build an
indexed per-element subgraph that re-executes shared producers per output
element, which is exponential in body depth. So it inlines instead, and ordinary
fusion materializes the intermediates: a single permutation lands as ~70 fusions
rather than one. The fusion contract is met by a *recognized* marker, never by
the fallback, which is why the switch has to track the pin.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array

from hash_frx.fusion import (
    FUSED_REGION_MARKER,
    FusionPath,
    fused_region,
    inert_region_spec,
)
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

KECCAK_F_MARKER = "hash_frx.perm.keccak_f"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# below. XLA recognizes the marker by name + attributes and deliberately does not
# gate on the version, which exists so a contract change can stage without a
# rename (`hash_frx.fusion`).
KECCAK_F_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships the dedicated Keccak emitters.
# Two separate things turn on this flag, and the second is why it must track the
# pin rather than being left optimistic: emitting an unrecognized *name* is
# byte-neutral (the composite inlines and the fusion is silently lost), but
# a `DEDICATED` fusion path also routes a `Sponge` over this permutation to
# `hash_frx.digest.field_sponge` carrying `permutation="keccak_f"`, which a
# plugin without the arm rejects outright as an unknown permutation — a hard
# compile failure, not
# a fallback. Nothing here builds that `Sponge` (the byte hashes go through
# `KeccakSponge` and its own marker), so today only the first applies; the second
# is what makes the flag a pin question rather than a preference.
#
# Flipped together with the `frx>=` floor in `pyproject.toml`, like
# `poseidon.sparse._DEDICATED_EMITTER_AVAILABLE`. Below that floor the marker is
# emitted, ignored, and computes the right bytes with none of the kernels, so no
# value test can see it violated — only `MarkerRecognizedTest`, which reads the
# compiled module, and only on a leg that has the emitter to begin with.
_DEDICATED_EMITTER_AVAILABLE = True

# Which backends have those emitters, which is a *different* question from the
# pin and has to be asked alongside it. `gpu_compiler.cc` sets
# `allow_keccak_f_fusion` and `allow_keccak_sponge_fusion`; the CPU compiler sets
# neither and says why where it builds the options — "the zerocheck-URM,
# constraint-pressure-group and Keccak emitters are GPU-only, so they stay false
# and their markers inline here." There is no `keccak.cc` under
# `xla/backends/cpu/codegen/emitters/` to set them from.
#
# Routing to a dedicated emitter that the backend does not have is not merely
# neutral, which is why this cannot be left optimistic the way an unrecognized
# name can. The whole-hash `KeccakSponge` marker traces the entire padded absorb
# and squeeze chain into one composite; where it is honoured that is the win, and
# where it is not it is pure cost. Measured on `enc-frx`'s ML-KEM-512 `decaps` at
# `B = 32`: 6.84s of tracing against 0.40s for the per-permutation routing, for a
# module that optimizes to the same 7645 loop fusions either way.
#
# A list rather than `!= "unknown"`, and for the same reason
# `blake3.testing.emitter` keeps one: a backend is on the generic path until an
# emitter is *written* for it, so a leg that gains an arm joins this tuple and
# nothing else here moves.
_EMITTER_BACKENDS = ("gpu",)


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry the Keccak emitters.

    Read per construction rather than at import, so importing this module does
    not initialize a backend, and so a test can pin either answer. The backend
    lookup behind `frx.default_backend()` is memoized, so this stays cheap on the
    per-call construction the byte hashes do.
    """
    return _DEDICATED_EMITTER_AVAILABLE and frx.default_backend() in _EMITTER_BACKENDS


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


def _rho_pi(a: Lane, tables: lanes.RhoTables) -> Lane:
    """Rotate every lane by its own offset, then move it to its pi destination.

    Both steps are static: the rotation tables are derived once from the offset
    grid the region takes as an operand, and the movement is a fixed reorder
    built from slices, so no index reaches the device.
    """
    rot_lo, rot_hi = lanes.rotl_each(a, tables)
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


def _rounds(state: Array, rho_offsets: Array, **_attrs: object) -> Array:
    """The 24 rounds, unrolled — the decomposition the marked region runs.

    Takes rho's offset grid as an operand rather than closing over it: an array
    the body materialises on the host is lifted into an unnamed operand ahead of
    the declared ones, once per materialisation site, so closing over it put 96
    anonymous constants in front of the state and left no ABI to write down.
    `_attrs` is marker metadata passed through, which the body does not read.
    """
    tables = lanes.rho_tables(rho_offsets)
    a = _unpack(state)
    for rnd in range(ROUNDS):
        a = _iota(_chi(_rho_pi(_theta(a), tables)), rnd)
    return _pack(a)


def _abi_operands(state: Array) -> tuple[Array, ...]:
    """The marked region's operands, in the order the emitter's ABI names them:

    ``[0] state``         `uint32[..., 50]` — 25 lanes as interleaved halves
    ``[1] rho_offsets``   `uint32[5, 5]`    — rho's per-lane rotation counts,
                                              `[y][x]`, as FIPS 202 tabulates them

    Keccak-f[1600] is otherwise parameterless: width, round count, round
    constants and the pi reorder are fixed by the standard, so an emitter bakes
    them and only the offsets ride, because the fallback decomposition needs a
    non-materialised copy of them (`_rounds`).
    """
    return state, fnp.asarray(_ROT)


def _marker_attrs() -> dict[str, object]:
    """The dedicated marker's `composite.attributes` — the recognizer's contract.
    Identifying rather than parameterising, the permutation having no free
    parameters; the body ignores them and the generic marker stays attrs-free."""
    return {"permutation": "keccak_f", "width": WIDTH, "rounds": ROUNDS}


# Module-level jit zone so the permutation body traces once per (permutation,
# state aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and one sponge absorb emits a permute per block. `inline=True`
# splices the cached jaxpr into the enclosing trace, so the emitted module is
# unchanged. Keccak-f has no free parameters, but the permutation is still the
# static key: the marker it carries is not a function of the parameters, so
# without it a dedicated and a generic instance collide in this cache.
@partial(frx.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: "KeccakF1600", state: Array) -> Array:
    """`permute` as ONE marked region.

    The marker is the contract rather than an optimisation: a permutation call
    *is* one marked region by construction, so an unmarked body leaves nothing
    naming the unit. `fusion_path` selects *which* marker, not whether there is
    one — the same thing `SparsePoseidon` does on its non-dedicated path.

    The generic marker is also what obliges the body to be straight-line and
    element-wise, since the generic rewriter accepts nothing else. The dedicated
    one carries the same body: a marker the plugin does not recognize inlines,
    and the fallback has to be the reference decomposition.
    """
    name, version = perm.fused_region_marker
    return fused_region(
        _rounds,
        *_abi_operands(state),
        name=name,
        version=version,
        **(_marker_attrs() if perm.fusion_path.is_one_kernel else {}),
    )


class KeccakF1600:
    """Keccak-f[1600] as a `Permutation` over `uint32` lane halves.

    Stateless and parameterless — the standard fixes width, rounds, offsets, and
    constants — so two instances differ only in the marker they route to, which
    is what keeps them a stable static jit-zone key the way the `Permutation`
    contract requires.
    """

    width = WIDTH
    dtype = fnp.uint32

    def __init__(self) -> None:
        # Read per instance rather than pinned on the class so the emitter flag
        # is a value the permutation carries, and a test can construct both
        # routings in one process.
        name = (
            KECCAK_F_MARKER if _routes_to_dedicated_emitter() else FUSED_REGION_MARKER
        )
        # A generic region carries no version: the recognizer reads only the name
        # there, so a version would claim a contract the marker does not have.
        self.fused_region_marker = (
            name,
            KECCAK_F_MARKER_VERSION if name != FUSED_REGION_MARKER else 0,
        )
        self.fusion_path = FusionPath.from_marker(name)

    def __eq__(self, other: object) -> bool:
        # The parameter surface is empty, so the marker IS the identity: without
        # it a dedicated and a generic instance collide in `_permute_body`'s
        # static-arg cache, and the second would reuse the first's marker.
        if not isinstance(other, KeccakF1600):
            return NotImplemented
        return self.fused_region_marker == other.fused_region_marker

    def __hash__(self) -> int:
        return hash((KeccakF1600, self.fused_region_marker))

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
        return _permute_body(self, state)

    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """The KeccakFusion ABI: operands `(leading, rho_offsets)`, the
        operand-fed permute, and the identifying attrs. Dedicated path only;
        otherwise the shared inert stub."""
        if not self.fusion_path.is_one_kernel:
            return inert_region_spec(self, leading)
        return _abi_operands(leading), _rounds, _marker_attrs()


if TYPE_CHECKING:
    _perm: type[Permutation] = KeccakF1600

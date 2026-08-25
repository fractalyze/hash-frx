# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ascon-p[12] — the permutation under Ascon-Hash256 and the Ascon XOFs.

Implements `Permutation` over `uint32` word halves: `width = 10` elements
carrying the five 64-bit words S0..S4, `dtype = uint32`. The seam takes any
dtype a construction can allocate and index, so this needs no field arithmetic;
`keccak/lane.py` carries the law that a 64-bit word rides as two `uint32`
halves rather than one `uint64` (with x64 off `uint64` truncates, and enabling
it flips process-wide defaults).

**The interleave is Keccak-f[1600]'s**: word `i` occupies elements `2i` (low
half) and `2i + 1` (high half), so a rate of `r` bytes is the contiguous prefix
`state[: r // 4]` and packing is a plain little-endian read. Sharing the layout
law with the other bit-oriented permutation in the package is what lets one byte
sponge schedule drive both.

The round steps are grid-wise over the five words rather than over ten loose
half arrays, which is load-bearing for the lowering rather than a style choice —
`permutation` carries the measurement. Every step is written against the
trailing axis alone, so the same body serves the unbatched `(10,)` state this
seam promises and the `[B, 5]` grids `ascon.ascon_hash256_bytes` absorbs into.

Ascon-p[12] is twelve rounds of constant addition (SP 800-232 §3.2), a 5-bit
S-box applied bitsliced across the five words (§3.3), and per-word linear
diffusion of two rotated copies (§3.4); §5.1 uses the same twelve-round
permutation for initialization, absorbing and squeezing alike. Section
references are to NIST SP 800-232 (final, August 2025,
https://doi.org/10.6028/NIST.SP.800-232).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
from frx import Array, lax

from hash_frx.fusion import (
    FUSED_REGION_MARKER,
    FusionPath,
    fused_region,
    inert_region_spec,
    routing,
)
from hash_frx.word import roll
from hash_frx.word64 import rotr64

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation

U32 = fnp.uint32

# A 64-bit quantity as (low 32 bits, high 32 bits) of equal shape. The state is
# one such pair of uint32 grids over the five words — word i in column i.
Lane = tuple[Array, Array]

WORDS = 5  # §5.1: a 320-bit state of five 64-bit words S0..S4
ROUNDS = 12  # §5.1: every phase runs Ascon-p[12]
# Elements of the flat seam state: five words, each two uint32 halves.
WIDTH = WORDS * 2

ASCON_P_MARKER = "hash_frx.perm.ascon_p"
# Marker revision riding as `composite.version`; version 1 is the one-operand
# ABI in `_abi_operands`. XLA recognizes a marker by name + attributes and
# deliberately does not gate on the version, which exists so a contract change
# can stage without a rename once a recognizer ships (`hash_frx.fusion`).
ASCON_P_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships a dedicated Ascon-p emitter,
# and on which backends. None exists yet — the pre-emitter posture the other
# emitterless families hold, and the one `ascon.ascon_hash256_bytes` holds for
# the digest marker: both flags flip together with the `frx>=` floor in `pyproject.toml`
# when an emitter lands, and `fusion_path_test`'s matrix law holds them to
# agree. Until then `permute` carries the generic region marker.
_DEDICATED_EMITTER_AVAILABLE = False

# Which backends have that emitter — a different question from the pin, asked
# alongside it. Empty until one is written; a backend gaining an arm joins this
# tuple and nothing else here moves (the keccak/poseidon2 pattern).
_EMITTER_BACKENDS: tuple[str, ...] = ()


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry this emitter
    (`fusion.routing`, which carries the rationale)."""
    return routing(_DEDICATED_EMITTER_AVAILABLE, _EMITTER_BACKENDS)


# Round i of Ascon-p[rnd] XORs c_i = const_{16-rnd+i} into S2 (§3.2, eqs. 3-4;
# Table 5) — for rnd = 12 that is const_4..const_15, listed here in round order.
# Only the low 8 bits are ever set, so the XOR touches the low half alone, as
# Python-int scalar literals in the traced body (the `keccak._iota`
# arrangement: a scalar is a literal in the emitted module, never a
# host-materialised array, so the operand-lifting rule does not apply and the
# ABI stays one operand).
ROUND_CONSTANTS = (
    0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B,
)  # fmt: skip

# The Σ_i rotation pairs (§3.4, eqs. 8-12): S_i ^= (S_i ⋙ r1) ^ (S_i ⋙ r2).
SIGMA_ROTATIONS = ((19, 28), (61, 39), (1, 6), (10, 17), (7, 41))

# The three word-position masks the substitution layer applies, as uint32 [5]
# 0/1 grids (`masks`): which words take the leading XOR of their predecessor,
# which the trailing one, and which is complemented.
Masks = tuple[Array, Array, Array]


def masks() -> Masks:
    """The S-box layer's word-position masks, derived from `iota` on device.

    Host-built 0/1 vectors would be arrays the decomposition materialises,
    which `lax.composite` lifts into unnamed operands ahead of the declared
    ABI — so they are counted on device instead, the `iota` remedy of
    docs/reference/conventions.md (`blake3.modes._counters` is the precedent).
    Computed once per permutation or digest and shared by every round.

    - ``pre``: words {0, 2, 4} — the even positions, `(i & 1) ^ 1`.
    - ``post``: words {0, 1, 3} — the positions where i + 1 keeps no bit of
      i (i + 1 a power of two or i = 0), `((i + 1) & i) == 0`.
    - ``word2``: word {2} alone, for the constant XOR and the complement.
    """
    idx = lax.iota(U32, WORDS)
    one = U32(1)
    pre = (idx & one) ^ one
    post = lax.convert_element_type(((idx + one) & idx) == 0, U32)
    word2 = lax.convert_element_type(idx == U32(2), U32)
    return pre, post, word2


def _substitution(lo: Array, hi: Array, m: Masks) -> Lane:
    """p_S (§3.3): the 5-bit S-box across the five words, grid-wise.

    The substitution is 64 parallel S-box applications with word S_i
    supplying bit plane x_i (eq. 5) — the state is already bitsliced, so no
    per-bit extraction happens anywhere. This is the Figure 3 circuit with
    its word-crossing wires spelled as static rolls of the word axis:

    - the leading XORs x0 ^= x4, x2 ^= x1, x4 ^= x3 all read word i - 1
      into word i, at the even positions — `roll(x, 1)` masked by ``pre``;
    - the χ core t_i = ~x_i & x_{i+1}, x_i ^= t_{i+1} is
      x ^= ~roll(x, -1) & roll(x, -2), the `keccak._chi` spelling;
    - the trailing XORs x1 ^= x0, x0 ^= x4, x3 ^= x2 again read word i - 1,
      at positions {0, 1, 3} — the same roll masked by ``post``;
    - x2 is complemented, an XOR with ``word2`` stretched to all-ones.

    Each masked stage's reads all land on values none of its writes touch,
    which is what lets the standard's sequential assignments run as one
    parallel grid op per stage. Every gate is element-wise and bit-parallel,
    so one spelling serves the lo and hi planes; `ascon_test` holds it to
    the oracle's Table 6 lookup on all 32 inputs.
    """
    pre, post, word2 = m
    lo = lo ^ (roll(lo, 1, axis=-1) * pre)
    hi = hi ^ (roll(hi, 1, axis=-1) * pre)
    lo = lo ^ ((~roll(lo, -1, axis=-1)) & roll(lo, -2, axis=-1))
    hi = hi ^ ((~roll(hi, -1, axis=-1)) & roll(hi, -2, axis=-1))
    lo = lo ^ (roll(lo, 1, axis=-1) * post)
    hi = hi ^ (roll(hi, 1, axis=-1) * post)
    invert = word2 * U32(0xFFFFFFFF)
    return lo ^ invert, hi ^ invert


def _linear_diffusion(lo: Array, hi: Array) -> Lane:
    """p_L (§3.4): S_i ^= (S_i ⋙ r1) ^ (S_i ⋙ r2), the Σ_i pairs — the one
    step where the halves interact, through the shared `word64.rotr64`. Each
    word carries its own rotation pair, so the grid splits into static
    columns along the word axis and stacks back, the way Grøstl's ShiftBytes
    splits rows.

    Indexed on the trailing axis, so one spelling serves the unbatched `(5,)`
    grids the `Permutation` seam hands in and the `[B, 5]` grids the batched
    digest absorbs into."""
    out_lo, out_hi = [], []
    for i, (r1, r2) in enumerate(SIGMA_ROTATIONS):
        w = (lo[..., i], hi[..., i])
        a = rotr64(w, r1)
        b = rotr64(w, r2)
        out_lo.append(w[0] ^ a[0] ^ b[0])
        out_hi.append(w[1] ^ a[1] ^ b[1])
    return fnp.stack(out_lo, axis=-1), fnp.stack(out_hi, axis=-1)


def permutation(lo: Array, hi: Array, m: Masks) -> Lane:
    """Ascon-p[12] (§3) on the (lo, hi) uint32 word grids: twelve rounds
    of p_C -> p_S -> p_L. The round loop is a Python-unrolled `for` — the
    count is static and small, and a `lax` loop would be a control-flow
    boundary (docs/reference/conventions.md). p_C touches only S2's low
    half: the constants are 8-bit, masked to word 2 like the complement.

    The grid is load-bearing for the lowering, not a style choice. Spelled
    over ten loose half arrays the whole digest is one deep element-wise
    DAG, and this toolchain's CPU pipeline first cancels any exact-inverse
    repack a body inserts (aligned slice-of-concatenate forwarding), then
    fuses the barrier-free chain into single kLoop kernels thousands of
    instructions deep — measured on this body, a lone [4, 1]-message digest
    stopped returning inside a 900s test timeout. The grid steps' rolls and
    per-word stacks are *load-bearing* data movement the canonicalizer
    keeps, so fusion regions stay round-sized (the Grøstl/Keccak texture:
    ~1.5K kernels for a 13-permutation digest, compiled and run in
    milliseconds).
    """
    for rc in ROUND_CONSTANTS:
        lo = lo ^ (m[2] * U32(rc))
        lo, hi = _substitution(lo, hi, m)
        lo, hi = _linear_diffusion(lo, hi)
    return lo, hi


def _unpack(state: Array) -> Lane:
    """`(10,)` interleaved -> the (lo, hi) `(5,)` word grids.

    The strided slices are static, so they lower to `slice` rather than a
    gather; word `i` sits at elements `2i` / `2i + 1`, the `keccak._unpack`
    law.
    """
    return state[0::2], state[1::2]


def _pack(lo: Array, hi: Array) -> Array:
    """The (lo, hi) word grids back to the `(10,)` interleaved state."""
    return fnp.stack([lo, hi], axis=1).reshape(WIDTH)


def _rounds(state: Array, **_attrs: object) -> Array:
    """The twelve rounds over the flat seam state — the decomposition the
    marked region runs. `_attrs` is marker metadata passed through, which the
    body does not read."""
    lo, hi = _unpack(state)
    lo, hi = permutation(lo, hi, masks())
    return _pack(lo, hi)


def _abi_operands(state: Array) -> tuple[Array, ...]:
    """The marked region's operands, in the order an emitter's ABI names them:

    ``[0] state``  `uint32[..., 10]` — five words as interleaved halves

    Ascon-p[12] is otherwise parameterless: width, round count, round constants
    and the Σ rotation pairs are fixed by the standard, so an emitter bakes them
    and nothing else rides. The twelve 8-bit round constants are scalar literals
    in the body (`ROUND_CONSTANTS` states why that is lifting-safe) and the
    S-box masks are counted from `iota` on device (`masks`), so neither is a
    captured host array — which is what keeps this a one-operand ABI rather
    than one with anonymous constants in front of the state.
    """
    return (state,)


def _marker_attrs() -> dict[str, object]:
    """The dedicated marker's `composite.attributes` — the recognizer's
    contract. Identifying rather than parameterising, the permutation having no
    free parameters; the body ignores them and the generic marker stays
    attrs-free."""
    return {"permutation": "ascon_p", "width": WIDTH, "rounds": ROUNDS}


# Module-level jit zone so the permutation body traces once per (permutation,
# state aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and one sponge absorb emits a permute per block. `inline=True`
# splices the cached jaxpr into the enclosing trace, so the emitted module is
# unchanged. Ascon-p has no free parameters, but the permutation is still the
# static key: the marker it carries is not a function of the parameters, so
# without it a dedicated and a generic instance collide in this cache.
@partial(frx.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: "AsconP", state: Array) -> Array:
    """`permute` as ONE marked region.

    The marker is the contract rather than an optimisation: a permutation call
    *is* one marked region by construction, so an unmarked body leaves nothing
    naming the unit. `fusion_path` selects *which* marker, not whether there is
    one — the same thing `KeccakF1600` and `Vision` do on their non-dedicated
    paths.

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


class AsconP:
    """Ascon-p[12] as a `Permutation` over `uint32` word halves.

    Stateless and parameterless — the standard fixes the word count, the round
    count, the round constants and the Σ rotations — so two instances differ
    only in the marker they route to, which is what keeps them a stable static
    jit-zone key the way the `Permutation` contract requires.

    **This seam is what a generic sponge takes.** `ascon.ascon_hash256_bytes`
    does not go through it: its whole padded absorb and squeeze is one
    name-routed region, so the permute inside is the bare round body rather than
    a nested marker — the arrangement `keccak.sponge._fused_hash` uses, where
    `fused_region_over` rebuilds the permute from the region's own operands.
    """

    width = WIDTH
    dtype = fnp.uint32

    def __init__(self) -> None:
        # Read per instance rather than pinned on the class so the emitter flag
        # is a value the permutation carries, and a test can construct both
        # routings in one process.
        name = ASCON_P_MARKER if _routes_to_dedicated_emitter() else FUSED_REGION_MARKER
        # A generic region carries no version: the recognizer reads only the
        # name there, so a version would claim a contract the marker does not
        # have.
        self.fused_region_marker = (
            name,
            ASCON_P_MARKER_VERSION if name != FUSED_REGION_MARKER else 0,
        )
        self.fusion_path = FusionPath.from_marker(name)

    def __eq__(self, other: object) -> bool:
        # The parameter surface is empty, so the marker IS the identity: without
        # it a dedicated and a generic instance collide in `_permute_body`'s
        # static-arg cache, and the second would reuse the first's marker.
        if not isinstance(other, AsconP):
            return NotImplemented
        return self.fused_region_marker == other.fused_region_marker

    def __hash__(self) -> int:
        return hash((AsconP, self.fused_region_marker))

    def permute(self, state: Array) -> Array:
        """Apply Ascon-p[12]: `(10,)` uint32 -> `(10,)`.

        Batch with `frx.vmap(permute)`; the body is element-wise over the word
        grid, so a batched call lowers to the same straight-line graph.
        """
        if state.ndim != 1 or state.shape[0] != WIDTH:
            raise ValueError(
                f"state must be a 1-D array of shape ({WIDTH},), got {state.shape}"
            )
        if state.dtype != self.dtype:
            raise TypeError(
                f"state dtype {state.dtype} must be {self.dtype} — a word is two "
                "uint32 halves, not one 64-bit word"
            )
        return _permute_body(self, state)

    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """The Ascon-p ABI: the state operand alone, the operand-fed permute,
        and the identifying attrs. Dedicated path only; otherwise the shared
        inert stub."""
        if not self.fusion_path.is_one_kernel:
            return inert_region_spec(self, leading)
        return _abi_operands(leading), _rounds, _marker_attrs()


if TYPE_CHECKING:
    _perm: type[Permutation] = AsconP

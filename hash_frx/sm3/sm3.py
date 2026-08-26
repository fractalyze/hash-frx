# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SM3 (GB/T 32905-2016; ISO/IEC 10118-3), authored in frx — byte-identical to
the Chinese national standard (and any conforming implementation, e.g.
OpenSSL's, which `hashlib.new("sm3")` reaches). Structurally SHA-256's close
cousin, and this module is `sha256.py`'s shape section for section:
Merkle–Damgård over 64-byte blocks, an 8×uint32 big-endian state, 64 rounds,
MD strengthening with a 64-bit bit-length field. What is SM3's own: the
message expansion (a 68-word W plus W'_j = W_j ⊕ W_{j+4}), boolean functions
that switch form at round 16, the P0/P1 linear diffusion permutations, a
per-round ROTATING constant ⟪T_j, j mod 32⟫, and an XOR feedforward where
SHA-2 adds.

The round/expansion loop is ONE `fori_loop` carrying a [B, 16] shift-register
window, split in two at round 16 where the boolean functions change form —
two loops with static bodies rather than one loop selecting per round, so
neither form's boolean work is ever computed and discarded. Only *static*
column slices index the window; the one traced index is the rotated-constant
table, which is why that table is a [64] operand (`sha256._compress`'s `k`
arrangement) rather than sixty-four scalar literals.

Bulk-parallel by construction, like `sha256`: a batch of `B` equal-length
messages is hashed in one data-parallel call.

Contract: `digest(msg)` takes uint8 `[B, L]` and returns uint8 `[B, 32]`
big-endian digests. Length `L` is static, so the padding is data-independent:
a host constant built from the length and concatenated on, which is what lets
`msg` itself be traced. SM3 is parameterless — one output length, no keyed
form in the standard — so the rows below carry by-type value identity.
Requires no x64; all arithmetic is uint32.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import (
    DeviceRow,
    device_message,
    padded_batch,
)
from hash_frx.extension.md import chain
from hash_frx.extension.pad import PadRule, Trailer
from hash_frx.fusion import FusionPath, routing
from hash_frx.markers import words_in_digest_marker
from hash_frx.word import pack_be, rotl, unpack_be

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

    pass

U32 = fnp.uint32

SM3_MARKER = "hash_frx.digest.sm3"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# in `sm3_merkle_damgard`. XLA recognizes a marker by name + attributes and
# deliberately does not gate on the version, which exists so a contract change
# can stage without a rename once a recognizer ships (`hash_frx.fusion`); an
# ABI change ships as a NEW name, the way `sha256_bytes` did.
SM3_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships a dedicated SM3 emitter, and
# on which backends. None exists yet — the pre-emitter half of the keccak
# arrangement (`keccak.permutation._DEDICATED_EMITTER_AVAILABLE` carries the
# family-wide rationale), the posture the sha512/blake2 siblings hold: both
# flags flip together with the `frx>=` floor in `pyproject.toml` when an
# emitter lands, and `fusion_path_test`'s matrix law holds them to agree. The
# marker is emitted regardless — there is no per-block routing alternative for
# a whole-hash digest — and unrecognized it inlines its decomposition: right
# bytes, `GENERIC` fusion path.
_DEDICATED_EMITTER_AVAILABLE = False

# Which backends have that emitter — a different question from the pin, asked
# alongside it. Empty until one is written; a backend gaining an arm joins
# this tuple and nothing else here moves (the keccak/poseidon2 pattern).
_EMITTER_BACKENDS: tuple[str, ...] = ()


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry this emitter
    (`fusion.routing`, which carries the rationale)."""
    return routing(_DEDICATED_EMITTER_AVAILABLE, _EMITTER_BACKENDS)


_BLOCK = 64  # GB/T 32905 §5.2: 512-bit blocks

# Initial value IV (GB/T 32905 §4.1) — the standard publishes the eight words
# directly, with no generation rule to recompute them from.
_H0 = np.array(
    [
        0x7380166F,
        0x4914B2B9,
        0x172442D7,
        0xDA8A0600,
        0xA96F30BC,
        0x163138AA,
        0xE38DEE4D,
        0xB0FB0E4E,
    ],
    dtype=np.uint32,
)


def _rotl_int(value: int, n: int) -> int:
    """Host rotate-left of a 32-bit Python int by `n mod 32` — total at 0,
    unlike the device `word.rotl`, because the T-table below hits every
    offset including 0 and 32."""
    n %= 32
    if n == 0:
        return value
    return ((value << n) | (value >> (32 - n))) & 0xFFFFFFFF


# The per-round constants, PRE-ROTATED: GB/T §4.2 defines two base words —
# T_j = 79CC4519 for rounds 0..15, 7A879D8A for 16..63 — and every round uses
# ⟪T_j, j mod 32⟫. The rotation count is the round index, so inside the
# `fori_loop` the constant is a traced-index table read; baking the rotations
# on the host turns sixty-four device rotates into one [64] table operand
# (`sha256._compress`'s round-constant arrangement).
_T = np.array(
    [_rotl_int(0x79CC4519 if j < 16 else 0x7A879D8A, j) for j in range(64)],
    dtype=np.uint32,
)

_Td = fnp.asarray(_T)

# The SM3 initial state as a device array — the standard start for a digest.
# A module-level device constant, the sha256/sha512 posture (this module
# accepts backend initialization at import, as its named sibling does).
INITIAL_STATE = fnp.asarray(_H0)  # uint32 [8]

# How this family pads, as the axes `extension/md.py` names.
# GB/T 32905 §4.2, byte-for-byte FIPS 180-4 §5.1.1's rule.
_PAD = PadRule(64, Trailer.BIT_LENGTH)


def _p0(x: Array) -> Array:
    """The P0 diffusion permutation (§4.3): X ⊕ ⟪X,9⟫ ⊕ ⟪X,17⟫ — applied to
    TT2 before it lands in E each round."""
    return x ^ rotl(x, 9) ^ rotl(x, 17)


def _p1(x: Array) -> Array:
    """The P1 diffusion permutation (§4.3): X ⊕ ⟪X,15⟫ ⊕ ⟪X,23⟫ — the message
    expansion's mixer."""
    return x ^ rotl(x, 15) ^ rotl(x, 23)


# The boolean functions (§4.2), in both round-16 forms — the ripemd160 `_f*`
# spelling. The first sixteen rounds use XOR for BOTH FF and GG; the rest use
# majority / choose.
def _xor3(x: Array, y: Array, z: Array) -> Array:
    return x ^ y ^ z


def _maj(x: Array, y: Array, z: Array) -> Array:
    return (x & y) | (x & z) | (y & z)


def _ch(x: Array, y: Array, z: Array) -> Array:
    return (x & y) | (~x & z)


_BoolFn = Callable[[Array, Array, Array], Array]


def _compress(state: Array, w16: Array, t: Array) -> Array:
    """One block: state [B, 8] (A..H) + message words w16 [B, 16] -> state
    [B, 8], big-endian words throughout. `t` is the [64] pre-rotated constant
    table (an explicit operand so the marked region passes it in the
    recognizer ABI order and captures nothing).

    The 64 rounds and the message expansion are fused into `fori_loop`s
    carrying a [B, 16] shift-register window: round j uses the oldest word
    `w[:, 0]` (= W_j) and its ⊕-partner `w[:, 4]` (= W_{j+4}, §5.3.2's W'_j),
    appends the freshly-expanded W_{j+16}, and shifts. TWO loops rather than
    one, split at round 16 where FF/GG change form (§4.2) — each body is
    static, so neither form's boolean work is computed and discarded the way
    a per-round select would. Only static column slices index the window; the
    rounds past j = 51 expand words beyond W_67 that no round reads, the same
    uniform-loop tail `sha256._compress` accepts.
    """

    def round_with(ff: _BoolFn, gg: _BoolFn) -> Callable[[Array, tuple], tuple]:
        def round_j(j: Array, carry: tuple) -> tuple:
            a, b, c, d, e, f, g, h, w = carry
            word = w[:, 0]
            wprime = w[:, 0] ^ w[:, 4]
            a12 = rotl(a, 12)
            ss1 = rotl(a12 + e + t[j], 7)
            ss2 = ss1 ^ a12
            tt1 = ff(a, b, c) + d + ss2 + wprime
            tt2 = gg(e, f, g) + h + ss1 + word
            # W_{j+16} = P1(W_j ⊕ W_{j+7} ⊕ ⟪W_{j+13},15⟫) ⊕ ⟪W_{j+3},7⟫
            #            ⊕ W_{j+10}  (§5.3.2)
            nxt = (
                _p1(w[:, 0] ^ w[:, 7] ^ rotl(w[:, 13], 15))
                ^ rotl(w[:, 3], 7)
                ^ w[:, 10]
            )
            w = fnp.concatenate([w[:, 1:], nxt[:, None]], axis=1)
            # §5.3.3's register rotation: D<-C, C<-⟪B,9⟫, B<-A, A<-TT1,
            # H<-G, G<-⟪F,19⟫, F<-E, E<-P0(TT2).
            return (tt1, a, rotl(b, 9), c, _p0(tt2), e, rotl(f, 19), g, w)

        return round_j

    init = (*(state[:, i] for i in range(8)), w16)
    mid = frx.lax.fori_loop(0, 16, round_with(_xor3, _xor3), init)
    out = frx.lax.fori_loop(16, 64, round_with(_maj, _ch), mid)
    # V_{i+1} = ABCDEFGH ⊕ V_i (§5.3.3) — the XOR feedforward, where the
    # SHA-2 family adds.
    return state ^ fnp.stack(out[:8], axis=1)


def block_to_words(blocks: Array) -> Array:
    """uint8 [B, nblocks*64] -> uint32 [B, nblocks, 16] big-endian message
    words (`sha256.block_to_words` at the same parameters)."""
    b = blocks.shape[0]
    return pack_be(blocks.reshape(b, blocks.shape[-1] // _BLOCK, _BLOCK))


def _padded_words(msg: Array) -> Array:
    """A uint8 [B, L] batch padded and packed: uint32 [B, nblocks, 16]. A
    concatenation and a reshape, so it holds a tracer as readily as a
    concrete array (the `sha256._padded_words` arrangement)."""
    tail = fnp.asarray(_PAD.tail(msg.shape[-1]))
    return block_to_words(padded_batch(msg, tail))


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and a batched consumer emits one digest call per tree level or
# transcript squeeze — so the uncached re-trace of the round body would
# dominate the first-trace floor (cf. sha256_bytes and sha512_merkle_damgard).
# `inline=True` splices the cached jaxpr into the enclosing trace, so the
# emitted module (one composite marker per chain) is unchanged.
@partial(frx.jit, inline=True)
def sm3_merkle_damgard(h0: Array, blocks: Array) -> Array:
    """The SM3 compression chain from state `h0` (uint32 [8], shared by the
    batch) over `blocks` (uint32 [B, nblocks, 16]) -> uint8 [B, 32] serialized
    final state, as the name-routed `hash_frx.digest.sm3` composite. SM3 is
    Merkle–Damgård (a 64-round compression, not straight-line), so it takes
    the name-routed marker — exempt from the generic single-kernel rule, the
    way `hash_frx.digest.sha256` is — and no plugin ships an emitter for the
    name yet (`_DEDICATED_EMITTER_AVAILABLE`): today the marker inlines its
    decomposition on every backend — identical bytes, no dedicated kernel —
    and an emitter landing changes the lowering, never the value.

    Operands are explicit in the recognizer's positional ABI order:

    ``[0] h0``     uint32 [8]              — the initial state (§4.1)
    ``[1] t``      uint32 [64]             — the PRE-ROTATED constant table:
                                             row j is ⟪T_j, j mod 32⟫ (§4.2),
                                             the rotation baked on the host
    ``[2] blocks`` uint32 [B, nblocks, 16]  — padded big-endian blocks
                                             (`block_to_words`)

    Passing all three (rather than capturing `_Td`) keeps that order — a
    captured constant would prepend and land at operand 0
    (docs/reference/conventions.md's operand-ABI rule)."""

    return chain(
        h0,
        blocks,
        constants=_Td,
        compress_block=_compress,
        serialize=unpack_be,
        marker=words_in_digest_marker(SM3_MARKER, SM3_MARKER_VERSION),
        primitive="sm3",
    )


def digest(msg: ArrayLike) -> fnp.ndarray:
    """SM3 of a batch of equal-length messages. msg: uint8 [B, L] -> [B, 32].

    Byte-identical to GB/T 32905 per message; the device compression is
    emitted as the one name-routed blocks marker (`sm3_merkle_damgard`) with
    the padded words packed here.

    **Traced or concrete.** `msg` may be a tracer, so a consumer can hash
    inside its own `@jit` or `vmap` without reaching past the `ByteHash`
    seam: the padding is built from the static length and never reads the
    message (`_PAD`), the same property `sha256.digest` states.
    """
    msg = device_message(msg)
    return sm3_merkle_damgard(INITIAL_STATE, _padded_words(msg))


class Sm3(DeviceRow):
    """`ByteHash` for device SM3 — `digest` runs the batch on the
    `hash_frx.digest.sm3` marker. No plugin recognizes that name yet, so
    `fusion_path` reads `GENERIC` on every backend today: the marker inlines,
    the bytes are the standard's, and an emitter landing flips the module
    flags and nothing here moves.

    For batched hashing where the messages already live on the device — a
    ShangMi-ecosystem proof workload verifying many SM2-over-SM3 signatures
    at once. A strictly-sequential caller has no in-package fast path."""

    digest_size = 32

    def __init__(self) -> None:
        super().__init__(FusionPath.from_routing(_routes_to_dedicated_emitter()))

    def digest(self, msg: ArrayLike) -> Array:
        return digest(msg)  # the module-level marker digest


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md).
    _bh_marker: type[ByteHash] = Sm3

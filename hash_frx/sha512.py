# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHA-512 over `(lo, hi)` uint32 half pairs, authored in frx — byte-identical to
the FIPS 180-4 standard (and any conforming implementation, e.g. Python's
`hashlib.sha512`). SHA-256's sibling: the same Merkle–Damgård structure at the
64-bit parameters — 80 rounds, 1024-bit blocks, a 16-byte length field, its own
round constants and initial state (`sha256.py` is the 32-bit original this module
mirrors section for section).

A SHA-512 word is 64 bits, and this toolchain cannot hold one safely: with x64
off `uint64` truncates to `uint32`, and enabling x64 flips the default dtypes
process-wide (`keccak/lane.py` states the law). So every 64-bit word is a pair of
`uint32` halves, and the 64-bit operations the compression needs beyond per-half
bitwise ops — rotate-right, logical shift-right, add-with-carry, the variadic
XOR — are the shared half-pair helpers in `word64.py` (lifted from here when
BLAKE2b became their second consumer).

**The wire layout is the FIPS byte stream packed big-endian into uint32s.** A
64-bit word rides as two uint32s with the HIGH half at the even index — element
`2i` is bits 63..32 of word `i`, element `2i+1` bits 31..0 — because that is what
packing the standard's big-endian bytes four at a time produces
(`block_to_words`). A digest is then exactly the serialized final state with no
cross-word reorder, and the `(lo, hi)` half pairs exist only inside the
compression, unpacked at its boundary.

Bulk-parallel by construction, like `sha256`: a batch of `B` equal-length
messages is hashed in one data-parallel call (the 80 rounds carry a per-message
a..h chain, but every message in the batch advances independently).

Contract: `digest(msg)` takes uint8 `[B, L]` (a batch of `B` messages, each `L`
bytes) and returns uint8 `[B, 64]` digests, big-endian (standard SHA-512 output
order). Length `L` is static, so the padding is data-independent: it is a host
constant built from the length and concatenated on, which is what lets `msg`
itself be traced. Requires no x64; all arithmetic is uint32.

The FIPS 180-4 truncated variants ride the same machinery: `sha384_digest`
(§6.5) and `sha512_256_digest` (§6.7) start the chain from their own §5.3
initial hash values and keep the leftmost 48/32 bytes — a different `h0`
operand on the SAME blocks marker, the truncation a caller-side slice, so no
new wire name exists and a recognizer serving SHA-512 serves both for free.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.tree_util import register_dataclass
from frx.typing import ArrayLike

from hash_frx.byte_hash import host_digest
from hash_frx.fusion import FusionPath, fused_region
from hash_frx.word import pack_be, split, unpack_be
from hash_frx.word64 import Pair, add64, rotr64, xor64

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

SHA512_MARKER = "hash_frx.digest.sha512"
# Marker revision riding as `composite.version`; version 1 is the operand ABI in
# `sha512_merkle_damgard`. XLA recognizes a marker by name + attributes and
# deliberately does not gate on the version, which exists so a contract change
# can stage without a rename once a recognizer ships (`hash_frx.fusion`); an ABI
# change ships as a NEW name, the way `sha256_bytes` did.
SHA512_MARKER_VERSION = 1

# No raw-bytes sibling (`sha256_bytes`) under this family, deliberately: that
# marker exists because the pinned plugin ships a recognizer for it, and no
# plugin claims any SHA-512 name yet — a second wire name with no consumer would
# be ABI surface for nothing (the grostl posture: no marker without a consumer).
# Issue #66's fusion-marker question resolves the same way as the dedicated
# emitter below — on measured evidence, when one is written.

# Whether the pinned Fractalyze XLA plugin ships a dedicated SHA-512 emitter,
# and on which backends. None exists yet — this is the pre-emitter half of the
# keccak arrangement (`keccak.permutation._DEDICATED_EMITTER_AVAILABLE` carries
# the family-wide rationale), the same posture `vision` and `grostl` hold: both
# flags flip together with the `frx>=` floor in `pyproject.toml` when an emitter
# lands, and `fusion_path_test`'s matrix law holds them to agree. The marker is
# emitted regardless — there is no per-block routing alternative for a
# whole-hash digest — and unrecognized it inlines its decomposition: right
# bytes, `GENERIC` fusion path.
_DEDICATED_EMITTER_AVAILABLE = False

# Which backends have that emitter — a different question from the pin, asked
# alongside it. Empty until one is written; a backend gaining an arm joins this
# tuple and nothing else here moves (the keccak/poseidon2 pattern).
_EMITTER_BACKENDS: tuple[str, ...] = ()


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry a SHA-512 emitter. Read per
    construction so importing does not initialize a backend; the lookup behind
    `frx.default_backend()` is memoized."""
    return _DEDICATED_EMITTER_AVAILABLE and frx.default_backend() in _EMITTER_BACKENDS


# Round constants (first 64 bits of the fractional parts of the cube roots of
# the first 80 primes, FIPS 180-4 §4.2.3) and initial hash state (sqrt of the
# first 8 primes, §5.3.5) — as host Python ints, split into uint32 halves below.
_K64 = (
    0x428A2F98D728AE22,
    0x7137449123EF65CD,
    0xB5C0FBCFEC4D3B2F,
    0xE9B5DBA58189DBBC,
    0x3956C25BF348B538,
    0x59F111F1B605D019,
    0x923F82A4AF194F9B,
    0xAB1C5ED5DA6D8118,
    0xD807AA98A3030242,
    0x12835B0145706FBE,
    0x243185BE4EE4B28C,
    0x550C7DC3D5FFB4E2,
    0x72BE5D74F27B896F,
    0x80DEB1FE3B1696B1,
    0x9BDC06A725C71235,
    0xC19BF174CF692694,
    0xE49B69C19EF14AD2,
    0xEFBE4786384F25E3,
    0x0FC19DC68B8CD5B5,
    0x240CA1CC77AC9C65,
    0x2DE92C6F592B0275,
    0x4A7484AA6EA6E483,
    0x5CB0A9DCBD41FBD4,
    0x76F988DA831153B5,
    0x983E5152EE66DFAB,
    0xA831C66D2DB43210,
    0xB00327C898FB213F,
    0xBF597FC7BEEF0EE4,
    0xC6E00BF33DA88FC2,
    0xD5A79147930AA725,
    0x06CA6351E003826F,
    0x142929670A0E6E70,
    0x27B70A8546D22FFC,
    0x2E1B21385C26C926,
    0x4D2C6DFC5AC42AED,
    0x53380D139D95B3DF,
    0x650A73548BAF63DE,
    0x766A0ABB3C77B2A8,
    0x81C2C92E47EDAEE6,
    0x92722C851482353B,
    0xA2BFE8A14CF10364,
    0xA81A664BBC423001,
    0xC24B8B70D0F89791,
    0xC76C51A30654BE30,
    0xD192E819D6EF5218,
    0xD69906245565A910,
    0xF40E35855771202A,
    0x106AA07032BBD1B8,
    0x19A4C116B8D2D0C8,
    0x1E376C085141AB53,
    0x2748774CDF8EEB99,
    0x34B0BCB5E19B48A8,
    0x391C0CB3C5C95A63,
    0x4ED8AA4AE3418ACB,
    0x5B9CCA4F7763E373,
    0x682E6FF3D6B2B8A3,
    0x748F82EE5DEFB2FC,
    0x78A5636F43172F60,
    0x84C87814A1F0AB72,
    0x8CC702081A6439EC,
    0x90BEFFFA23631E28,
    0xA4506CEBDE82BDE9,
    0xBEF9A3F7B2C67915,
    0xC67178F2E372532B,
    0xCA273ECEEA26619C,
    0xD186B8C721C0C207,
    0xEADA7DD6CDE0EB1E,
    0xF57D4F7FEE6ED178,
    0x06F067AA72176FBA,
    0x0A637DC5A2C898A6,
    0x113F9804BEF90DAE,
    0x1B710B35131C471B,
    0x28DB77F523047D84,
    0x32CAAB7B40C72493,
    0x3C9EBE0A15C9BEBC,
    0x431D67C49C100D4C,
    0x4CC5D4BECB3E42B6,
    0x597F299CFC657E2A,
    0x5FCB6FAB3AD6FAEC,
    0x6C44198C4A475817,
)
_H64 = (
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179,
)


def _pairs(values: tuple[int, ...]) -> np.ndarray:
    """64-bit host constants as uint32 in the module's big-endian pair layout —
    element 2i the high half of value i, 2i+1 the low. Split on the host, where
    Python integers are exact (`word.split`), never by materialising a 64-bit
    device value and narrowing it (`keccak/lane.py`)."""
    out = np.empty(2 * len(values), dtype=np.uint32)
    for i, value in enumerate(values):
        lo, hi = split(value)
        out[2 * i] = hi
        out[2 * i + 1] = lo
    return out


_K = _pairs(_K64)  # uint32 [160]
_H0 = _pairs(_H64)  # uint32 [16]

_Kd = fnp.asarray(_K)

# The SHA-512 initial hash state (§5.3.5) as a device array in the pair layout —
# the standard start for a full digest, and the resume point a streaming hash
# broadcasts from.
INITIAL_STATE = fnp.asarray(_H0)  # uint32 [16]

# The truncated variants' initial hash values, transcribed from the same §5.3
# family: SHA-384 (§5.3.4 — sqrt of the 9th..16th primes, the way _H64 is the
# first eight) and SHA-512/256 (§5.3.6 — the SHA-512/t generation function:
# SHA-512 with the ⊕a5…a5 initial state over the ASCII string "SHA-512/256").
# Everything else — padding, schedule, rounds, K — is SHA-512's own (§6.5/§6.7),
# which is why a variant is a different `h0` OPERAND on the same blocks marker
# plus a caller-side output truncation, and no new wire name exists for either.
_H384_64 = (
    0xCBBB9D5DC1059ED8,
    0x629A292A367CD507,
    0x9159015A3070DD17,
    0x152FECD8F70E5939,
    0x67332667FFC00B31,
    0x8EB44A8768581511,
    0xDB0C2E0D64F98FA7,
    0x47B5481DBEFA4FA4,
)
_H512_256_64 = (
    0x22312194FC2BF72C,
    0x9F555FA3C84C64C2,
    0x2393B86B6F53B151,
    0x963877195940EABD,
    0x96283EE2A88EFFE3,
    0xBE5E1E2553863992,
    0x2B0199FC2C85B8AA,
    0x0EB72DDC81C52CA2,
)

INITIAL_STATE_384 = fnp.asarray(_pairs(_H384_64))  # uint32 [16]
INITIAL_STATE_512_256 = fnp.asarray(_pairs(_H512_256_64))  # uint32 [16]


# The 64-bit `(lo, hi)` half-pair helpers — `rotr64`, `add64`, the variadic
# `xor64` and the `Pair` alias — live in `hash_frx.word64`, shared with
# BLAKE2b's G and Ascon's Σ (the literal consumers that met the lift bar these
# helpers carried while module-local; `word64`'s docstring holds the charter).
# The σ shift below stays module-local: no second family reads it.
# `keccak/lane.py`'s rotate stays keccak's: mirrored direction, not the same
# function.


def _shr64(a: Pair, n: int) -> Pair:
    """Logical shift-right of the 64-bit value by `0 < n < 32` — every σ
    shift in §4.1.3 is 6 or 7, so the cross-half cases never arise and the
    bound keeps every uint32 shift defined."""
    lo, hi = a
    s = fnp.uint32(n)
    c = fnp.uint32(32 - n)
    return (lo >> s) | (hi << c), hi >> s


@lru_cache(maxsize=None)
def _padding_tail(length: int) -> np.ndarray:
    """What FIPS 180-4 §5.1.2 appends to a `length`-byte message: uint8 [P].

    `0x80 ‖ 0x00* ‖ toByte(8·length, 16)`, padding to 128-byte blocks — and
    every term is a function of the length alone, so the tail is a host constant
    built *from the length* rather than written *into the message*, which is
    what lets `digest` take a traced message (the `sha256._padding_tail`
    arrangement). The length field is 16 bytes; its high 8 are zero for any
    message below 2^61 bytes, so only the low 8 are ever written.

    Shared by the whole batch, since one call hashes messages of one length.
    """
    nblocks = (length + 16) // 128 + 1  # room for the 0x80 byte + 16-byte length
    tail = np.zeros(nblocks * 128 - length, dtype=np.uint8)
    tail[0] = 0x80
    tail[-8:] = np.frombuffer(
        np.uint64(length * 8).byteswap().tobytes(), dtype=np.uint8
    )
    return tail


def _compress(state: Array, w32: Array, k: Array) -> Array:
    """One block: state [B, 16] (a..h as pairs) + message words w32 [B, 32] ->
    state [B, 16], everything in the module's big-endian pair layout. `k` is the
    [160] round-constant operand (explicit so the marked region passes it in the
    recognizer ABI order and captures nothing).

    `sha256._compress`'s arrangement at the 64-bit parameters: the 80-round
    compression and the message schedule are fused into ONE `fori_loop`, the
    shift-register window riding as a [B, 16] array per half and every 64-bit
    add going through the carry helper. Only *static* column slices index the
    window (round constants are the one traced index, as in sha256), so XLA
    keeps the window + a..h fusion-/register-friendly.
    """
    b = state.shape[0]
    kp = k.reshape(80, 2)  # [t, 0] the high half of K_t, [t, 1] the low
    wp = w32.reshape(b, 16, 2)
    w_lo0, w_hi0 = wp[..., 1], wp[..., 0]
    sp = state.reshape(b, 8, 2)

    def round_t(t: Array, carry: tuple) -> tuple:
        a, bb, c, d, e, f, g, h, w_lo, w_hi = carry
        word: Pair = (w_lo[:, 0], w_hi[:, 0])
        krow = kp[t]  # one row gather; the halves split statically below
        kt: Pair = (krow[1], krow[0])
        # Σ1 = ROTR14 ⊕ ROTR18 ⊕ ROTR41; Ch = (e ∧ f) ⊕ (¬e ∧ g) (§4.1.3).
        s1 = xor64(rotr64(e, 14), rotr64(e, 18), rotr64(e, 41))
        ch: Pair = ((e[0] & f[0]) ^ (~e[0] & g[0]), (e[1] & f[1]) ^ (~e[1] & g[1]))
        t1 = add64(add64(add64(add64(h, s1), ch), kt), word)
        # Σ0 = ROTR28 ⊕ ROTR34 ⊕ ROTR39; Maj = (a∧b) ⊕ (a∧c) ⊕ (b∧c).
        s0 = xor64(rotr64(a, 28), rotr64(a, 34), rotr64(a, 39))
        maj: Pair = (
            (a[0] & bb[0]) ^ (a[0] & c[0]) ^ (bb[0] & c[0]),
            (a[1] & bb[1]) ^ (a[1] & c[1]) ^ (bb[1] & c[1]),
        )
        t2 = add64(s0, maj)
        # Schedule w[t+16] = σ1(w14) + w9 + σ0(w1) + w0, with σ0 = ROTR1 ⊕
        # ROTR8 ⊕ SHR7 and σ1 = ROTR19 ⊕ ROTR61 ⊕ SHR6 (§4.1.3) — the same
        # window indices as SHA-256's schedule.
        w1: Pair = (w_lo[:, 1], w_hi[:, 1])
        w14: Pair = (w_lo[:, 14], w_hi[:, 14])
        sig0 = xor64(rotr64(w1, 1), rotr64(w1, 8), _shr64(w1, 7))
        sig1 = xor64(rotr64(w14, 19), rotr64(w14, 61), _shr64(w14, 6))
        nxt = add64(add64(add64(word, sig0), (w_lo[:, 9], w_hi[:, 9])), sig1)
        w_lo = fnp.concatenate([w_lo[:, 1:], nxt[0][:, None]], axis=1)
        w_hi = fnp.concatenate([w_hi[:, 1:], nxt[1][:, None]], axis=1)
        return (add64(t1, t2), a, bb, c, add64(d, t1), e, f, g, w_lo, w_hi)

    chain0 = [(sp[:, i, 1], sp[:, i, 0]) for i in range(8)]  # a..h as pairs
    out = frx.lax.fori_loop(0, 80, round_t, (*chain0, w_lo0, w_hi0))
    final = []
    for i in range(8):  # h_i' = h_i + var_i, the per-block feedforward (§6.4.2)
        s = add64(chain0[i], out[i])
        final.extend([s[1], s[0]])  # back to the pair layout: hi, then lo
    return fnp.stack(final, axis=1)


def block_to_words(blocks: Array) -> Array:
    """uint8 [B, nblocks*128] -> uint32 [B, nblocks, 32] big-endian words.

    Four bytes big-endian per uint32, so each 64-bit message word's first four
    bytes — its HIGH half, the stream being big-endian — land at the even index
    and the low half at the odd one: the module's pair layout, produced by the
    packing itself. What every path here packs its blocks with, whether they
    came from padding a whole message (`_padded_words`) or from a caller
    building its own blocks incrementally (a streaming hash).
    """
    b = blocks.shape[0]
    return pack_be(blocks.reshape(b, -1, 128))


def _padded_words(msg: Array, tail: Array | None = None) -> Array:
    """A uint8 [B, L] batch padded and packed: uint32 [B, nblocks, 32].

    A concatenation and a reshape, which is the whole point: the message is only
    ever an operand here, never something written into a host buffer, so this
    holds a tracer as readily as a concrete array. `tail` defaults to the
    FIPS 180-4 §5.1.2 padding for L.
    """
    if tail is None:
        tail = fnp.asarray(_padding_tail(msg.shape[-1]))
    return block_to_words(
        fnp.concatenate(
            [msg, fnp.broadcast_to(tail, (msg.shape[0], tail.shape[0]))], axis=-1
        )
    )


def compress(state: Array, blocks_words: Array, k: Array | None = None) -> Array:
    """Fold `blocks_words` (uint32 [B, nblocks, 32] big-endian) into the SHA-512
    midstate `state` (uint32 [B, 16]), block by block. `INITIAL_STATE` broadcast
    is the standard start; a streaming hash resumes from a prior midstate. `k`
    defaults to the module `_Kd`; the marked region passes its `k` operand
    explicitly so it captures nothing."""
    kt = _Kd if k is None else k
    nblocks = blocks_words.shape[1]
    for i in range(nblocks):  # nblocks is static and small
        state = _compress(state, blocks_words[:, i], kt)
    return state


def serialize_digest(state: Array) -> Array:
    """SHA-512 midstate uint32 [B, 16] -> uint8 [B, 64] big-endian digest. The
    pair layout is the big-endian word stream already, so this is the per-uint32
    big-endian byte expansion (`word.unpack_be`) with no cross-word reorder."""
    return unpack_be(state)


def deserialize_digest(digest: Array) -> Array:
    """uint8 [B, 64] big-endian digest -> SHA-512 midstate uint32 [B, 16] — the
    inverse of `serialize_digest`. A digest IS the serialized final midstate
    (the per-block feedforward is inside the compression), so unpacking one
    resumes the stream: a streaming absorb rides the digest-shaped marker and
    reads the next midstate back out."""
    return pack_be(digest)


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits the leaf + every internal level of a Merkle
# commit plus each transcript squeeze — so the uncached re-trace of the 80-round
# body would dominate the first-trace floor (cf. poseidon2._permute_body).
# `inline=True` splices the cached jaxpr into the enclosing trace, so the
# emitted module (one composite marker per chain) is unchanged.
@partial(frx.jit, inline=True)
def sha512_merkle_damgard(h0: Array, blocks: Array) -> Array:
    """The SHA-512 compression chain from midstate `h0` (uint32 [16], shared by
    the batch) over `blocks` (uint32 [B, nblocks, 32]) -> uint8 [B, 64]
    serialized final state, as the name-routed `hash_frx.digest.sha512`
    composite. SHA-512 is Merkle–Damgård (an 80-round compression, not
    straight-line), so it takes the name-routed marker — exempt from the generic
    single-kernel rule, the way `hash_frx.digest.sha256` is — and no plugin
    ships an emitter for the name yet (`_DEDICATED_EMITTER_AVAILABLE`): today
    the marker inlines its decomposition on every backend — identical bytes, no
    dedicated kernel — and an emitter landing changes the lowering, never the
    value.

    `h0 = INITIAL_STATE` is a whole-message digest; any other midstate resumes a
    stream (`deserialize_digest` of the result is the next midstate), so the
    streaming absorb/finalize and the batch digest share this one marker.

    Operands are explicit in the recognizer's positional ABI order, every 64-bit
    quantity in the module's big-endian pair layout (high half at even index):

    ``[0] h0``     uint32 [16]             — the midstate H (§5.3.5 to start)
    ``[1] k``      uint32 [160]            — the K round constants (§4.2.3)
    ``[2] blocks`` uint32 [B, nblocks, 32] — padded blocks (`block_to_words`)

    Passing all three (rather than capturing `_Kd`) keeps that order — a
    captured constant would prepend and land at operand 0
    (docs/reference/conventions.md's operand-ABI rule)."""

    def decomposition(h0: Array, k: Array, blocks: Array, **_attrs: object) -> Array:
        state = fnp.broadcast_to(h0, (blocks.shape[0], 16))
        return serialize_digest(compress(state, blocks, k))

    return fused_region(
        decomposition,
        h0,
        _Kd,
        blocks,
        name=SHA512_MARKER,
        version=SHA512_MARKER_VERSION,
    )


def digest(msg: ArrayLike) -> fnp.ndarray:
    """SHA-512 of a batch of equal-length messages. msg: uint8 [B, L] -> [B, 64].

    Byte-identical to the FIPS 180-4 standard per message. The device
    compression is emitted as the one name-routed blocks marker
    (`sha512_merkle_damgard`) with the padded words packed here; there is no
    raw-bytes sibling to route to (the module-level absence note).

    **Traced or concrete.** `msg` may be a tracer, so a consumer can hash inside
    its own `@jit` or `vmap` without reaching past the seam for
    `sha512_merkle_damgard` — which would make it name SHA-512. The padding is
    built from the static length and never reads the message (`_padding_tail`),
    which is the same property `sha256.digest` states.
    """
    msg = fnp.asarray(msg, dtype=fnp.uint8)
    return sha512_merkle_damgard(INITIAL_STATE, _padded_words(msg))


def sha384_digest(msg: ArrayLike) -> fnp.ndarray:
    """SHA-384 (FIPS 180-4 §6.5) of a batch: uint8 [B, L] -> [B, 48].

    SHA-512's compression chain from the §5.3.4 initial state, keeping the
    leftmost 384 bits — the variant differs in nothing else, so it rides the
    same `hash_frx.digest.sha512` blocks marker with its IV as the `h0`
    operand, and the truncation is a caller-side slice OUTSIDE the marker (the
    marker's wire contract is untouched; a recognizer serving SHA-512 serves
    this for free, which is what `h0`-as-operand was designed to buy). Traced
    or concrete, like `digest` — and since `h0`'s AVAL is the same uint32 [16]
    for every variant, all three share one trace of the chain per message
    shape."""
    msg = fnp.asarray(msg, dtype=fnp.uint8)
    return sha512_merkle_damgard(INITIAL_STATE_384, _padded_words(msg))[:, :48]


def sha512_256_digest(msg: ArrayLike) -> fnp.ndarray:
    """SHA-512/256 (FIPS 180-4 §6.7) of a batch: uint8 [B, L] -> [B, 32].

    The §5.3.6 initial state on the same blocks marker, truncated to 256 bits
    by the caller-side slice — `sha384_digest`'s arrangement at the other
    §5.3 table. The 64-bit-word road to a length-extension-safe 256-bit
    digest (truncation hides the final state, unlike SHA-512 itself)."""
    msg = fnp.asarray(msg, dtype=fnp.uint8)
    return sha512_merkle_damgard(INITIAL_STATE_512_256, _padded_words(msg))[:, :32]


# ---------------------------------------------------------------------------
# Streaming Merkle–Damgård midstate (the fixed-shape, scan-threadable core).
#
# `digest` above pads a whole message once on host; this keeps SHA-512's
# incremental state so a byte Fiat-Shamir transcript can thread `@jit` / a
# `lax.scan` carry — `sha256`'s streaming section at the 64-bit parameters
# (128-byte blocks, a 16-byte length field). The midstate is over every COMPLETE
# 128-byte block, plus the (<128 B) trailing partial block and the running byte
# length — all fixed shapes. A squeeze `SHA512(buffer ‖ ctr)` is
# `finalize(state, ctr)`: a non-mutating copy that pads at the current length,
# reproducing `digest`'s bytes incrementally.
# ---------------------------------------------------------------------------

_BLOCK = 128  # SHA-512 block size in bytes


@register_dataclass
@dataclass(frozen=True)
class Sha512State:
    """Incremental SHA-512 state as an FRX pytree. Fixed shapes → scan-threadable.

    The two byte counters share one `int32[2]` leaf rather than riding as two
    scalar fields, for the reason measured on `Sha256State`: every absorb
    updates both, and as separate output/carry leaves each cost their own
    scalar kernel per state hand-off."""

    h: Array  # uint32[16] — midstate over all complete 128-byte blocks so far
    pending: Array  # uint8[128] — trailing partial block, valid prefix [:counts[0]]
    counts: Array  # int32[2] = [pending_len (0..127), total bytes absorbed]

    @property
    def pending_len(self) -> Array:
        return self.counts[0]

    @property
    def total_len(self) -> Array:
        return self.counts[1]


def sha512_stream_init() -> Sha512State:
    """A fresh incremental hash (no bytes absorbed)."""
    return Sha512State(
        h=INITIAL_STATE,
        pending=fnp.zeros(_BLOCK, dtype=fnp.uint8),
        counts=fnp.zeros(2, dtype=fnp.int32),
    )


def sha512_stream_absorb(state: Sha512State, data: Array) -> Sha512State:
    """Absorb `data` (uint8 [L], L static) into the incremental hash: fold every
    newly-complete 128-byte block into the midstate, keep the `<128 B` remainder
    as the new pending block. The block loop is a Python-unrolled,
    active-count-masked schedule over STATIC slices (never a traced-index gather
    / scan-carry scatter) — `sha256_stream_absorb`'s pattern."""
    length = data.shape[0]
    pl = state.pending_len
    combined_src = fnp.concatenate([state.pending, data.astype(fnp.uint8)])  # [128+L]
    new_len = pl + fnp.int32(length)
    active_blocks = new_len // _BLOCK
    max_blocks = (_BLOCK - 1 + length) // _BLOCK  # static upper bound

    # Drop the pending buffer's invalid gap [pending_len:128] from the stream:
    # for stream position j, source index is j while j < pending_len, else
    # shifted to skip past the gap.
    total_slots = (max_blocks + 1) * _BLOCK
    pos = fnp.arange(total_slots, dtype=fnp.int32)
    src_idx = pos + fnp.where(pos < pl, fnp.int32(0), _BLOCK - pl)
    src_idx = fnp.clip(src_idx, 0, combined_src.shape[0] - 1)
    combined = combined_src[src_idx]  # [total_slots], valid prefix [0:new_len]

    # Fold the newly-complete blocks through the marked chain. The live block
    # count depends on pending_len by AT MOST one — (pl + L) // 128 spans
    # {L // 128, (127 + L) // 128} — so run the chain at both static candidate
    # counts and select; the discarded candidate is the only one that ever sees
    # the gap-shifted junk tail block.
    h = state.h
    min_blocks = length // _BLOCK
    if max_blocks == 0:
        h_new = h
    else:
        words = block_to_words(
            combined[: max_blocks * _BLOCK].reshape(1, max_blocks * _BLOCK)
        )  # [1, max_blocks, 32]
        h_hi = deserialize_digest(sha512_merkle_damgard(h, words))[0]
        if min_blocks == max_blocks:
            h_new = h_hi
        else:
            h_lo = (
                deserialize_digest(sha512_merkle_damgard(h, words[:, :min_blocks]))[0]
                if min_blocks > 0
                else h
            )
            h_new = fnp.where(active_blocks == max_blocks, h_hi, h_lo)

    tail_len = new_len - active_blocks * _BLOCK
    tail = frx.lax.dynamic_slice(combined, (active_blocks * _BLOCK,), (_BLOCK,))
    slot = fnp.arange(_BLOCK, dtype=fnp.int32)
    pending = fnp.where(slot < tail_len, tail, fnp.uint8(0))
    # One fused counter update: [pending_len', total_len'] = [tail_len, total+L].
    counts = fnp.stack([tail_len, state.counts[1] + fnp.int32(length)])
    return Sha512State(h_new, pending, counts)


def sha512_stream_finalize(state: Sha512State, extras: Array) -> Array:
    """`SHA512(absorbed ‖ extras[b])` for each row of `extras` (uint8 [B, E], E
    static) — a non-mutating copy of the hash finished at the current length.
    One call finishes a whole batch of counter blocks (the transcript's
    counter-mode squeeze) sharing the base state. Returns uint8 [B, 64]
    big-endian digests.

    The trailing content is `pending[:pending_len] ‖ extras[b]` (≤ 127 + E
    bytes), so with the `0x80` byte and the 16-byte length it spans at most two
    blocks; the second block is compressed unconditionally and selected away
    when one suffices.
    """
    batch, e = extras.shape
    pl = state.pending_len
    content_len = pl + fnp.int32(e)
    msg_bytes = state.total_len + fnp.int32(e)
    # SHA-512's 128-bit length field (§5.1.2). The high 64 bits are zero
    # outright; the low 64 ride as a uint32 half pair off the int32 byte count —
    # the bit length reaches 2^34 for a count near 2^31, so the low half alone
    # (sha256's arrangement) would not do.
    count = msg_bytes.astype(fnp.uint32)
    bit_lo = count << fnp.uint32(3)
    bit_hi = count >> fnp.uint32(29)
    len_bytes = fnp.concatenate(
        [fnp.zeros(8, dtype=fnp.uint8), unpack_be(fnp.stack([bit_hi, bit_lo]))]
    )  # [16] big-endian

    # Need a 2nd block for pad + length? One block holds ≤ 128 − 17 content.
    two_blocks = content_len > fnp.int32(_BLOCK - 17)
    active_bytes = fnp.where(two_blocks, fnp.int32(2 * _BLOCK), fnp.int32(_BLOCK))

    pos = fnp.arange(2 * _BLOCK, dtype=fnp.int32)
    # content = pending[:pl] ‖ extras[b], skipping the pending gap [pl:128].
    combined_src = fnp.concatenate(
        [fnp.broadcast_to(state.pending, (batch, _BLOCK)), extras.astype(fnp.uint8)],
        axis=1,
    )  # [B, 128+E]
    src_idx = fnp.clip(
        pos + fnp.where(pos < pl, fnp.int32(0), _BLOCK - pl), 0, _BLOCK + e - 1
    )
    content = combined_src[:, src_idx]  # [B, 256]

    is_content = (pos < content_len)[None, :]
    is_pad80 = (pos == content_len)[None, :]
    len_start = active_bytes - fnp.int32(16)
    is_len = ((pos >= len_start) & (pos < active_bytes))[None, :]
    len_val = len_bytes[fnp.clip(pos - len_start, 0, 15)][None, :]
    region = fnp.where(
        is_content,
        content,
        fnp.where(is_pad80, fnp.uint8(0x80), fnp.where(is_len, len_val, fnp.uint8(0))),
    )  # [B, 256]

    # Both finalize shapes ride the marked chain from the shared midstate; the
    # 1-vs-2-block choice is data-dependent, so emit both and select.
    words = block_to_words(region)  # [B, 2, 32]
    d2 = sha512_merkle_damgard(state.h, words)
    d1 = sha512_merkle_damgard(state.h, words[:, :1])
    return fnp.where(two_blocks, d2, d1)


# ---------------------------------------------------------------------------
# ByteHash seam implementations (SHA-512). Both hash to the identical FIPS 180-4
# bytes and differ only in substrate — `fusion_path` is the type-level signal.
# Param-free, so value identity is by type (no jit re-trace). The split names
# the same workloads as the SHA-256 pair: batched device hashing vs a
# strictly-sequential host caller (the crossover measured there; this family is
# un-fused today, so its device batch case starts from the GENERIC path).
# ---------------------------------------------------------------------------
class Sha512:
    """`ByteHash` for device SHA-512 — `digest` runs the batch on the
    `hash_frx.digest.sha512` marker. No plugin recognizes that name yet, so
    `fusion_path` reads `GENERIC` on every backend today: the marker inlines,
    the bytes are the standard's, and an emitter landing flips the module flags
    and nothing here moves.

    For batched hashing where the messages already live on the device — the
    SLH-DSA category-3/5 `H`/`T_l` verification batches being the motivating
    case (issue #66). Wrong choice for a caller that hashes one short message at
    a time and reads the result on the host."""

    digest_size = 64

    def __init__(self) -> None:
        # Read per instance rather than pinned on the class: the emitter switch
        # is a property of the pin and the backend, and a value read at import
        # would pin the answer before anything could vary it.
        self.fusion_path = FusionPath.from_routing(_routes_to_dedicated_emitter())

    def digest(self, msg: ArrayLike) -> Array:
        return digest(msg)  # the module-level marker digest above

    def __eq__(self, other: object) -> bool:
        # By type, because SHA-512 is parameterless — the `Sha256` form, stated
        # there: `type(other) is not type(self)` rather than `isinstance`,
        # which is asymmetric under subclassing and blocks Python's
        # reflected-`__eq__` fallback.
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))


class HostSha512:
    """`ByteHash` for host SHA-512 — `digest` loops `hashlib` per message on the
    host (eager, no device kernel), so `fusion_path = HOST`. The fast path for a
    strictly-sequential byte challenger, and the signing-path row issue #66
    anticipates: `PRF_msg` carries no performance claim, and `hashlib` on a
    small buffer beats a device dispatch per call.

    Shipped rather than testonly, like `HostSha256` and unlike the grostl
    partner: `hashlib` implements SHA-512, so the host row is a real fast path
    and not merely the differential oracle."""

    digest_size = 64
    # The one legitimate class constant of the taxonomy: a host row is HOST on
    # every backend, so nothing here varies with the pin.
    fusion_path = FusionPath.HOST

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(
            lambda row: hashlib.sha512(row).digest(), self.digest_size, msg
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))


# ---------------------------------------------------------------------------
# The truncated variants' rows (§6.5 SHA-384, §6.7 SHA-512/256) — the same
# four-way split as above, one device + one host row per variant. Each class is
# param-free (the output length is the VARIANT, fixed by its IV table, not a
# parameter a caller picks — the SHAKE contrast), so value identity is by type
# like `Sha512`'s, and the distinct types are what keep a family holding
# several rows from colliding.
# ---------------------------------------------------------------------------
class Sha384:
    """`ByteHash` for device SHA-384 — `sha384_digest`, i.e. the
    `hash_frx.digest.sha512` marker from the §5.3.4 initial state with the
    48-byte truncation outside. The fusion story is therefore `Sha512`'s,
    read off the same module flags: GENERIC everywhere today, and the day a
    sha512 emitter lands this row flips with it, for free."""

    digest_size = 48

    def __init__(self) -> None:
        # Per instance, not per class — the `Sha512.__init__` rationale.
        self.fusion_path = FusionPath.from_routing(_routes_to_dedicated_emitter())

    def digest(self, msg: ArrayLike) -> Array:
        return sha384_digest(msg)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))


class Sha512_256:
    """`ByteHash` for device SHA-512/256 — `sha512_256_digest` on the shared
    marker, the §5.3.6 initial state, the 32-byte truncation outside. The
    64-bit-word answer to a length-extension-safe 256-bit digest."""

    digest_size = 32

    def __init__(self) -> None:
        self.fusion_path = FusionPath.from_routing(_routes_to_dedicated_emitter())

    def digest(self, msg: ArrayLike) -> Array:
        return sha512_256_digest(msg)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))


class HostSha384:
    """`ByteHash` for host SHA-384 over `hashlib.sha384` — a guaranteed
    `hashlib` constructor, so the row ships unconditionally like
    `HostSha512`: the fast path for a strictly-sequential byte caller (TLS
    transcripts, JWT PS384/ES384 verification one signature at a time)."""

    digest_size = 48
    fusion_path = FusionPath.HOST

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(
            lambda row: hashlib.sha384(row).digest(), self.digest_size, msg
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))


class HostSha512_256:
    """`ByteHash` for host SHA-512/256 over `hashlib.new("sha512_256")`.

    Not a guaranteed constructor: the name reaches `hashlib` through OpenSSL,
    which carries SHA-512/256 in its DEFAULT provider (probed 2026-08-21 —
    the opposite of RIPEMD-160's legacy-provider retreat that forced that
    family's host partner testonly), so the row ships; on an OpenSSL built
    without it, `hashlib.new` raises its own unsupported-algorithm error at
    the first digest."""

    digest_size = 32
    fusion_path = FusionPath.HOST

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(
            lambda row: hashlib.new("sha512_256", row).digest(),
            self.digest_size,
            msg,
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md).
    _bh_marker: type[ByteHash] = Sha512
    _bh_host: type[ByteHash] = HostSha512
    _bh_384: type[ByteHash] = Sha384
    _bh_host_384: type[ByteHash] = HostSha384
    _bh_512_256: type[ByteHash] = Sha512_256
    _bh_host_512_256: type[ByteHash] = HostSha512_256

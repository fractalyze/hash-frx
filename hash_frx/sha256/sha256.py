# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHA-256 over uint32 lanes, authored in frx — byte-identical to the FIPS 180-4
standard (and any conforming implementation, e.g. Python's `hashlib.sha256`).

Bulk-parallel by construction: a batch of `B` equal-length messages is hashed in
one data-parallel call (the 64 rounds carry a per-message a..h chain, but every
message in the batch advances independently). That maps the many-independent-hash
workloads — Merkle leaf/internal levels, batched proof-of-work grinding — onto a
GPU's width. A byte hash, unlike the algebraic `Permutation`s in this package
(Poseidon2/Poseidon), so it is a standalone primitive rather than a `Permutation`.

SHA-224 rides here too, as `sha224_digest` / `Sha224`: the same chain from the
§5.3.2 initial state with the output truncated outside the marker, which is what
`sha512.py` does for SHA-384 and the two SHA-512/t rows.

Contract: `digest(msg)` takes uint8 `[B, L]` (a batch of `B` messages, each `L`
bytes) and returns uint8 `[B, 32]` digests, big-endian (standard SHA-256 output
order). Length `L` is static, so the padding is data-independent: it is a host
constant built from the length and concatenated on, which is what lets `msg`
itself be traced. Requires no x64; all arithmetic is uint32 (wraps mod 2^32 in
XLA).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.tree_util import register_dataclass
from frx.typing import ArrayLike

from hash_frx.byte_hash import (
    DeviceRow,
    at_capacity,
    capacity,
    device_message,
    message_length,
    padded_batch,
    require_capacity_buffer,
)
from hash_frx.extension.md import (
    MdStream,
    chain,
    masked_chain,
    padded_message_region,
)
from hash_frx.extension.pad import PadRule, Trailer
from hash_frx.fusion import FusionPath, fused_region, routing
from hash_frx.word import pack_be, rotr, unpack_be

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

U32 = fnp.uint32

SHA256_MARKER = "hash_frx.digest.sha256"
# Marker revision riding as `composite.version`. XLA recognizes the marker by
# name + attributes and deliberately does not gate on the version; it lets a
# future contract change be staged without renaming the marker (cf. POSEIDON2).
# That staging only covers attribute additions: the pinned recognizer hard-fails
# on an operand-ABI mismatch under this name rather than declining, so an ABI
# change ships as a NEW name (below), which an old plugin declines and inlines.
SHA256_MARKER_VERSION = 1

SHA256_BYTES_MARKER = "hash_frx.digest.sha256_bytes"
# The raw-bytes whole-message form:
#
#   [h0 u32[8], k u32[64], msg u8[B, LMAX], len s32[]] -> u8[B, 32]
#
# FIPS 180-4 padding and word packing live inside the marker. The blocks
# marker above takes pre-padded words, so every consumer runs a pad-and-pack
# pass over the whole batch first — at a Merkle-leaf-scale batch
# (2^20 x 1 KiB) that pass writes and re-reads ~1.1 GB a recognizing emitter
# can instead synthesize in registers.
#
# `LMAX` is the buffer's CAPACITY and `len` says how much of it is live, which
# is what collapses the compile count: one kernel serves every length the buffer
# can hold, where a length carried in the message SHAPE makes each one a fresh
# module (`sha256_bytes` says why the spare capacity is free).
#
# `len` is ONE scalar for the whole batch, deliberately: the CPU emitter packs
# 16 digest rows into vector lanes, and a per-row length would stop them
# agreeing on the block count and on every padding byte. The recognizer rejects
# a non-scalar `len` rather than silently taking row 0's, so a per-row ABI stays
# a separate decision rather than an accident.
#
# The recognizer dispatches on the OPERANDS rather than on the name, so this
# name also claims a whole-message ABI that this module does not emit
# (`markers.py` carries that fact). What follows from it here: the version below
# covers both, so a contract change under this name is one decision for the
# pair. The `frx>=` floor in pyproject.toml names the first
# wheel whose recognizer claims the form emitted here; an older one declines it
# on its operands and inlines — right bytes, no kernel.
SHA256_BYTES_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships the dedicated SHA-256 emitter,
# and on which backends (`allow_sha256_fusion` is set in both the CPU and the
# GPU compiler). Two flags with the family-wide rationale in
# `keccak.permutation`; a backend absent from the tuple — Metal today — still
# emits the marker (there is no per-block routing alternative for a whole-hash
# digest), which inlines unrecognized: right bytes, `GENERIC` fusion path.
_DEDICATED_EMITTER_AVAILABLE = True
_EMITTER_BACKENDS = ("cpu", "gpu")

# Whether the pinned plugin claims the whole-message marker, and on which
# backends. Its own flag and its own backend tuple rather than the pair above,
# because acceptance is per-MARKER and not per-family: Fractalyze XLA gates this
# one on `allow_sha256_bytes_len_fusion`, separate from the `allow_sha256_fusion`
# both compilers set for the blocks marker, so a backend can carry one and
# decline the other. A backend that declines gets neither a kernel nor the
# compile saving — worse, the decomposition speculates every block the buffer
# could need — so `digest` falls back to the blocks form there instead of
# emitting into a decline.
_BYTES_EMITTER_AVAILABLE = True
_BYTES_EMITTER_BACKENDS = ("cpu", "gpu")


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry this emitter
    (`fusion.routing`, which carries the rationale)."""
    return routing(_DEDICATED_EMITTER_AVAILABLE, _EMITTER_BACKENDS)


def _routes_to_bytes_marker() -> bool:
    """Whether `digest` should emit the whole-message marker on this backend
    (`fusion.routing`; `_BYTES_EMITTER_AVAILABLE` says why this is a separate
    question from the blocks marker's own switch)."""
    return routing(_BYTES_EMITTER_AVAILABLE, _BYTES_EMITTER_BACKENDS)


# Round constants (first 32 bits of the fractional parts of the cube roots of the
# first 64 primes) and initial hash state (sqrt of first 8 primes).
_K = np.array(
    [
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
        0xD807AA98,
        0x12835B01,
        0x243185BE,
        0x550C7DC3,
        0x72BE5D74,
        0x80DEB1FE,
        0x9BDC06A7,
        0xC19BF174,
        0xE49B69C1,
        0xEFBE4786,
        0x0FC19DC6,
        0x240CA1CC,
        0x2DE92C6F,
        0x4A7484AA,
        0x5CB0A9DC,
        0x76F988DA,
        0x983E5152,
        0xA831C66D,
        0xB00327C8,
        0xBF597FC7,
        0xC6E00BF3,
        0xD5A79147,
        0x06CA6351,
        0x14292967,
        0x27B70A85,
        0x2E1B2138,
        0x4D2C6DFC,
        0x53380D13,
        0x650A7354,
        0x766A0ABB,
        0x81C2C92E,
        0x92722C85,
        0xA2BFE8A1,
        0xA81A664B,
        0xC24B8B70,
        0xC76C51A3,
        0xD192E819,
        0xD6990624,
        0xF40E3585,
        0x106AA070,
        0x19A4C116,
        0x1E376C08,
        0x2748774C,
        0x34B0BCB5,
        0x391C0CB3,
        0x4ED8AA4A,
        0x5B9CCA4F,
        0x682E6FF3,
        0x748F82EE,
        0x78A5636F,
        0x84C87814,
        0x8CC70208,
        0x90BEFFFA,
        0xA4506CEB,
        0xBEF9A3F7,
        0xC67178F2,
    ],
    dtype=np.uint32,
)
_H0 = np.array(
    [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ],
    dtype=np.uint32,
)

# SHA-224's initial hash value (FIPS 180-4 §5.3.2). Not a truncation of
# SHA-256's: both tables are fractional parts of prime square roots, but
# SHA-256 takes the first eight primes and this takes the ninth to sixteenth —
# the same eight SHA-384 does, at the other 32 bits. `sha512._H384_64` is
# therefore this table in its high halves, which is a relationship a test can
# assert and a mistyped word cannot survive.
_H224 = np.array(
    [
        0xC1059ED8,
        0x367CD507,
        0x3070DD17,
        0xF70E5939,
        0xFFC00B31,
        0x68581511,
        0x64F98FA7,
        0xBEFA4FA4,
    ],
    dtype=np.uint32,
)


_Kd = fnp.asarray(_K)

# How this family pads, as the axes `extension/md.py` names.
# FIPS 180-4 §5.1.1.
_PAD = PadRule(64, Trailer.BIT_LENGTH)


def _compress(state: Array, w16: Array, k: Array) -> Array:
    """One block: state [B, 8] (a..h) + message words w16 [B, 16] -> state [B, 8].
    `k` is the [64] round-constant table (an explicit operand so the marked
    region passes it in the recognizer ABI order and captures nothing).

    The 64-round compression and the message schedule are fused into ONE
    `fori_loop` carrying a [B, 16] shift-register window: round t uses the oldest
    word `w[:,0]`, appends the freshly-scheduled `w[t+16]`, and shifts. The
    window is read by *static* column slices only, so XLA keeps it and a..h
    fusion-/register-friendly — critical for GPU throughput. The one dynamic
    index is `k[t]` into the round-constant operand, which the name-routed
    emitter reads directly and the generic single-kernel rule exempts (a
    name-routed body may loop and gather; see the fusion contract).
    """

    def round_t(t: Array, carry: tuple) -> tuple:
        a, b, c, d, e, f, g, h, w = carry
        word = w[:, 0]
        kt = k[t]
        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = h + S1 + ch + kt + word
        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = S0 + maj
        # schedule next word w[t+16] = sigma1(w14) + w9 + sigma0(w1) + w0
        s0 = rotr(w[:, 1], 7) ^ rotr(w[:, 1], 18) ^ (w[:, 1] >> U32(3))
        s1 = rotr(w[:, 14], 17) ^ rotr(w[:, 14], 19) ^ (w[:, 14] >> U32(10))
        nxt = w[:, 0] + s0 + w[:, 9] + s1
        w = fnp.concatenate([w[:, 1:], nxt[:, None]], axis=1)
        return (t1 + t2, a, b, c, d + t1, e, f, g, w)

    init = (*(state[:, i] for i in range(8)), w16)
    a, b, c, d, e, f, g, h, _ = frx.lax.fori_loop(0, 64, round_t, init)
    return state + fnp.stack([a, b, c, d, e, f, g, h], axis=1)


# The SHA-256 initial hash state (sqrt of the first 8 primes) as a device array —
# the standard start for a full digest, and the resume point a streaming hash
# broadcasts from.
INITIAL_STATE = fnp.asarray(_H0)  # uint32 [8]

# SHA-224's start (§5.3.2). Everything after it — padding, schedule, rounds, K —
# is SHA-256's own, which is why the variant is a different `h0` OPERAND on the
# shared blocks marker plus a caller-side truncation, and no new wire name
# exists for it.
INITIAL_STATE_224 = fnp.asarray(_H224)  # uint32 [8]


def block_to_words(blocks: Array) -> Array:
    """uint8 [B, nblocks*64] -> uint32 [B, nblocks, 16] big-endian message words.

    What every path here packs its blocks with, whether they came from padding a
    whole message (`_padded_words`) or from a caller building its own blocks
    incrementally (a streaming hash).
    """
    b = blocks.shape[0]
    return pack_be(blocks.reshape(b, blocks.shape[-1] // 64, 64))


def _padded_words(msg: Array) -> Array:
    """A uint8 [B, L] batch padded and packed: uint32 [B, nblocks, 16].

    A concatenation and a reshape, which is the whole point: the message is only
    ever an operand here, never something written into a host buffer, so this
    holds a tracer as readily as a concrete array. The tail is the FIPS 180-4
    padding for L, a host constant because L is static.
    """
    tail = fnp.asarray(_PAD.tail(msg.shape[-1]))
    return block_to_words(padded_batch(msg, tail))


def _runtime_padded_words(msg: Array, length: Array) -> Array:
    """The first `length` bytes of each row, padded and packed: uint8 [B, LMAX]
    plus an int32 scalar -> uint32 [B, NB, 16].

    What `_padded_words` builds from a static length, built from a traced one.
    The region is `md.padded_message_region`'s — shared with every other
    runtime-length family — and what stays here is this family's own big-endian
    packing of it, which is the half that is genuinely SHA-256's.
    """
    return block_to_words(padded_message_region(_PAD, msg, length))


def compress(state: Array, blocks_words: Array, k: Array | None = None) -> Array:
    """Fold `blocks_words` (uint32 [B, nblocks, 16] big-endian) into the SHA-256
    midstate `state` (uint32 [B, 8]), block by block. `INITIAL_STATE` broadcast is
    the standard start; a streaming hash resumes from a prior midstate. `k` is
    a parameter rather than a capture, so a caller inside a marked region can
    thread that region's own `k` operand; it defaults to the module `_Kd`."""
    kt = _Kd if k is None else k
    nblocks = blocks_words.shape[1]
    for i in range(nblocks):  # nblocks is static and small
        state = _compress(state, blocks_words[:, i], kt)
    return state


def serialize_digest(state: Array) -> Array:
    """SHA-256 midstate uint32 [B, 8] -> uint8 [B, 32] big-endian digest."""
    return unpack_be(state)


def deserialize_digest(digest: Array) -> Array:
    """uint8 [B, 32] big-endian digest -> SHA-256 midstate uint32 [B, 8] — the
    inverse of `serialize_digest`. A digest IS the serialized final midstate (the
    per-block feedforward is inside the compression), so unpacking one resumes
    the stream: a streaming absorb rides the digest-shaped marker and reads the
    next midstate back out."""
    return pack_be(digest)


# The compression's key in the plugin's primitive registry. Two markers
# carry it -- this family's chain, and the stream-finalize region that wraps
# it -- and they must agree, or the outer would claim a different compression
# than the one it folds.
_PRIMITIVE = "sha256"


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits the leaf + every internal level of a Merkle
# commit plus each transcript squeeze — so the uncached re-trace of the 64-round
# body would dominate the first-trace floor (cf. poseidon2._permute_body).
# `inline=True` splices the cached jaxpr into the enclosing trace, so the emitted
# module (one composite marker per chain) is unchanged.
@partial(frx.jit, inline=True)
def sha256_merkle_damgard(h0: Array, blocks: Array, constants: Array = _Kd) -> Array:
    """The SHA-256 compression chain from midstate `h0` (uint32 [8], shared by
    the batch) over `blocks` (uint32 [B, nblocks, 16]) -> uint8 [B, 32]
    serialized final state, as the name-routed `hash_frx.digest.sha256`
    composite. SHA-256 is Merkle-Damgard (a 64-round compression, not
    straight-line), so it takes the name-routed marker (exempt from the generic
    single-kernel rule, the way `hash_frx.perm.poseidon2` is) and routes to the
    dedicated Sha256Fusion emitter; with no emitter wired the marker inlines its
    decomposition, so the bytes are unchanged.

    `h0 = INITIAL_STATE` is a whole-message digest; any other midstate resumes a
    stream (`deserialize_digest` of the result is the next midstate), so the
    streaming transcript and the batch digest share this one marker. Operands are
    explicit in the recognizer's positional ABI order [h0, k, blocks]; passing
    all three (rather than capturing `_Kd`) keeps that order — a captured
    constant would prepend and land at operand 0."""

    return chain(
        h0,
        blocks,
        constants=constants,
        compress_block=_compress,
        serialize=serialize_digest,
        marker=(SHA256_MARKER, SHA256_MARKER_VERSION),
        primitive=_PRIMITIVE,
    )


# Module-level jit zone for the same reason as `sha256_merkle_damgard`: the
# composite re-traces its decomposition per emission, and a Merkle commit emits
# one digest call per tree level.
@partial(frx.jit, inline=True)
def sha256_bytes(buf: Array, length: Array) -> Array:
    """Whole-message SHA-256 over the live prefix of a capacity buffer, as ONE
    marked region: uint8 [B, LMAX] with an int32 scalar byte count -> uint8
    [B, 32].

    The digest is of `buf[:, :length]`; `LMAX` sizes the allocation and nothing
    else, since the emitter loops on `length` and never reads past it. That is
    the whole point of the form — one compiled kernel serves every length the
    buffer can hold, where a length carried in the message shape compiles once
    per length.

    `length` rides as an int32 SCALAR — `np.int32`, not a Python int and not a
    device array. `jit` keys on an operand's aval rather than its value, so any
    of the three compiles once; the two rejected spellings differ elsewhere. A
    Python int is weakly typed, so an x64 build would widen it to `s64[]`, which
    the recognizer declines — inlining the decomposition for right bytes and no
    kernel, the silent mode this package's floor exists to prevent. A device
    scalar pins the type but costs a transfer per call, measured at ~14 us
    against the ~3 us digest it accompanies.

    It is one scalar for the whole batch (`SHA256_BYTES_MARKER` says why), so
    every row of `buf` is hashed at the same length.

    Whole-message only: a resumed stream pads at its running total rather than
    at `length`, so the streaming path stays on the blocks marker.

    Operands are explicit in the recognizer's positional ABI order
    [h0, k, msg, len]: a captured constant would prepend and land at operand 0,
    which is not where the recognizer reads it.
    """
    require_capacity_buffer(buf)

    def decomposition(
        h0: Array, k: Array, msg: Array, ln: Array, **_attrs: object
    ) -> Array:
        words = _runtime_padded_words(msg, ln)
        live = _PAD.nblocks(ln)
        state = fnp.broadcast_to(h0, (msg.shape[0], 8))
        state = masked_chain(
            state,
            count=words.shape[1],
            compress_block=lambda s, i: _compress(s, words[:, i], k),
            live=live,
        )
        return serialize_digest(state)

    return fused_region(
        decomposition,
        INITIAL_STATE,
        _Kd,
        buf,
        length,
        name=SHA256_BYTES_MARKER,
        version=SHA256_BYTES_MARKER_VERSION,
    )


def digest(msg: ArrayLike) -> fnp.ndarray:
    """SHA-256 of a batch of equal-length messages. msg: uint8 [B, L] -> [B, 32].

    Byte-identical to the FIPS 180-4 standard per message. The device
    compression is emitted as one name-routed marker; which one depends on the
    pinned plugin. The whole-message form (`sha256_bytes`) where its
    recognizer exists, hashing out of a `byte_hash.capacity` buffer so that a
    host caller whose lengths vary compiles once per width rather than once per
    length; else the blocks form with the padded words packed here. Both produce
    identical bytes, and differ only in where the padding runs and in what the
    compilation is keyed on.

    **Traced or concrete.** `msg` may be a tracer, so a consumer can hash inside
    its own `@jit` or `vmap` without reaching past the seam for
    `sha256_merkle_damgard` — which would make it name SHA-256. The padding is
    what used to prevent that: it wrote `0x80` and the bit length *into* a host
    buffer holding the message, so the message had to be concrete. Built from the
    length instead, it never reads the message at all.
    """
    if _routes_to_bytes_marker():
        length = message_length(msg)
        buf = at_capacity(msg, capacity(msg, _PAD.block_size))
        return sha256_bytes(buf, np.int32(length))
    return sha256_merkle_damgard(INITIAL_STATE, _padded_words(device_message(msg)))


def sha224_digest(msg: ArrayLike) -> fnp.ndarray:
    """SHA-224 (FIPS 180-4 §6.3) of a batch: uint8 [B, L] -> [B, 28].

    SHA-256's compression chain from the §5.3.2 initial state, keeping the
    leftmost 224 bits — the variant differs in nothing else, so it rides the
    same `hash_frx.digest.sha256` blocks marker with its IV as the `h0` operand
    and the truncation is a caller-side slice OUTSIDE the marker. A recognizer
    serving SHA-256 serves this for free, which is what `h0`-as-operand buys.
    Traced or concrete, like `digest`.

    Only the blocks form: `sha256_bytes` carries `INITIAL_STATE` as operand 0
    of its own region, so the whole-message marker is SHA-256's by construction
    and a variant cannot ride it. `digest`'s routing branch has no counterpart
    here.
    """
    msg = device_message(msg)
    return sha256_merkle_damgard(INITIAL_STATE_224, _padded_words(msg))[:, :28]


# ---------------------------------------------------------------------------
# Streaming Merkle–Damgård midstate (the fixed-shape, scan-threadable core).
#
# `digest` above pads a whole message once on host; this keeps SHA-256's
# incremental state so a byte Fiat-Shamir transcript can thread `@jit` / a
# `lax.scan` carry (`Sha256FieldTranscript`, `sha256_field_transcript.py`). The
# midstate is over every COMPLETE 64-byte block, plus the (<64 B) trailing partial
# block and the running byte length — all fixed shapes. A squeeze
# `SHA256(buffer ‖ ctr)` is `finalize(state, ctr_le8)`: a non-mutating copy that
# pads at the current length, reproducing `digest`'s bytes incrementally.
# ---------------------------------------------------------------------------

_BLOCK = 64  # SHA-256 block size in bytes


@register_dataclass
@dataclass(frozen=True)
class Sha256State:
    """Incremental SHA-256 state as an FRX pytree. Fixed shapes → scan-threadable.

    The two byte counters share one `int32[2]` leaf rather than riding as two
    scalar fields: every absorb updates both, and as separate output/carry
    leaves each cost their own scalar kernel per state hand-off — measured as
    two of the ~12 launches of a single transcript squeeze, and one per
    iteration inside every FS `while_loop` carry."""

    h: Array  # uint32[8] — midstate over all complete 64-byte blocks so far
    pending: Array  # uint8[64] — trailing partial block, valid prefix [:counts[0]]
    counts: Array  # int32[2] = [pending_len (0..63), total bytes absorbed]

    @property
    def pending_len(self) -> Array:
        return self.counts[0]

    @property
    def total_len(self) -> Array:
        return self.counts[1]


def sha256_stream_init() -> Sha256State:
    """A fresh incremental hash (no bytes absorbed)."""
    return Sha256State(
        h=INITIAL_STATE,
        pending=fnp.zeros(_BLOCK, dtype=fnp.uint8),
        counts=fnp.zeros(2, dtype=fnp.int32),
    )


def sha256_stream_absorb(state: Sha256State, data: Array) -> Sha256State:
    """Absorb `data` (uint8 [L], L static) into the incremental hash: fold every
    newly-complete 64-byte block into the midstate, keep the `<64 B` remainder as
    the new pending block. The block loop is a Python-unrolled, active-count-masked
    schedule over STATIC slices (never a traced-index gather / scan-carry scatter)
    — the CPU-safe pattern `transcript.DuplexTranscript` uses."""

    return _STREAM.absorb(state, data)


# The incremental schedule, with this family's pieces plugged in. `chain` is
# the same marked region `digest` runs, so the streaming path and the batch
# digest go through ONE marker rather than two.
_STREAM = MdStream(
    pad=_PAD,
    block_to_words=block_to_words,
    deserialize=deserialize_digest,
    chain=sha256_merkle_damgard,
    constants=_Kd,
    primitive=_PRIMITIVE,
    make_state=Sha256State,
)


def sha256_stream_finalize(state: Sha256State, extras: Array) -> Array:
    """`SHA256(absorbed ‖ extras[b])` for each row of `extras` (uint8 [B, E], E
    static) — a non-mutating copy of the hash finished at the current length. One
    call finishes a whole batch of counter blocks (the transcript's counter-mode
    squeeze) sharing the base state. Returns uint8 [B, 32] big-endian digests.

    The trailing content is `pending[:pending_len] ‖ extras[b]` (≤ 63 + E bytes),
    so with the `0x80` byte and the 8-byte length it spans at most two blocks; the
    second block is compressed unconditionally and selected away when one suffices.

    `E` is bounded by that layout: `pending_len` is runtime data, so only
    `E <= 56` (the block minus the length field) fits two blocks at every
    reachable `pending_len` — a wider tail is rejected rather than silently
    overlapping the padding. Absorb the prefix and finalize with the remainder.

    The stream counts bytes in an int32, which wraps past 2 GiB — benignly, as
    it happens: only the length field reads the count, and it reinterprets the
    wrapped value as a uint32, recovering the right bit pattern. So the real
    ceiling is 4 GiB (2^32 bytes), where the uint32 itself wraps and two
    lengths encode alike. Far below FIPS 180-4's 2^61 bytes, and widening the
    counter rides the batch-polymorphic state rework on the redesign epic.
    """

    return _STREAM.finalize(state, extras)


# ---------------------------------------------------------------------------
# The ByteHash seam implementation (SHA-256). Param-free, so value identity is
# by type (no jit re-trace). What a strictly-sequential caller pays for reaching
# it rather than `hashlib` is measured in `docs/blocks/hash.md`.
# ---------------------------------------------------------------------------
class Sha256(DeviceRow):
    """`ByteHash` for device SHA-256 — `digest` runs the batch on the
    `hash_frx.digest.sha256` marker (data-parallel, lowers to one dedicated kernel
    where the pin and the backend route the name, so `fusion_path` reads
    `DEDICATED` there and `GENERIC` elsewhere).

    For batched hashing where the messages already live on the device — Merkle
    leaves being the motivating case. Wrong choice for a caller that hashes one
    short message at a time and reads the result on the host."""

    digest_size = 32

    def __init__(self) -> None:
        super().__init__(FusionPath.from_routing(_routes_to_dedicated_emitter()))

    def digest(self, msg: ArrayLike) -> Array:
        return digest(msg)  # the module-level marker digest above


class Sha224(DeviceRow):
    """`ByteHash` for device SHA-224 — `sha224_digest`, i.e. the
    `hash_frx.digest.sha256` marker from the §5.3.2 initial state with the
    28-byte truncation outside. The fusion story is therefore `Sha256`'s, read
    off the same module flags: this row routes wherever SHA-256 routes, for
    free, because the marker it emits carries the same name."""

    digest_size = 28

    def __init__(self) -> None:
        super().__init__(FusionPath.from_routing(_routes_to_dedicated_emitter()))

    def digest(self, msg: ArrayLike) -> Array:
        return sha224_digest(msg)


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md).
    _bh_marker: type[ByteHash] = Sha256
    _bh_224: type[ByteHash] = Sha224

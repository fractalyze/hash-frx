# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHA-256 over uint32 lanes, authored in frx — byte-identical to the FIPS 180-4
standard (and any conforming implementation, e.g. Python's `hashlib.sha256`).

Bulk-parallel by construction: a batch of `B` equal-length messages is hashed in
one data-parallel call (the 64 rounds carry a per-message a..h chain, but every
message in the batch advances independently). That maps the many-independent-hash
workloads — Merkle leaf/internal levels, batched proof-of-work grinding — onto a
GPU's width. A byte hash, unlike the algebraic `Permutation`s in this package
(Poseidon2/Poseidon), so it is a standalone primitive rather than a `Permutation`.

Contract: `digest(msg)` takes uint8 `[B, L]` (a batch of `B` messages, each `L`
bytes) and returns uint8 `[B, 32]` digests, big-endian (standard SHA-256 output
order). Length `L` is static, so the padding is data-independent: it is a host
constant built from the length and concatenated on, which is what lets `msg`
itself be traced. Requires no x64; all arithmetic is uint32 (wraps mod 2^32 in
XLA).
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
# The raw-bytes whole-message form: operands [h0 u32[8], k u32[64],
# msg u8[B, L], tail u8[P]] -> u8[B, 32], with FIPS 180-4 padding and word
# packing inside the marker. The blocks marker above takes pre-padded words,
# so every consumer runs a pad-and-pack pass over the whole batch first —
# at a Merkle-leaf-scale batch (2^20 x 1 KiB) that pass writes and re-reads
# ~1.1 GB a recognizing emitter can instead synthesize in registers, the
# padding being a function of the static length.
SHA256_BYTES_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships the dedicated SHA-256 emitter,
# and on which backends (`allow_sha256_fusion` is set in both the CPU and the
# GPU compiler). Two flags with the family-wide rationale in
# `keccak.permutation`; a backend absent from the tuple — Metal today — still
# emits the marker (there is no per-block routing alternative for a whole-hash
# digest), which inlines unrecognized: right bytes, `GENERIC` fusion path.
_DEDICATED_EMITTER_AVAILABLE = True
_EMITTER_BACKENDS = ("cpu", "gpu")

# Whether the pinned plugin claims `hash_frx.digest.sha256_bytes`. A plugin
# without that recognizer declines the name and inlines the decomposition —
# right bytes, `GENERIC` path — which at digest scale is the slow path, so
# `digest` only routes to the bytes marker where a recognizer exists. True
# since the pyproject floor: that wheel's emitter pads and packs in-register
# from the bytes operand, retiring the padded-words materialization.
_BYTES_EMITTER_AVAILABLE = True


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry the SHA-256 emitter. Read
    per construction so importing does not initialize a backend; the lookup
    behind `frx.default_backend()` is memoized."""
    return _DEDICATED_EMITTER_AVAILABLE and frx.default_backend() in _EMITTER_BACKENDS


def _routes_to_bytes_marker() -> bool:
    """Whether `digest` should emit the raw-bytes marker on this backend."""
    return _BYTES_EMITTER_AVAILABLE and frx.default_backend() in _EMITTER_BACKENDS


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


_Kd = fnp.asarray(_K)


@lru_cache(maxsize=None)
def _padding_tail(length: int) -> np.ndarray:
    """What FIPS 180-4 §5.1.1 appends to a `length`-byte message: uint8 [P].

    `0x80 ‖ 0x00* ‖ toByte(8·length, 8)`, and every term is a function of the
    length alone. So the tail is a host constant built *from the length* rather
    than written *into the message* — which is what lets `digest` take a traced
    message: the padding never has to read one.

    Shared by the whole batch, since one call hashes messages of one length.
    """
    nblocks = (length + 8) // 64 + 1  # room for the 0x80 byte + 8-byte length
    tail = np.zeros(nblocks * 64 - length, dtype=np.uint8)
    tail[0] = 0x80
    tail[-8:] = np.frombuffer(
        np.uint64(length * 8).byteswap().tobytes(), dtype=np.uint8
    )
    return tail


def _compress(state: Array, w16: Array, k: Array) -> Array:
    """One block: state [B, 8] (a..h) + message words w16 [B, 16] -> state [B, 8].
    `k` is the [64] round-constant table (an explicit operand so the marked
    region passes it in the recognizer ABI order and captures nothing).

    The 64-round compression and the message schedule are fused into ONE
    `fori_loop` carrying a [B, 16] shift-register window: round t uses the oldest
    word `w[:,0]`, appends the freshly-scheduled `w[t+16]`, and shifts. Only
    *static* column slices are used (no dynamic array indexing), so XLA keeps the
    window + a..h fusion-/register-friendly — critical for GPU throughput.
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


def block_to_words(blocks: Array) -> Array:
    """uint8 [B, nblocks*64] -> uint32 [B, nblocks, 16] big-endian message words.

    What every path here packs its blocks with, whether they came from padding a
    whole message (`_padded_words`) or from a caller building its own blocks
    incrementally (a streaming hash).
    """
    b = blocks.shape[0]
    return pack_be(blocks.reshape(b, -1, 64))


def _padded_words(msg: Array, tail: Array | None = None) -> Array:
    """A uint8 [B, L] batch padded and packed: uint32 [B, nblocks, 16].

    A concatenation and a reshape, which is the whole point: the message is only
    ever an operand here, never something written into a host buffer, so this
    holds a tracer as readily as a concrete array. `tail` defaults to the
    FIPS 180-4 padding for L; `sha256_bytes` passes its tail operand explicitly
    so its marked region captures nothing.
    """
    if tail is None:
        tail = fnp.asarray(_padding_tail(msg.shape[-1]))
    return block_to_words(
        fnp.concatenate(
            [msg, fnp.broadcast_to(tail, (msg.shape[0], tail.shape[0]))], axis=-1
        )
    )


def compress(state: Array, blocks_words: Array, k: Array | None = None) -> Array:
    """Fold `blocks_words` (uint32 [B, nblocks, 16] big-endian) into the SHA-256
    midstate `state` (uint32 [B, 8]), block by block. `INITIAL_STATE` broadcast is
    the standard start; a streaming hash resumes from a prior midstate. `k`
    defaults to the module `_Kd`; the marked region passes its `k` operand
    explicitly so it captures nothing."""
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


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits the leaf + every internal level of a Merkle
# commit plus each transcript squeeze — so the uncached re-trace of the 64-round
# body would dominate the first-trace floor (cf. poseidon2._permute_body).
# `inline=True` splices the cached jaxpr into the enclosing trace, so the emitted
# module (one composite marker per chain) is unchanged.
@partial(frx.jit, inline=True)
def sha256_merkle_damgard(h0: Array, blocks: Array) -> Array:
    """The SHA-256 compression chain from midstate `h0` (uint32 [8], shared by
    the batch) over `blocks` (uint32 [B, nblocks, 16]) -> uint8 [B, 32]
    serialized final state, as the name-routed `hash_frx.sha256` composite. SHA-256
    is Merkle-Damgard (a 64-round compression, not straight-line), so it takes the
    name-routed marker (exempt from the generic single-kernel rule, the way
    `hash_frx.poseidon2` is) and routes to the dedicated Sha256Fusion emitter; with
    no emitter wired the marker inlines its decomposition, so the bytes are
    unchanged.

    `h0 = INITIAL_STATE` is a whole-message digest; any other midstate resumes a
    stream (`deserialize_digest` of the result is the next midstate), so the
    streaming transcript and the batch digest share this one marker. Operands are
    explicit in the recognizer's positional ABI order [h0, k, blocks]; passing
    all three (rather than capturing `_Kd`) keeps that order — a captured
    constant would prepend and land at operand 0."""

    def decomposition(h0: Array, k: Array, blocks: Array, **_attrs: object) -> Array:
        state = fnp.broadcast_to(h0, (blocks.shape[0], 8))
        return serialize_digest(compress(state, blocks, k))

    return fused_region(
        decomposition,
        h0,
        _Kd,
        blocks,
        name=SHA256_MARKER,
        version=SHA256_MARKER_VERSION,
    )


# Module-level jit zone for the same reason as `sha256_merkle_damgard`: the
# composite re-traces its decomposition per emission, and a Merkle commit
# emits one digest call per tree level.
@partial(frx.jit, inline=True)
def sha256_bytes(msg: Array) -> Array:
    """Whole-message SHA-256 over raw bytes as ONE marked region:
    uint8 [B, L] -> uint8 [B, 32], padding and word packing inside the marker.

    The blocks marker (`sha256_merkle_damgard`) takes pre-padded words, so its
    consumers pack the padded word layout as ordinary graph ops first — a
    materialized pass over the whole batch that a recognizing emitter can do
    in registers instead, the padding being a function of the static length.
    This form hands the emitter the raw bytes; where no recognizer is wired
    the marker inlines its decomposition, so the bytes are unchanged.

    Whole-message only: a resumed stream pads at its running total rather
    than at `L`, so the streaming path stays on the blocks marker.

    Operands are explicit in the recognizer's positional ABI order
    [h0, k, msg, tail]. `tail` is `_padding_tail(L)` passed as an operand
    rather than captured (a captured constant would prepend at operand 0);
    it is derivable from the static L — a recognizing emitter reads it
    rather than re-deriving it — and load-bearing for the inlined
    decomposition."""

    def decomposition(
        h0: Array, k: Array, msg: Array, tail: Array, **_attrs: object
    ) -> Array:
        state = fnp.broadcast_to(h0, (msg.shape[0], 8))
        return serialize_digest(compress(state, _padded_words(msg, tail), k))

    return fused_region(
        decomposition,
        INITIAL_STATE,
        _Kd,
        msg,
        fnp.asarray(_padding_tail(msg.shape[-1])),
        name=SHA256_BYTES_MARKER,
        version=SHA256_BYTES_MARKER_VERSION,
    )


def digest(msg: ArrayLike) -> fnp.ndarray:
    """SHA-256 of a batch of equal-length messages. msg: uint8 [B, L] -> [B, 32].

    Byte-identical to the FIPS 180-4 standard per message. The device
    compression is emitted as one name-routed marker; which one depends on the
    pinned plugin: the raw-bytes form (`sha256_bytes`) where its recognizer
    exists, else the blocks form with the padded words packed here — the two
    produce identical bytes, split only by where the padding runs.

    **Traced or concrete.** `msg` may be a tracer, so a consumer can hash inside
    its own `@jit` or `vmap` without reaching past the seam for
    `sha256_merkle_damgard` — which would make it name SHA-256. The padding is
    what used to prevent that: it wrote `0x80` and the bit length *into* a host
    buffer holding the message, so the message had to be concrete. Built from the
    length instead, it never reads the message at all.
    """
    msg = fnp.asarray(msg, dtype=fnp.uint8)
    if _routes_to_bytes_marker():
        return sha256_bytes(msg)
    return sha256_merkle_damgard(INITIAL_STATE, _padded_words(msg))


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
    length = data.shape[0]
    pl = state.pending_len
    combined_src = fnp.concatenate([state.pending, data.astype(fnp.uint8)])  # [64+L]
    new_len = pl + fnp.int32(length)
    active_blocks = new_len // _BLOCK
    max_blocks = (_BLOCK - 1 + length) // _BLOCK  # static upper bound

    # Drop the pending buffer's invalid gap [pending_len:64] from the stream: for
    # stream position j, source index is j while j < pending_len, else shifted to
    # skip past the gap.
    total_slots = (max_blocks + 1) * _BLOCK
    pos = fnp.arange(total_slots, dtype=fnp.int32)
    src_idx = pos + fnp.where(pos < pl, fnp.int32(0), _BLOCK - pl)
    src_idx = fnp.clip(src_idx, 0, combined_src.shape[0] - 1)
    combined = combined_src[src_idx]  # [total_slots], valid prefix [0:new_len]

    # Fold the newly-complete blocks through the marked chain. The live block
    # count depends on pending_len by AT MOST one — (pl + L) // 64 spans
    # {L // 64, (63 + L) // 64} — so run the chain at both static candidate
    # counts and select; the discarded candidate is the only one that ever sees
    # the gap-shifted junk tail block.
    h = state.h
    min_blocks = length // _BLOCK
    if max_blocks == 0:
        h_new = h
    else:
        words = block_to_words(
            combined[: max_blocks * _BLOCK].reshape(1, max_blocks * _BLOCK)
        )  # [1, max_blocks, 16]
        h_hi = deserialize_digest(sha256_merkle_damgard(h, words))[0]
        if min_blocks == max_blocks:
            h_new = h_hi
        else:
            h_lo = (
                deserialize_digest(sha256_merkle_damgard(h, words[:, :min_blocks]))[0]
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
    return Sha256State(h_new, pending, counts)


def sha256_stream_finalize(state: Sha256State, extras: Array) -> Array:
    """`SHA256(absorbed ‖ extras[b])` for each row of `extras` (uint8 [B, E], E
    static) — a non-mutating copy of the hash finished at the current length. One
    call finishes a whole batch of counter blocks (the transcript's counter-mode
    squeeze) sharing the base state. Returns uint8 [B, 32] big-endian digests.

    The trailing content is `pending[:pending_len] ‖ extras[b]` (≤ 63 + E bytes),
    so with the `0x80` byte and the 8-byte length it spans at most two blocks; the
    second block is compressed unconditionally and selected away when one suffices.
    """
    batch, e = extras.shape
    pl = state.pending_len
    content_len = pl + fnp.int32(e)
    msg_bytes = state.total_len + fnp.int32(e)
    # SHA-256's 64-bit length field; the high 32 bits are zero for any message
    # below 2**32 bits (512 MiB) — so the length is a uint32 and no x64 is needed.
    bitlen = msg_bytes.astype(fnp.uint32) * fnp.uint32(8)
    len_bytes = unpack_be(fnp.stack([fnp.uint32(0), bitlen]))  # [8] big-endian

    two_blocks = content_len > fnp.int32(55)  # need a 2nd block for pad + length?
    active_bytes = fnp.where(two_blocks, fnp.int32(128), fnp.int32(64))

    pos = fnp.arange(128, dtype=fnp.int32)
    # content = pending[:pl] ‖ extras[b], skipping the pending gap [pl:64].
    combined_src = fnp.concatenate(
        [fnp.broadcast_to(state.pending, (batch, _BLOCK)), extras.astype(fnp.uint8)],
        axis=1,
    )  # [B, 64+E]
    src_idx = fnp.clip(
        pos + fnp.where(pos < pl, fnp.int32(0), _BLOCK - pl), 0, _BLOCK + e - 1
    )
    content = combined_src[:, src_idx]  # [B, 128]

    is_content = (pos < content_len)[None, :]
    is_pad80 = (pos == content_len)[None, :]
    len_start = active_bytes - fnp.int32(8)
    is_len = ((pos >= len_start) & (pos < active_bytes))[None, :]
    len_val = len_bytes[fnp.clip(pos - len_start, 0, 7)][None, :]
    region = fnp.where(
        is_content,
        content,
        fnp.where(is_pad80, fnp.uint8(0x80), fnp.where(is_len, len_val, fnp.uint8(0))),
    )  # [B, 128]

    # Both finalize shapes ride the marked chain from the shared midstate; the
    # 1-vs-2-block choice is data-dependent, so emit both and select.
    words = block_to_words(region)  # [B, 2, 16]
    d2 = sha256_merkle_damgard(state.h, words)
    d1 = sha256_merkle_damgard(state.h, words[:, :1])
    return fnp.where(two_blocks, d2, d1)


# ---------------------------------------------------------------------------
# ByteHash seam implementations (SHA-256). Both hash to the identical FIPS 180-4
# bytes and differ only in substrate — `fusion_path` is the type-level signal.
# Param-free, so value identity is by type (no jit re-trace).
#
# Two implementations of one function is a cost, so each names the workload it
# exists for. The split is by batch shape, not by machine: device latency is flat
# out to B=64 because it is dispatch-bound, and the crossover sits near B=48 on
# the CPU backend too. A sequential caller pays roughly 22x at B=1 for choosing
# the device path; a batched one pays roughly 5.5x at B=1024 for choosing the
# host path.
# ---------------------------------------------------------------------------
class Sha256:
    """`ByteHash` for device SHA-256 — `digest` runs the batch on the
    `hash_frx.sha256` marker (data-parallel, lowers to one dedicated kernel
    where the pin and the backend route the name, so `fusion_path` reads
    `DEDICATED` there and `GENERIC` elsewhere).

    For batched hashing where the messages already live on the device — Merkle
    leaves being the motivating case. Wrong choice for a caller that hashes one
    short message at a time and reads the result on the host."""

    digest_size = 32

    def __init__(self) -> None:
        # Read per instance rather than pinned on the class: the emitter switch
        # is a property of the pin and the backend, and a value read at import
        # would pin the answer before anything could vary it.
        self.fusion_path = FusionPath.from_routing(_routes_to_dedicated_emitter())

    def digest(self, msg: ArrayLike) -> Array:
        return digest(msg)  # the module-level marker digest above

    def __eq__(self, other: object) -> bool:
        # By type, because SHA-256 is parameterless — but `type(other) is not
        # type(self)` rather than `isinstance`, which is asymmetric under
        # subclassing and blocks Python's reflected-`__eq__` fallback. Every
        # `ByteHash` agrees on this form; the `Permutation` side spells the
        # parameterless case with `isinstance`.
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))


class HostSha256:
    """`ByteHash` for host SHA-256 — `digest` loops `hashlib` per message on the
    host (eager, no device kernel), so `fusion_path = HOST`. The fast path for a
    strictly-sequential byte challenger: `hashlib` on a small buffer beats a
    device dispatch per squeeze.

    For a host-shaped byte transcript, which holds a `bytes` buffer and reads
    each digest back immediately — so a device hash there buys a kernel and pays
    a device-to-host sync per squeeze. It is also the instance that makes the
    `HOST` path reachable, which consumers branch on: zorch's byte transcript
    grinds a wide nonce window against a fused hash and one nonce at a time
    against this one."""

    digest_size = 32
    # The one legitimate class constant of the taxonomy: a host row is HOST on
    # every backend, so nothing here varies with the pin.
    fusion_path = FusionPath.HOST

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(
            lambda row: hashlib.sha256(row).digest(), self.digest_size, msg
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md).
    _bh_marker: type[ByteHash] = Sha256
    _bh_host: type[ByteHash] = HostSha256

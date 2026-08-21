# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Unkeyed BLAKE2b over `(lo, hi)` uint32 half pairs, authored in frx —
byte-identical to RFC 7693 (and any conforming implementation, e.g. Python's
`hashlib.blake2b`). The host row `byte_hashes.HostBlake2b` shipped first;
its docstring carries the family taxonomy (HAIFA: Merkle–Damgård plus a byte
counter and a finalization flag threaded into every compression), why the
family is a `ByteHash` rather than a `Permutation`, and why the host row led.

A BLAKE2b word is 64 bits, so every word rides as a pair of `uint32` halves
(`keccak/lane.py` states the toolchain law), and the compression's arithmetic
is the shared half-pair layer `word64.py` — BLAKE2b's G is the second
consumer that layer was lifted for, its `n = 32` rotation landing on the
three-case rotate's pure half swap. Inside a compression the sixteen working
words ride as four 4-lane ROW grids, the standard's own 4×4 matrix reading
(`_compress` states why the grid, and not 32 loose half lanes, is the shape
that survives lowering — the `ascon._permutation` lesson at this family's
parameters).

**Everything is little-endian** (RFC 7693 §2.4): blocks pack with the shared
`word.pack_le`, and a 64-bit word rides as two uint32s with the LOW half at
the even index — element `2i` is bits 31..0 of word `i`, element `2i+1` bits
63..32 — because that is what packing the standard's little-endian bytes four
at a time produces. The mirror image of `sha512`'s big-endian pair layout,
each produced by its own standard's byte order. A digest is then exactly the
serialized final state truncated to `digest_size`, with no cross-word
reorder.

**The counter schedule is host integers, not operands.** The length `L` is
static, so the byte offset `t` at every block (§3.2: the bytes hashed through
the end of that block — a full 128 per interior block, the true length on the
final one, zero pad never counted) and the final-block flag are known at
trace time; both fold into the IV words `v[12]`/`v[14]` XOR on the host and
enter the body as uint32 scalar literals. The split is exact for any `t`
below 2^64 (`word.split` on an exact Python int), and §3.2's high offset word
`v[13]` XORs `t >> 64` — identically zero until a message passes 2^64 bytes,
so it is not emitted.

Bulk-parallel by construction, like `sha256`: a batch of `B` equal-length
messages is hashed in one data-parallel call, every message advancing
independently through the shared schedule.

Contract: `digest(msg, digest_size)` takes uint8 `[B, L]` and returns uint8
`[B, digest_size]` little-endian digests (RFC 7693 §3.3: the first
`digest_size` bytes of the serialized final state). `L` is static, so the
zero pad is data-independent (`_padding_tail`), which is what lets `msg`
itself be traced. `digest_size` is 1..64 and rides the VALUE surface — RFC
7693 folds it into the initial state through the parameter block, so
`Blake2b(digest_size=32)` is a different hash from `Blake2b(64)`, not one
hash asked for fewer bytes (the SHAKE/BLAKE3 rule; the param-free by-type
equality of `Sha256`/`Ripemd160` does not apply). Requires no x64; all
arithmetic is uint32.

Keyed hashing, salting and the tree parameters are RFC 7693 / [BLAKE2]
features this module does not yet carry — each is a value-surface addition
riding the same marker (the key would prepend a block and flip `kk` in the
parameter word), added when a consumer needs it rather than speculatively.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.blake2b.byte_hashes import MAX_DIGEST_SIZE
from hash_frx.fusion import FusionPath, fused_region
from hash_frx.word import pack_le, roll, split, unpack_le
from hash_frx.word64 import Pair, add64, rotr64, xor64

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

U32 = fnp.uint32

BLAKE2B_MARKER = "hash_frx.digest.blake2b"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# in `blake2b_bytes`. XLA recognizes a marker by name + attributes and
# deliberately does not gate on the version, which exists so a contract change
# can stage without a rename once a recognizer ships (`hash_frx.fusion`); an
# ABI change ships as a NEW name, the way `sha256_bytes` did.
BLAKE2B_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships a dedicated BLAKE2b emitter,
# and on which backends. None exists yet — the pre-emitter half of the keccak
# arrangement (`keccak.permutation._DEDICATED_EMITTER_AVAILABLE` carries the
# family-wide rationale), the same posture `vision`, `grostl`, `ascon` and
# `ripemd160` hold: both flags flip together with the `frx>=` floor in
# `pyproject.toml` when an emitter lands, and `fusion_path_test`'s matrix law
# holds them to agree. The marker is emitted regardless — there is no
# per-block routing alternative for a whole-hash digest — and unrecognized it
# inlines its decomposition: right bytes, `GENERIC` fusion path.
_DEDICATED_EMITTER_AVAILABLE = False

# Which backends have that emitter — a different question from the pin, asked
# alongside it. Empty until one is written; a backend gaining an arm joins
# this tuple and nothing else here moves (the keccak/poseidon2 pattern).
_EMITTER_BACKENDS: tuple[str, ...] = ()


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry a BLAKE2b emitter. Read
    per construction so importing does not initialize a backend; the lookup
    behind `frx.default_backend()` is memoized."""
    return _DEDICATED_EMITTER_AVAILABLE and frx.default_backend() in _EMITTER_BACKENDS


_BLOCK = 128  # RFC 7693 §2.1: bb = 128-byte blocks (16 words of 64 bits)

# Initialization vector (RFC 7693 §2.6: IV[i] = floor(2^64 · frac(sqrt of the
# i+1-th prime)) — the SHA-512 IV, as §2.6 itself notes). Restated rather than
# imported from `sha512`: the constants are the standard's own, and reaching
# for a sibling family's private table would couple the modules for eight
# literals a reviewer checks against §2.6 either way.
_IV64 = (
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179,
)

# Message word schedule (RFC 7693 §2.7), one row per round of sixteen G
# inputs; BLAKE2b's rounds 10 and 11 reuse rows 0 and 1 (`i mod 10`). Static
# Python tuples deliberately: the unrolled round loop reads them at trace
# time, so a schedule entry becomes a static column slice of the message
# words rather than an indexed read of a table array (the no-gather rule,
# docs/reference/conventions.md).
_SIGMA = (
    # fmt: off
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
    (11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4),
    (7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8),
    (9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13),
    (2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9),
    (12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11),
    (13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10),
    (6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5),
    (10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0),
    # fmt: on
)


def _pairs_le(values: tuple[int, ...] | list[int]) -> np.ndarray:
    """64-bit host constants as uint32 in the module's little-endian pair
    layout — element 2i the LOW half of value i, 2i+1 the high
    (`sha512._pairs` mirrored). Split on the host, where Python integers are
    exact (`word.split`), never by materialising a 64-bit device value and
    narrowing it (`keccak/lane.py`)."""
    out = np.empty(2 * len(values), dtype=np.uint32)
    for i, value in enumerate(values):
        lo, hi = split(value)
        out[2 * i] = lo
        out[2 * i + 1] = hi
    return out


# The §2.6 IV as uint32 [16] in the pair layout — the constant operand every
# compression reads for v[8..15]. Threaded as an explicit operand, the
# `sha256.compress` round-constant-table remedy: eight 64-bit values have no
# index structure an in-body `iota` could count out, and a host-built array
# captured by the body would lift into an unnamed operand ahead of the
# declared ABI. Held as HOST numpy here and `fnp.asarray`ed at each use —
# a module-level device constant would initialize a backend at import
# (`sha512` accepts that; this family's import stays light), and caching the
# device array lazily is a trap: built first inside a trace it would cache a
# TRACER, which leaks into every later call.
_IV_PAIRS = _pairs_le(_IV64)


@lru_cache(maxsize=None)
def _initial_state(digest_size: int) -> np.ndarray:
    """The initial state h for an unkeyed BLAKE2b-`digest_size` hash: the IV
    with the parameter-block word XORed into h[0] (RFC 7693 §2.5/§3.3:
    p[0] = 0x0101kknn — nn the hash size in bytes, kk = 0 unkeyed, bytes 2
    and 3 set as 01), as HOST uint32 [16] in the module's little-endian pair
    layout (`fnp.asarray`ed at use, for the reason `_IV_PAIRS` states).
    Built on the host, where Python ints are exact (`word.split`), and
    cached per size: there are at most 64 of them. This is where the digest
    length enters the hash — which is why truncating a longer digest is the
    WRONG bytes at every shorter length, and why the range check lives here
    on the module path as well as on the row's constructor."""
    if not 1 <= digest_size <= MAX_DIGEST_SIZE:
        raise ValueError(
            f"digest_size must be in 1..{MAX_DIGEST_SIZE} (RFC 7693), got "
            f"{digest_size}"
        )
    words = list(_IV64)
    words[0] ^= 0x01010000 ^ digest_size
    return _pairs_le(words)


@lru_cache(maxsize=None)
def _padding_tail(length: int) -> np.ndarray:
    """What RFC 7693 §3.3 appends to a `length`-byte message: uint8 [P] of
    zeros, P = (−length) mod 128 — the final block zero-padded to the block
    boundary, plus the special case that an unkeyed empty message still
    processes one all-zero block (dd = 1). Unlike the SHA-2/RIPEMD tails
    there is no 0x80 marker and no length field: the length enters through
    the `t` counter instead (HAIFA), which is what makes the tail all zeros —
    and empty at a block multiple, where the operand rides zero-length. A
    function of the static length alone, so `digest` can take a traced
    message (the `sha256._padding_tail` arrangement)."""
    if length == 0:
        return np.zeros(_BLOCK, dtype=np.uint8)
    return np.zeros((-length) % _BLOCK, dtype=np.uint8)


def _fold_counter(row: Array, v12: int, v14: int) -> Array:
    """XOR §3.2's two folded words into one half of the working vector's IV
    row — the byte offset into lane 0 (`v[12]`) and the final-block mask into
    lane 2 (`v[14]`) — as static lane slices re-concatenated, each value
    entering as a uint32 scalar literal, never as a host-materialised mask
    array that would lift into an unnamed operand
    (docs/reference/conventions.md). A zero value emits nothing and its lane
    passes through: the mask on every interior block, the offset on an empty
    message's one block."""
    lanes = [
        row[..., 0:1] ^ U32(v12) if v12 else row[..., 0:1],
        row[..., 1:2],
        row[..., 2:3] ^ U32(v14) if v14 else row[..., 2:3],
        row[..., 3:4],
    ]
    return fnp.concatenate(lanes, axis=-1)


def _g_row(
    va: Pair, vb: Pair, vc: Pair, vd: Pair, x: Pair, y: Pair
) -> tuple[Pair, Pair, Pair, Pair]:
    """The G mixing primitive (RFC 7693 §3.1) vectorized over four lanes at
    once — one call is a whole column or diagonal step (§3.2 runs four
    independent G's in each), every op element-wise on [B, 4] half grids.
    The rotation constants (32, 24, 16, 63) are BLAKE2b's (§2.1); the first
    is `rotr64`'s pure half swap."""
    va = add64(add64(va, vb), x)
    vd = rotr64(xor64(vd, va), 32)
    vc = add64(vc, vd)
    vb = rotr64(xor64(vb, vc), 24)
    va = add64(add64(va, vb), y)
    vd = rotr64(xor64(vd, va), 16)
    vc = add64(vc, vd)
    vb = rotr64(xor64(vb, vc), 63)
    return va, vb, vc, vd


def _roll_row(p: Pair, shift: int) -> Pair:
    """Rotate a row's four lanes by a static `shift` (`word.roll`: two static
    slices and a concatenate, never a gather), both halves alike."""
    return roll(p[0], shift, axis=-1), roll(p[1], shift, axis=-1)


def _sched(cols: list[Pair], idxs: tuple[int, ...]) -> tuple[Pair, Pair]:
    """The (x, y) message rows one vectorized G step reads: lane j's x is
    m[idxs[2j]] and its y m[idxs[2j + 1]] (§3.2's call pattern; a SIGMA row's
    first eight indices feed the column step, its last eight the diagonal) —
    the once-sliced word columns stacked into [B, 4] half grids at trace
    time. The SIGMA row is a Python tuple, so a schedule entry picks a column
    at trace time, never through an indexed read of a table array (the
    no-gather rule)."""
    xs, ys = idxs[0::2], idxs[1::2]

    def lanes(picks: tuple[int, ...]) -> Pair:
        return (
            fnp.stack([cols[i][0] for i in picks], axis=-1),
            fnp.stack([cols[i][1] for i in picks], axis=-1),
        )

    return lanes(xs), lanes(ys)


def _compress(
    state: Array, iv_lo: Array, iv_hi: Array, w32: Array, t: int, f: bool
) -> Array:
    """One compression F(h, m, t, f) (RFC 7693 §3.2): state [B, 16] (h as
    pairs) + the IV operand's [8] halves (split once by the caller, low at
    the even index) + message words w32 [B, 32] -> state [B, 16], everything
    in the module's little-endian pair layout. `t` (the byte offset at the
    end of this block) and `f` (the final-block flag) are HOST values, not
    operands: the length is static, so both fold into the working vector's
    IV lanes as scalar-literal XORs before anything reaches the device (the
    module docstring states the sub-2^64 exactness and why §3.2's `v[13]`
    XOR never appears).

    **The working vector rides as four 4-lane rows** — va = v[0..3],
    vb = v[4..7], vc = v[8..11], vd = v[12..15], each a pair of [B, 4] (or
    broadcastable [4]) half grids, the standard's own 4×4 matrix reading: a
    round is one vectorized G down the columns, a roll of rows 1..3 by
    1/2/3 that brings §3.2's diagonal quadruples into lane alignment, one
    vectorized G down those, and the rolls undone. The grid is load-bearing
    for the lowering, not a style choice: spelled over 32 loose [B] half
    lanes the whole block is one barrier-free element-wise DAG, and this
    toolchain's CPU pipeline fuses it into kLoop kernels thousands of
    instructions deep — measured on that spelling of this body, a lone
    (1, 3)-message digest stopped returning inside a 794 s run
    (`ascon._permutation` documents the same pathology). The rolls and
    schedule stacks are *load-bearing* data movement the canonicalizer
    keeps, so fusion regions stay round-sized.

    The rounds are a Python-unrolled `for` — the count is static and small,
    and unrolling is what turns each SIGMA entry into a static column slice
    (a `lax` loop would need a gather into the schedule, and is a
    control-flow boundary besides, docs/reference/conventions.md)."""
    b = state.shape[0]
    cols = [(w32[:, 2 * i], w32[:, 2 * i + 1]) for i in range(16)]  # m[i]
    ha: Pair = (state[:, 0:8:2], state[:, 1:8:2])  # h[0..3] as [B, 4] grids
    hb: Pair = (state[:, 8:16:2], state[:, 9:16:2])  # h[4..7]

    # v[0..7] := h, v[8..15] := IV (§3.2), with the offset/flag XORs folded
    # in per lane: v[12] ^= t mod 2^64, v[14] ^= 0xFF..FF on the final block.
    va, vb = ha, hb
    vc: Pair = (iv_lo[0:4], iv_hi[0:4])
    t_lo, t_hi = split(t)
    ff = 0xFFFFFFFF if f else 0
    vd: Pair = (
        _fold_counter(iv_lo[4:8], t_lo, ff),
        _fold_counter(iv_hi[4:8], t_hi, ff),
    )

    for r in range(12):
        s = _SIGMA[r % 10]  # rounds 10 and 11 reuse rows 0 and 1 (§2.7)
        x, y = _sched(cols, s[:8])
        va, vb, vc, vd = _g_row(va, vb, vc, vd, x, y)  # the column step
        vb, vc, vd = _roll_row(vb, -1), _roll_row(vc, -2), _roll_row(vd, -3)
        x, y = _sched(cols, s[8:])
        va, vb, vc, vd = _g_row(va, vb, vc, vd, x, y)  # the diagonal step
        vb, vc, vd = _roll_row(vb, 1), _roll_row(vc, 2), _roll_row(vd, 3)

    out_a = xor64(ha, va, vc)  # h_i' = h_i ^ v_i ^ v_{i+8} (§3.2)
    out_b = xor64(hb, vb, vd)
    lo = fnp.concatenate([out_a[0], out_b[0]], axis=-1)  # [B, 8]
    hi = fnp.concatenate([out_a[1], out_b[1]], axis=-1)
    # Interleave back to the pair layout: [lo0, hi0, lo1, hi1, ...].
    return fnp.stack([lo, hi], axis=-1).reshape(b, 16)


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and a batched consumer emits one digest call per tree level or
# transcript squeeze — so the uncached re-trace of the 96-G body would
# dominate the first-trace floor (cf. sha256_bytes and ripemd160_bytes).
# `inline=True` splices the cached jaxpr into the enclosing trace, so the
# emitted module (one composite marker per digest) is unchanged. The cache
# keys on the operand avals alone, and `h0`'s aval is uint32 [16] for EVERY
# digest size — the size lives in h0's VALUE (the parameter block) and in the
# caller-side truncation — so all digest sizes of one message shape share ONE
# trace, which `blake2b_test` pins.
@partial(frx.jit, inline=True)
def blake2b_bytes(h0: Array, msg: Array, tail: Array) -> Array:
    """Whole-message unkeyed BLAKE2b over raw bytes as ONE marked region:
    uint8 [B, L] -> uint8 [B, 64], padding, word packing, the compression
    chain and the serialization all inside the marker. The result is the
    UNTRUNCATED little-endian final state: the digest-size truncation is a
    pure output slice (§3.3 keeps the first nn bytes) and stays outside, so
    the marker is one wire shape for every digest size — which hash ran rides
    in `h0`, where `_initial_state` folded the parameter block.

    A name-routed digest marker, so it is exempt from the generic
    single-kernel rule (`sha256.sha256_merkle_damgard` states the exemption)
    and the body may chain blocks; the 12 rounds per block are
    Python-unrolled regardless, the count being static. No plugin ships a
    BLAKE2b recognizer yet (`_DEDICATED_EMITTER_AVAILABLE`), so today the
    marker inlines its decomposition on every backend — identical bytes, no
    dedicated kernel — and an emitter landing changes the lowering, never the
    value.

    Operands are explicit in the recognizer's positional ABI order:

    ``[0] h0``   uint32 [16]  — the parameter-XORed initial state (§3.3)
    ``[1] iv``   uint32 [16]  — the §2.6 IV, v[8..15] of every compression
    ``[2] msg``  uint8 [B, L] — the unpadded message batch
    ``[3] tail`` uint8 [P]    — the zero pad to the 128-byte boundary
                                (P = (−L) mod 128; a full block at L = 0,
                                empty at a block multiple)

    Passing `h0` and `iv` explicitly (rather than closing over module
    constants) is the operand-ABI rule in docs/reference/conventions.md: a
    host-materialised array captured by the body would be lifted into an
    unnamed operand *ahead* of these, one per call site, leaving no layout to
    write down. The counter schedule enters as scalar literals and the SIGMA
    rows as trace-time tuples, so neither ever becomes an operand. `iv` and
    `tail` are derivable (a constant; a function of the static L) — a
    recognizing emitter reads them rather than re-deriving them — and both
    are load-bearing for the inlined decomposition; the length the `t`
    counter needs is `msg`'s static trailing dimension."""

    def decomposition(
        h0: Array, iv: Array, msg: Array, tail: Array, **_attrs: object
    ) -> Array:
        b, ll = msg.shape
        padded = fnp.concatenate(
            [msg, fnp.broadcast_to(tail, (b, tail.shape[0]))], axis=-1
        )
        words = pack_le(padded.reshape(b, -1, _BLOCK))  # [B, nblocks, 32]
        state = fnp.broadcast_to(h0, (b, 16))
        iv_lo, iv_hi = iv[0:16:2], iv[1:16:2]  # [8] halves, low at even index
        nblocks = words.shape[1]
        for i in range(nblocks):  # static, small
            last = i == nblocks - 1
            # t = bytes hashed through the end of this block (§3.2): a full
            # block's worth per interior block, the true length on the final
            # one (§3.3) — the zero pad is never counted.
            t = ll if last else (i + 1) * _BLOCK
            state = _compress(state, iv_lo, iv_hi, words[:, i], t, last)
        return unpack_le(state)  # little-endian serialization: [B, 64]

    return fused_region(
        decomposition,
        h0,
        fnp.asarray(_IV_PAIRS),
        msg,
        tail,
        name=BLAKE2B_MARKER,
        version=BLAKE2B_MARKER_VERSION,
    )


def digest(msg: ArrayLike, digest_size: int = MAX_DIGEST_SIZE) -> fnp.ndarray:
    """Unkeyed BLAKE2b of a batch of equal-length messages. msg: uint8 [B, L]
    -> [B, digest_size] (default 64: BLAKE2b-512, the named full-length form).

    Byte-identical to RFC 7693 per message; the whole digest is emitted as
    the one name-routed `hash_frx.digest.blake2b` marker (`blake2b_bytes`),
    and the digest-size truncation is the caller-side slice that docstring
    states.

    **Traced or concrete.** `msg` may be a tracer, so a consumer can hash
    inside its own `@jit` or `vmap` without reaching past the `ByteHash`
    seam: the zero pad is built from the static length and never reads the
    message (`_padding_tail`), which is the same property `sha256.digest`
    states.
    """
    msg = fnp.asarray(msg, dtype=fnp.uint8)
    full = blake2b_bytes(
        fnp.asarray(_initial_state(digest_size)),
        msg,
        fnp.asarray(_padding_tail(msg.shape[-1])),
    )
    return full[:, :digest_size]


class Blake2b:
    """`ByteHash` for device unkeyed BLAKE2b — `digest` runs the batch through
    the `hash_frx.digest.blake2b` marker. No plugin recognizes that name yet,
    so `fusion_path` reads `GENERIC` on every backend today: the marker
    inlines, the bytes are the standard's, and an emitter landing flips the
    module flags and nothing here moves.

    For batched hashing where the messages already live on the device — an
    EIP-152/Zcash/Filecoin proof workload verifying many compressions at
    once. The strictly-sequential caller's fast path is the shipped
    `byte_hashes.HostBlake2b`, whose docstring is why the host row came
    first.

    The output length rides the value surface: `Blake2b(digest_size=32)` is a
    different hash from `Blake2b(64)` — RFC 7693 folds the length into the
    initial state — so `__eq__`/`__hash__` cover it, the same rule the host
    row states, and the param-free by-type equality of `Sha256`/`Ripemd160`
    does not apply here."""

    def __init__(self, digest_size: int = MAX_DIGEST_SIZE) -> None:
        # Range-checked here rather than left to the first `digest`, where
        # the caller can no longer choose another length (the host row's
        # rule; `_initial_state` re-checks on the module path).
        if not 1 <= digest_size <= MAX_DIGEST_SIZE:
            raise ValueError(
                f"digest_size must be in 1..{MAX_DIGEST_SIZE} (RFC 7693), got "
                f"{digest_size}"
            )
        self.digest_size = digest_size
        # Read per instance rather than pinned on the class: the emitter
        # switch is a property of the pin and the backend, and a value read
        # at import would pin the answer before anything could vary it.
        self.fusion_path = FusionPath.from_routing(_routes_to_dedicated_emitter())

    def digest(self, msg: ArrayLike) -> Array:
        return digest(msg, self.digest_size)  # the module-level marker digest

    def __eq__(self, other: object) -> bool:
        # By value over `digest_size` (`type(other) is not type(self)` rather
        # than `isinstance`, which is asymmetric under subclassing and blocks
        # Python's reflected-`__eq__` fallback) — so a device row never
        # equals the host row, and swapping substrate re-traces.
        if type(other) is not type(self):
            return NotImplemented
        return self.digest_size == other.digest_size

    def __hash__(self) -> int:
        return hash((type(self), self.digest_size))


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_blake2b: type[ByteHash] = Blake2b

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Unkeyed BLAKE2s (RFC 7693), host and device rows — BLAKE2b's 32-bit
sibling, byte-identical to the standard (and any conforming implementation,
e.g. Python's `hashlib.blake2s`). The construction is the same HAIFA shape the
`blake2b` package documents — Merkle–Damgård plus a byte counter and a
finalization flag threaded into every compression — at the 32-bit parameters:
uint32 words, 64-byte blocks, TEN rounds of the ARX G at rotations
(16, 12, 8, 7), SHA-256's IV, digests of 1..32 bytes.

**One module, beside `blake2b/`'s package — the placement decision #200 asked
for, recorded.** The sibling's package split (`byte_hashes.py` + `blake2b.py`)
is an artifact of its host-first history, not a shape this family needs, and
no `blake2/` merge earns its churn: the two share NO code — the 64-bit G rides
`word64` half pairs while this one is native uint32 (a 32-bit word is exactly
what the toolchain holds), and each module restates the small tables a
reviewer checks against RFC 7693 either way (the `blake2b._IV64` argument).
The message schedule is the RFC's one SIGMA table; BLAKE2s simply stops after
its first ten rows (§2.7 — no `i mod 10` reuse, the round count IS ten).

**Everything is little-endian** (RFC 7693 §2.4): blocks pack with the shared
`word.pack_le`, and a digest is exactly the serialized final state truncated
to `digest_size`, with no cross-word reorder.

**The counter schedule is host integers, not operands.** The length `L` is
static, so the byte offset `t` at every block (§3.2: the bytes hashed through
the end of that block — a full 64 per interior block, the true length on the
final one, zero pad never counted) and the final-block flag are known at
trace time; both fold into the IV words `v[12]`/`v[13]`/`v[14]` XOR on the
host and enter the body as uint32 scalar literals. BLAKE2s's `t` is 64-bit
over TWO uint32 words — `v[12] ^= t mod 2^32`, `v[13] ^= t >> 32` — and the
high word's XOR is the identity until a message passes 4 GiB, so it is
emitted only when nonzero (the zero-skip in `_fold_counter`).

Bulk-parallel by construction, like `sha256`: a batch of `B` equal-length
messages is hashed in one data-parallel call, every message advancing
independently through the shared schedule.

Contract: `digest(msg, digest_size)` takes uint8 `[B, L]` and returns uint8
`[B, digest_size]` little-endian digests (RFC 7693 §3.3). `L` is static, so
the zero pad is data-independent (`_padding_tail`), which is what lets `msg`
itself be traced. `digest_size` is 1..32 and rides the VALUE surface — RFC
7693 folds it into the initial state through the parameter block, so
`Blake2s(digest_size=16)` is a different hash from `Blake2s(32)`, not one
hash asked for fewer bytes (the SHAKE/BLAKE3 rule, stated on the sibling's
rows). Requires no x64.

Keyed hashing, salting and the tree parameters are RFC 7693 features this
module does not yet carry — value-surface additions riding the same marker,
added when a consumer (WireGuard's Noise IK uses keyed BLAKE2s) needs them
rather than speculatively.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import device_message, host_digest
from hash_frx.fusion import FusionPath, fused_region, routing
from hash_frx.word import pack_le, roll, rotr, unpack_le

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

U32 = fnp.uint32

MAX_DIGEST_SIZE = 32  # RFC 7693: BLAKE2s digests are 1..32 bytes

BLAKE2S_MARKER = "hash_frx.digest.blake2s"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# in `blake2s_bytes`. XLA recognizes a marker by name + attributes and
# deliberately does not gate on the version, which exists so a contract change
# can stage without a rename once a recognizer ships (`hash_frx.fusion`); an
# ABI change ships as a NEW name, the way `sha256_bytes` did.
BLAKE2S_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships a dedicated BLAKE2s emitter,
# and on which backends. None exists yet — the pre-emitter half of the keccak
# arrangement (`keccak.permutation._DEDICATED_EMITTER_AVAILABLE` carries the
# family-wide rationale), the posture the blake2b sibling and the other
# emitterless digests hold: both flags flip together with the `frx>=` floor in
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
    """Whether the pin *and* the backend both carry this emitter
    (`fusion.routing`, which carries the rationale)."""
    return routing(_DEDICATED_EMITTER_AVAILABLE, _EMITTER_BACKENDS)


_BLOCK = 64  # RFC 7693 §2.1: bb = 64-byte blocks (16 words of 32 bits)

# Initialization vector (RFC 7693 §2.6: the SHA-256 IV, as §2.6 itself notes —
# floor(2^32 · frac(sqrt of the i+1-th prime))). Restated rather than imported
# from `sha256`: the constants are the standard's own, and reaching for a
# sibling family's private table would couple the modules for eight literals a
# reviewer checks against §2.6 either way (the `blake2b._IV64` argument).
_IV32 = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)

# Message word schedule (RFC 7693 §2.7), one row per round of sixteen G
# inputs. The table is the RFC's one SIGMA — shared on paper with BLAKE2b —
# and BLAKE2s's round count is exactly its ten rows, so no reuse indexing
# appears. Static Python tuples deliberately: the unrolled round loop reads
# them at trace time, so a schedule entry becomes a static column pick of the
# message words rather than an indexed read of a table array (the no-gather
# rule, docs/reference/conventions.md).
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

# The §2.6 IV as uint32 [8] — the constant operand every compression reads for
# v[8..15]. Threaded as an explicit operand for the reason the sibling's
# `_IV_PAIRS` states: eight words have no index structure an in-body `iota`
# could count out, and a host-built array captured by the body would lift into
# an unnamed operand ahead of the declared ABI. Held as HOST numpy here and
# `fnp.asarray`ed at each use — a module-level device constant would
# initialize a backend at import, and a lazily-cached device array built first
# inside a trace would cache a TRACER.
_IV = np.array(_IV32, dtype=np.uint32)


@lru_cache(maxsize=None)
def _initial_state(digest_size: int) -> np.ndarray:
    """The initial state h for an unkeyed BLAKE2s-`digest_size` hash: the IV
    with the parameter-block word XORed into h[0] (RFC 7693 §2.5/§3.3:
    p[0] = 0x0101kknn — nn the hash size in bytes, kk = 0 unkeyed, bytes 2
    and 3 set as 01), as HOST uint32 [8] (`fnp.asarray`ed at use, for the
    reason `_IV` states). Cached per size: there are at most 32 of them.
    This is where the digest length enters the hash — which is why truncating
    a longer digest is the WRONG bytes at every shorter length, and why the
    range check lives here on the module path as well as on the row's
    constructor."""
    if not 1 <= digest_size <= MAX_DIGEST_SIZE:
        raise ValueError(
            f"digest_size must be in 1..{MAX_DIGEST_SIZE} (RFC 7693), got "
            f"{digest_size}"
        )
    out = _IV.copy()
    out[0] ^= 0x01010000 ^ digest_size
    return out


@lru_cache(maxsize=None)
def _padding_tail(length: int) -> np.ndarray:
    """What RFC 7693 §3.3 appends to a `length`-byte message: uint8 [P] of
    zeros, P = (−length) mod 64 — the final block zero-padded to the block
    boundary, plus the special case that an unkeyed empty message still
    processes one all-zero block (dd = 1). No 0x80 marker and no length
    field: the length enters through the `t` counter instead (HAIFA), which
    is what makes the tail all zeros — and empty at a block multiple, where
    the operand rides zero-length. A function of the static length alone, so
    `digest` can take a traced message (the `sha256._padding_tail`
    arrangement)."""
    if length == 0:
        return np.zeros(_BLOCK, dtype=np.uint8)
    return np.zeros((-length) % _BLOCK, dtype=np.uint8)


def _fold_counter(row: Array, t_lo: int, t_hi: int, ff: int) -> Array:
    """XOR §3.2's folded words into the working vector's second IV row — the
    byte offset's low word into lane 0 (`v[12]`), its high word into lane 1
    (`v[13]`), the final-block mask into lane 2 (`v[14]`) — as static lane
    slices re-concatenated, each value entering as a uint32 scalar literal,
    never as a host-materialised mask array that would lift into an unnamed
    operand (docs/reference/conventions.md). A zero value emits nothing and
    its lane passes through: `t_hi` below 4 GiB of input, the mask on every
    interior block, `t_lo` on an empty message's one block."""
    lanes = [
        row[..., 0:1] ^ U32(t_lo) if t_lo else row[..., 0:1],
        row[..., 1:2] ^ U32(t_hi) if t_hi else row[..., 1:2],
        row[..., 2:3] ^ U32(ff) if ff else row[..., 2:3],
        row[..., 3:4],
    ]
    return fnp.concatenate(lanes, axis=-1)


def _g_row(
    va: Array, vb: Array, vc: Array, vd: Array, x: Array, y: Array
) -> tuple[Array, Array, Array, Array]:
    """The G mixing primitive (RFC 7693 §3.1) vectorized over four lanes at
    once — one call is a whole column or diagonal step (§3.2 runs four
    independent G's in each), every op element-wise on [B, 4] uint32 grids.
    The rotation constants (16, 12, 8, 7) are BLAKE2s's (§2.1); the adds wrap
    mod 2^32 in the dtype, no carry machinery (the whole reason the 32-bit
    member's core is simpler than the sibling's half-pair G)."""
    va = va + vb + x
    vd = rotr(vd ^ va, 16)
    vc = vc + vd
    vb = rotr(vb ^ vc, 12)
    va = va + vb + y
    vd = rotr(vd ^ va, 8)
    vc = vc + vd
    vb = rotr(vb ^ vc, 7)
    return va, vb, vc, vd


def _sched(cols: list[Array], idxs: tuple[int, ...]) -> tuple[Array, Array]:
    """The (x, y) message rows one vectorized G step reads: lane j's x is
    m[idxs[2j]] and its y m[idxs[2j + 1]] (§3.2's call pattern; a SIGMA row's
    first eight indices feed the column step, its last eight the diagonal) —
    the once-sliced word columns stacked into [B, 4] grids at trace time. The
    SIGMA row is a Python tuple, so a schedule entry picks a column at trace
    time, never through an indexed read of a table array (the no-gather
    rule)."""
    xs, ys = idxs[0::2], idxs[1::2]
    return (
        fnp.stack([cols[i] for i in xs], axis=-1),
        fnp.stack([cols[i] for i in ys], axis=-1),
    )


def _compress(
    state: Array, iv_a: Array, iv_b: Array, w32: Array, t: int, f: bool
) -> Array:
    """One compression F(h, m, t, f) (RFC 7693 §3.2): state [B, 8] + the IV
    operand's [4] rows (split once by the caller: `iv[0:4]`, `iv[4:8]`) +
    message words w32 [B, 16] -> state [B, 8]. `t` (the byte offset at the
    end of this block) and `f` (the final-block flag) are HOST values, not
    operands: the length is static, so both fold into the working vector's IV
    lanes as scalar-literal XORs before anything reaches the device (the
    module docstring states the v[12]/v[13] split and the sub-4-GiB
    zero-skip).

    **The working vector rides as four 4-lane rows** — va = v[0..3],
    vb = v[4..7], vc = v[8..11], vd = v[12..15], each a [B, 4] (or
    broadcastable [4]) uint32 grid, the standard's own 4×4 matrix reading: a
    round is one vectorized G down the columns, a roll of rows 1..3 by 1/2/3
    that brings §3.2's diagonal quadruples into lane alignment, one
    vectorized G down those, and the rolls undone. The layout is inherited
    from the sibling's measured lesson (`blake2b._compress`: the same body
    over loose per-word lanes fed this toolchain's CPU pipeline kLoop
    kernels thousands of instructions deep and stopped returning), kept here
    where it is also simply the shortest spelling.

    The ten rounds are a Python-unrolled `for` — the count is static and
    small, and unrolling is what turns each SIGMA entry into a static column
    pick (a `lax` loop would need a gather into the schedule, and is a
    control-flow boundary besides, docs/reference/conventions.md)."""
    cols = [w32[:, i] for i in range(16)]  # m[i]
    ha, hb = state[:, 0:4], state[:, 4:8]

    # v[0..7] := h, v[8..15] := IV (§3.2), with the offset/flag XORs folded
    # in per lane: v[12] ^= t mod 2^32, v[13] ^= t >> 32, v[14] ^= 0xFF..FF
    # on the final block.
    va, vb = ha, hb
    vc = iv_a
    vd = _fold_counter(iv_b, t & 0xFFFFFFFF, t >> 32, 0xFFFFFFFF if f else 0)

    for r in range(10):
        s = _SIGMA[r]
        x, y = _sched(cols, s[:8])
        va, vb, vc, vd = _g_row(va, vb, vc, vd, x, y)  # the column step
        vb, vc, vd = roll(vb, -1, -1), roll(vc, -2, -1), roll(vd, -3, -1)
        x, y = _sched(cols, s[8:])
        va, vb, vc, vd = _g_row(va, vb, vc, vd, x, y)  # the diagonal step
        vb, vc, vd = roll(vb, 1, -1), roll(vc, 2, -1), roll(vd, 3, -1)

    # h_i' = h_i ^ v_i ^ v_{i+8}, the feedforward (§3.2).
    return fnp.concatenate([ha ^ va ^ vc, hb ^ vb ^ vd], axis=-1)


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and a batched consumer emits one digest call per tree level or
# transcript squeeze — so the uncached re-trace of the 80-G body would
# dominate the first-trace floor (cf. sha256_bytes and blake2b_bytes).
# `inline=True` splices the cached jaxpr into the enclosing trace, so the
# emitted module (one composite marker per digest) is unchanged. The cache
# keys on the operand avals alone, and `h0`'s aval is uint32 [8] for EVERY
# digest size — the size lives in h0's VALUE (the parameter block) and in the
# caller-side truncation — so all digest sizes of one message shape share ONE
# trace, which `blake2s_test` pins.
@partial(frx.jit, inline=True)
def blake2s_bytes(h0: Array, msg: Array, tail: Array) -> Array:
    """Whole-message unkeyed BLAKE2s over raw bytes as ONE marked region:
    uint8 [B, L] -> uint8 [B, 32], padding, word packing, the compression
    chain and the serialization all inside the marker. The result is the
    UNTRUNCATED little-endian final state: the digest-size truncation is a
    pure output slice (§3.3 keeps the first nn bytes) and stays outside, so
    the marker is one wire shape for every digest size — which hash ran rides
    in `h0`, where `_initial_state` folded the parameter block.

    A name-routed digest marker, so it is exempt from the generic
    single-kernel rule (`sha256.sha256_merkle_damgard` states the exemption)
    and the body may chain blocks; the ten rounds per block are
    Python-unrolled regardless, the count being static. No plugin ships a
    BLAKE2s recognizer yet (`_DEDICATED_EMITTER_AVAILABLE`), so today the
    marker inlines its decomposition on every backend — identical bytes, no
    dedicated kernel — and an emitter landing changes the lowering, never the
    value.

    Operands are explicit in the recognizer's positional ABI order:

    ``[0] h0``   uint32 [8]   — the parameter-XORed initial state (§3.3)
    ``[1] iv``   uint32 [8]   — the §2.6 IV, v[8..15] of every compression
    ``[2] msg``  uint8 [B, L] — the unpadded message batch
    ``[3] tail`` uint8 [P]    — the zero pad to the 64-byte boundary
                                (P = (−L) mod 64; a full block at L = 0,
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
        words = pack_le(
            padded.reshape(b, padded.shape[-1] // _BLOCK, _BLOCK)
        )  # [B, nblocks, 16]
        state = fnp.broadcast_to(h0, (b, 8))
        iv_a, iv_b = iv[0:4], iv[4:8]
        nblocks = words.shape[1]
        for i in range(nblocks):  # static, small
            last = i == nblocks - 1
            # t = bytes hashed through the end of this block (§3.2): a full
            # block's worth per interior block, the true length on the final
            # one (§3.3) — the zero pad is never counted.
            t = ll if last else (i + 1) * _BLOCK
            state = _compress(state, iv_a, iv_b, words[:, i], t, last)
        return unpack_le(state)  # little-endian serialization: [B, 32]

    return fused_region(
        decomposition,
        h0,
        fnp.asarray(_IV),
        msg,
        tail,
        name=BLAKE2S_MARKER,
        version=BLAKE2S_MARKER_VERSION,
    )


def digest(msg: ArrayLike, digest_size: int = MAX_DIGEST_SIZE) -> fnp.ndarray:
    """Unkeyed BLAKE2s of a batch of equal-length messages. msg: uint8 [B, L]
    -> [B, digest_size] (default 32: BLAKE2s-256, the named full-length form).

    Byte-identical to RFC 7693 per message; the whole digest is emitted as
    the one name-routed `hash_frx.digest.blake2s` marker (`blake2s_bytes`),
    and the digest-size truncation is the caller-side slice that docstring
    states.

    **Traced or concrete.** `msg` may be a tracer, so a consumer can hash
    inside its own `@jit` or `vmap` without reaching past the `ByteHash`
    seam: the zero pad is built from the static length and never reads the
    message (`_padding_tail`), which is the same property `sha256.digest`
    states.
    """
    msg = device_message(msg)
    full = blake2s_bytes(
        fnp.asarray(_initial_state(digest_size)),
        msg,
        fnp.asarray(_padding_tail(msg.shape[-1])),
    )
    return full[:, :digest_size]


class Blake2s:
    """`ByteHash` for device unkeyed BLAKE2s — `digest` runs the batch through
    the `hash_frx.digest.blake2s` marker. No plugin recognizes that name yet,
    so `fusion_path` reads `GENERIC` on every backend today: the marker
    inlines, the bytes are the standard's, and an emitter landing flips the
    module flags and nothing here moves.

    For batched hashing where the messages already live on the device. The
    strictly-sequential caller's fast path is `HostBlake2s` below — the
    WireGuard-style transcript caller reading each digest back immediately.

    The output length rides the value surface: `Blake2s(digest_size=16)` is a
    different hash from `Blake2s(32)` — RFC 7693 folds the length into the
    initial state — so `__eq__`/`__hash__` cover it, the same rule the
    sibling's rows state, and the param-free by-type equality of
    `Sha256`/`Ripemd160` does not apply here."""

    def __init__(self, digest_size: int = MAX_DIGEST_SIZE) -> None:
        # Range-checked here rather than left to the first `digest`, where
        # the caller can no longer choose another length (`_initial_state`
        # re-checks on the module path).
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


class HostBlake2s:
    """`ByteHash` for host unkeyed BLAKE2s over `hashlib.blake2s` — a
    guaranteed constructor, so the row ships unconditionally: the fast path
    for a strictly-sequential byte caller, free the way the sibling's host
    row was. The output length rides the value surface, same as `Blake2s`."""

    fusion_path = FusionPath.HOST

    def __init__(self, digest_size: int = MAX_DIGEST_SIZE) -> None:
        if not 1 <= digest_size <= MAX_DIGEST_SIZE:
            raise ValueError(
                f"digest_size must be in 1..{MAX_DIGEST_SIZE} (RFC 7693), got "
                f"{digest_size}"
            )
        self.digest_size = digest_size

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(
            lambda row: hashlib.blake2s(row, digest_size=self.digest_size).digest(),
            self.digest_size,
            msg,
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.digest_size == other.digest_size

    def __hash__(self) -> int:
        return hash((type(self), self.digest_size))


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md).
    _bh_blake2s: type[ByteHash] = Blake2s
    _bh_host_blake2s: type[ByteHash] = HostBlake2s

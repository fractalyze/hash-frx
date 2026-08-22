# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ascon-Hash256 over uint32 lane halves, authored in frx — byte-identical to
NIST SP 800-232 (final, August 2025, "Ascon-Based Lightweight Cryptography
Standards for Constrained Devices", https://doi.org/10.6028/NIST.SP.800-232;
section references below are to that document).

The lightweight-cryptography standard's hash: a 320-bit state of five 64-bit
words S0..S4, an 8-byte rate, and one permutation — Ascon-p[12], twelve rounds
of constant addition (§3.2), a 5-bit S-box applied bitsliced across the five
words (§3.3), and per-word linear diffusion of two rotated copies (§3.4) —
used for initialization, absorbing and squeezing alike (§5.1). Its value in
this package is the second byte sponge (#169's unblock condition) and the
duplex-seam exercise #157 queued it for.

**A state word is a (lo, hi) pair of uint32 halves, never a uint64** — the
`keccak/lane.py` law: with x64 off `uint64` truncates, and enabling it flips
process-wide defaults. The five words ride as one `[B, 5]` grid per half
(the keccak `(5, 5)`-grid arrangement, one row of it), and the round steps
are grid-wise: the S-box needs no bit extraction — the words ARE the bit
planes, and its word-crossing terms are static rolls of the grid — while the
per-word Σ rotations split the grid the way Grøstl's ShiftBytes splits rows.
[`permutation.py`](permutation.py) states why the grid, and not ten loose
half arrays, is the shape that survives lowering, and carries Ascon-p itself as
a `Permutation` — the seam a generic sponge takes.

**Byte order is the final SP's, not the classic Ascon papers'.** SP 800-232
switched the word convention to little-endian (§1.2's change log; §A.1: the
first eight bytes load into S0 "in little-endian notation"), so blocks pack
with the shared `word.pack_le` and the padding marker is the byte 0x01
(§A.2), not the CAESAR-era 0x80. The KAT anchors in `ascon/testing` are what
hold this to the standard.

Bulk-parallel by construction, like `sha256`: a batch of `B` equal-length
messages is hashed in one data-parallel call, every message advancing
independently through the shared schedule.

Contract: `digest(msg)` takes uint8 `[B, L]` and returns uint8 `[B, 32]`
digests (H0 ‖ H1 ‖ H2 ‖ H3, §5.1 eq. 63). Length `L` is static, so the
padding is data-independent: a host tail built from the length alone
(`_padding_tail`), concatenated on — which is what lets `msg` itself be
traced. Requires no x64; everything is uint32 halves and uint8 bytes.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.ascon.permutation import WORDS, masks, permutation
from hash_frx.byte_hash import DeviceRow, device_message, padded_batch
from hash_frx.fusion import FusionPath, fused_region, routing
from hash_frx.word import pack_le, split, unpack_le

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

ASCON_HASH256_MARKER = "hash_frx.digest.ascon_hash256"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# in `ascon_hash256_bytes`. XLA recognizes a marker by name + attributes and
# deliberately does not gate on the version, which exists so a contract change
# can stage without a rename once a recognizer ships (`hash_frx.fusion`); an
# ABI change ships as a NEW name, the way `sha256_bytes` did.
ASCON_HASH256_MARKER_VERSION = 1

ASCON_HASH256_DIGEST_SIZE = 32

# Whether the pinned Fractalyze XLA plugin ships a dedicated Ascon emitter,
# and on which backends. None exists yet — this is the pre-emitter half of the
# keccak arrangement (`keccak.permutation._DEDICATED_EMITTER_AVAILABLE`
# carries the family-wide rationale), the same posture `vision` and `grostl`
# hold: both flags flip together with the `frx>=` floor in `pyproject.toml`
# when an emitter lands, and `fusion_path_test`'s matrix law holds them to
# agree. The marker is emitted regardless — there is no per-block routing
# alternative for a whole-hash digest — and unrecognized it inlines its
# decomposition: right bytes, `GENERIC` fusion path.
_DEDICATED_EMITTER_AVAILABLE = False

# Which backends have that emitter — a different question from the pin, asked
# alongside it. Empty until one is written; a backend gaining an arm joins
# this tuple and nothing else here moves (the keccak/poseidon2 pattern).
_EMITTER_BACKENDS: tuple[str, ...] = ()


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry this emitter
    (`fusion.routing`, which carries the rationale)."""
    return routing(_DEDICATED_EMITTER_AVAILABLE, _EMITTER_BACKENDS)


_RATE = 8  # bytes: §5.1, rate 64 bits / capacity 256
# The initialization S = Ascon-p[12](IV ‖ 0^256) for IV = 0x0000080100cc0002
# (§5.1 eq. 54; Table 14), precomputed — the SP publishes the result (§A.3,
# Table 12), transcribed here and split into halves on the host, where Python
# ints are exact (`word.split`). `ascon_test` holds this transcription to the
# oracle's own derivation from the IV.
_INITIAL_STATE_WORDS = (
    0x9B1E5494E934D681,
    0x4BC3A01E333751D2,
    0xAE65396C6B34B81A,
    0x3C7FD4A4D56A4DB3,
    0x1A5C464906C5976D,
)
# uint32 [5, 2]: word i as (lo, hi) at [i, 0] / [i, 1].
_INITIAL_STATE = np.array([split(w) for w in _INITIAL_STATE_WORDS], dtype=np.uint32)

_INITIAL_STATEd = fnp.asarray(_INITIAL_STATE)


@lru_cache(maxsize=None)
def _padding_tail(length: int) -> np.ndarray:
    """What pad(·, 64) (Algorithm 2) appends to a `length`-byte message:
    uint8 [P].

    The bit 1 then zeros to the next rate boundary — in the little-endian
    byte convention the byte 0x01 then zero bytes (§A.2: y = x ⊕ (1 ≪ 8n)).
    Never empty, so a rate-aligned message gains a whole padding block. Every
    term is a function of the length alone, so the tail is a host constant
    built *from the length* rather than written *into the message* — which is
    what lets `digest` take a traced message (the `sha256._padding_tail`
    arrangement). Shared by the whole batch, since one call hashes messages
    of one length.
    """
    tail = np.zeros(_RATE - length % _RATE, dtype=np.uint8)
    tail[0] = 0x01
    return tail


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and a Merkle commit emits one digest call per tree level — so the
# uncached re-trace of the multi-permutation body would dominate the
# first-trace floor (cf. sha256_bytes and grostl256_bytes). `inline=True`
# splices the cached jaxpr into the enclosing trace, so the emitted module
# (one composite marker per digest) is unchanged.
@partial(frx.jit, inline=True)
def ascon_hash256_bytes(msg: Array) -> Array:
    """Whole-message Ascon-Hash256 over raw bytes as ONE marked region:
    uint8 [B, L] -> uint8 [B, 32], padding, the absorb chain and the squeeze
    all inside the marker.

    A name-routed digest marker, so it is exempt from the generic
    single-kernel rule (`sha256.sha256_merkle_damgard` states the exemption)
    and the body may chain blocks; the twelve rounds of each permutation are
    Python-unrolled regardless, the count being static and small. No plugin
    ships an Ascon recognizer yet (`_DEDICATED_EMITTER_AVAILABLE`), so today
    the marker inlines its decomposition on every backend — identical bytes,
    no dedicated kernel — and an emitter landing changes the lowering, never
    the value.

    Operands are explicit in the recognizer's positional ABI order:

    ``[0] init``  uint32 [5, 2] — Ascon-p[12](IV ‖ 0^256) precomputed (§A.3,
                                  Table 12), word i as (lo, hi) at [i, 0]/[i, 1]
    ``[1] msg``   uint8 [B, L]  — the unpadded message batch
    ``[2] tail``  uint8 [P]     — the Algorithm 2 padding for the static L

    Passing the state table explicitly (rather than closing over the module
    constant) is the operand-ABI rule in docs/reference/conventions.md: a
    host-materialised array captured by the body would be lifted into an
    unnamed operand *ahead* of these, one per call site, leaving no layout to
    write down. The twelve 8-bit round constants are NOT operands: they ride
    as scalar literals in the body (`permutation.ROUND_CONSTANTS` states why that
    is lifting-safe), and the S-box masks are derived from `iota` on device
    (`permutation.masks`); the three-operand count is pinned in `ascon_test`. `tail` is
    derivable from the static L — a recognizing emitter reads it rather than
    re-deriving it — and load-bearing for the inlined decomposition.
    """

    def decomposition(init: Array, msg: Array, tail: Array, **_attrs: object) -> Array:
        b = msg.shape[0]
        padded = padded_batch(msg, tail)
        # Blocks as (lo, hi) uint32 pairs, packed little-endian (§A.1):
        # [B, nblocks, 2], [..., 0] the low half.
        words = pack_le(padded.reshape(b, padded.shape[-1] // _RATE, _RATE))
        lo = fnp.broadcast_to(init[:, 0], (b, WORDS))
        hi = fnp.broadcast_to(init[:, 1], (b, WORDS))
        m = masks()
        # Absorbing (§5.1): every 64-bit block XORs into S0 — column 0,
        # patched by slice + concatenate so the other four words see no op
        # (the `keccak._patch_lane_zero` spelling and its reasoning) — each
        # block followed by the permutation. Algorithm 5 defers the last
        # block's permutation to the squeeze; same schedule.
        for i in range(padded.shape[-1] // _RATE):  # static, small
            lo = fnp.concatenate([lo[:, :1] ^ words[:, i, :1], lo[:, 1:]], axis=-1)
            hi = fnp.concatenate([hi[:, :1] ^ words[:, i, 1:], hi[:, 1:]], axis=-1)
            lo, hi = permutation(lo, hi, m)
        # Squeezing (§5.1): H_0..H_3 read S0 with the permutation between
        # reads; H = H_0 ‖ H_1 ‖ H_2 ‖ H_3 (eq. 63), each word written back
        # little-endian.
        h = [lo[:, 0], hi[:, 0]]
        for _ in range(3):
            lo, hi = permutation(lo, hi, m)
            h += [lo[:, 0], hi[:, 0]]
        return unpack_le(fnp.stack(h, axis=-1))

    return fused_region(
        decomposition,
        _INITIAL_STATEd,
        msg,
        fnp.asarray(_padding_tail(msg.shape[-1])),
        name=ASCON_HASH256_MARKER,
        version=ASCON_HASH256_MARKER_VERSION,
    )


def digest(msg: ArrayLike) -> fnp.ndarray:
    """Ascon-Hash256 of a batch of equal-length messages. msg: uint8 [B, L]
    -> [B, 32].

    Byte-identical to NIST SP 800-232 per message; the whole digest is
    emitted as the one name-routed `hash_frx.digest.ascon_hash256` marker
    (`ascon_hash256_bytes`).

    **Traced or concrete.** `msg` may be a tracer, so a consumer can hash
    inside its own `@jit` or `vmap` without reaching past the `ByteHash`
    seam: the padding is built from the static length and never reads the
    message (`_padding_tail`), which is the same property `sha256.digest`
    states.
    """
    msg = device_message(msg)
    return ascon_hash256_bytes(msg)


class AsconHash256(DeviceRow):
    """`ByteHash` for device Ascon-Hash256 — `digest` runs the batch through
    the `hash_frx.digest.ascon_hash256` marker. No plugin recognizes that
    name yet, so `fusion_path` reads `GENERIC` on every backend today: the
    marker inlines, the bytes are the standard's, and an emitter landing
    flips the module flags and nothing here moves.

    For batched hashing where the messages already live on the device. The
    strictly-sequential caller's alternative is the testonly
    `ascon.testing.host_ascon_hash256.HostAsconHash256` — pure-Python, so
    unlike the SHA-2/SHA-3 host rows it does not ship (`hashlib` has no
    Ascon)."""

    digest_size = ASCON_HASH256_DIGEST_SIZE

    def __init__(self) -> None:
        super().__init__(FusionPath.from_routing(_routes_to_dedicated_emitter()))

    def digest(self, msg: ArrayLike) -> Array:
        return digest(msg)  # the module-level marker digest above


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_ascon_hash256: type[ByteHash] = AsconHash256

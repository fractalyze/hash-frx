# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The NIST SP 800-232 hashes over uint32 lane halves, authored in frx —
byte-identical to that standard (final, August 2025, "Ascon-Based Lightweight
Cryptography Standards for Constrained Devices",
https://doi.org/10.6028/NIST.SP.800-232; section references below are to that
document).

**All three of the standard's hashes live here, and the fourth function does
not.** SP 800-232 defines Ascon-Hash256 (§5.1), Ascon-XOF128 (§5.2),
Ascon-CXOF128 (§5.3) and Ascon-AEAD128 (§4). The three hashes are these rows;
the AEAD is a mode over ciphertext, which this repo's non-negotiable keeps out
(see the root `CLAUDE.md`), and enc-frx owns it — building it by importing
`AsconP` from `permutation.py` rather than by duplicating the permutation.

The three are one schedule at three initial states. Same 8-byte rate, same
Algorithm 2 padding, same absorb and squeeze; what a row brings is its IV, its
output length, and — for Ascon-CXOF128 — a length-encoded customization string
absorbed ahead of the message. They are separate rows rather than one row with
settings because the IVs differ, so they are different hashes at every input:
`AsconCxof128(b"", n)` is NOT `AsconXof128(n)`.

The lightweight-cryptography standard's hash: a 320-bit state of five 64-bit
words S0..S4, an 8-byte rate, and one permutation — Ascon-p[12], twelve rounds
of constant addition (§3.2), a 5-bit S-box applied bitsliced across the five
words (§3.3), and per-word linear diffusion of two rotated copies (§3.4) —
used for initialization, absorbing and squeezing alike (§5.1). Its value in
this package is the second byte sponge — the one the shared schedule in
[`extension/sponge.py`](../extension/sponge.py) was shaped against — and the
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
digests (H0 ‖ H1 ‖ H2 ‖ H3, §5.1 eq. 63); the two XOFs return `[B, n]` for the
length their row was built with. Length `L` is static, so the padding is
data-independent: a host tail built from the length alone (`_PAD`),
concatenated on — which is what lets `msg` itself be traced. The customization
is host `bytes` for the same reason, so it never makes a message untraceable
either. Requires no x64; everything is uint32 halves and uint8 bytes.
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

from hash_frx.ascon.permutation import WORDS, Lane, masks, permutation
from hash_frx.byte_hash import DeviceRow, device_message, padded_batch
from hash_frx.extension.pad import SpongePad
from hash_frx.extension.sponge import absorb_squeeze, squeeze_blocks
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

ASCON_XOF128_MARKER = "hash_frx.digest.ascon_xof128"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# in `ascon_xof128_bytes`, which is Ascon-Hash256's with a different initial
# state and an `output_size` attribute.
ASCON_XOF128_MARKER_VERSION = 1

ASCON_CXOF128_MARKER = "hash_frx.digest.ascon_cxof128"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# in `ascon_cxof128_bytes`, which is Ascon-XOF128's with a FOURTH operand — the
# length-encoded customization blocks — ahead of the message.
ASCON_CXOF128_MARKER_VERSION = 1

# §5.3: |Z| <= 2048 bits. The cap is what makes the length field a fixed eight
# bytes rather than a variable-width encoding, so it is the standard's bound
# and not a defensive limit.
MAX_CUSTOMIZATION_BYTES = 256

# Whether the pinned Fractalyze XLA plugin ships a dedicated Ascon emitter,
# and on which backends. None exists yet — this is the pre-emitter half of the
# keccak arrangement (`keccak.permutation._DEDICATED_EMITTER_AVAILABLE`
# carries the family-wide rationale), the posture the other emitterless
# digests hold: both flags flip together with the `frx>=` floor in
# `pyproject.toml` when an emitter lands, and `fusion_path_test`'s matrix
# law holds them to agree. The marker is emitted regardless — there is no
# per-block routing alternative for a whole-hash digest — and unrecognized
# it inlines its decomposition: right bytes, `GENERIC` fusion path.
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
# The Ascon-XOF128 initialization, S = Ascon-p[12](IV ‖ 0^256) for
# IV = 0x0000080000cc0003 (§B, Table 13; the same IV layout as Ascon-Hash256's
# with the version byte 3 and an output length of 0, which is what "arbitrary
# output" is spelled as). Precomputed on the same terms as the hash's above;
# `ascon_test` holds this transcription to the oracle's own derivation from the
# IV, which is what makes it a second reading of the standard rather than a
# second copy of one.
_XOF128_INITIAL_STATE_WORDS = (
    0xDA82CE768D9447EB,
    0xCC7CE6C75F1EF969,
    0xE7508FD780085631,
    0x0EE0EA53416B58CC,
    0xE0547524DB6F0BDE,
)

# The Ascon-CXOF128 initialization, S = Ascon-p[12](IV ‖ 0^256) for
# IV = 0x0000080000CC0004 (§B, Table 13; the XOF's IV layout at version 4 —
# the customization is what the version distinguishes, not the output length,
# which is 0 for both). Precomputed on the same terms as the two above, and
# held to `ascon_test`'s independent derivation for the same reason.
_CXOF128_INITIAL_STATE_WORDS = (
    0x675527C2A0E8DE03,
    0x43D12D7DC0377BBC,
    0xE9901DEC426E81B5,
    0x2AB14907720780B6,
    0x8F3F1D02D432BC46,
)

# uint32 [5, 2]: word i as (lo, hi) at [i, 0] / [i, 1].
_INITIAL_STATE = np.array([split(w) for w in _INITIAL_STATE_WORDS], dtype=np.uint32)
_XOF128_INITIAL_STATE = np.array(
    [split(w) for w in _XOF128_INITIAL_STATE_WORDS], dtype=np.uint32
)
_CXOF128_INITIAL_STATE = np.array(
    [split(w) for w in _CXOF128_INITIAL_STATE_WORDS], dtype=np.uint32
)

_INITIAL_STATEd = fnp.asarray(_INITIAL_STATE)
_XOF128_INITIAL_STATEd = fnp.asarray(_XOF128_INITIAL_STATE)
_CXOF128_INITIAL_STATEd = fnp.asarray(_CXOF128_INITIAL_STATE)


# The Algorithm 2 pad: the bit 1 then zeros to the next rate boundary, which in
# the little-endian byte convention is the byte 0x01 then zero bytes (§A.2:
# y = x ⊕ (1 ≪ 8n)). No trailing bit, unlike FIPS 202's `pad10*1` — the axis
# `SpongePad` carries, and the reason Ascon's tail and Keccak's are one rule
# rather than two transcriptions of one shape.
_PAD = SpongePad(rate=_RATE, head=0x01, final_bit=False)


def _customization_prefix(customization: bytes) -> np.ndarray:
    """The blocks Ascon-CXOF128 absorbs AHEAD of the message: `Z0 ‖ Z ‖ pad`,
    as uint8 [Q] (§5.3).

    `Z0` is the customization's length **in BITS**, eight little-endian bytes —
    a whole rate block on its own — then the string, then the message pad rule
    applied to the string's length. Host arithmetic over a construction
    parameter, so it never reads a message and never reaches a device as
    anything but a materialized constant.

    **Not cSHAKE's `bytepad`/`encode_string`.** The SHA-3 family solves the same
    problem with a left-encoded BYTE count inside a rate-padded envelope, and
    reading one encoding off the other is the error a self-consistent
    implementation hides — right shape, right block count, wrong digest. The
    published vectors in `ascon/testing/reference.py` are what tell them apart.

    Rate-aligned by construction: `Z0` is exactly one block and `_PAD` closes
    `Z` to a boundary, which is what lets the caller simply prepend these bytes
    to the padded message rather than run a second schedule.
    """
    if len(customization) > MAX_CUSTOMIZATION_BYTES:
        raise ValueError(
            f"customization is {len(customization)} bytes; §5.3 allows at most "
            f"{MAX_CUSTOMIZATION_BYTES}"
        )
    return np.concatenate(
        [
            np.frombuffer((len(customization) * 8).to_bytes(8, "little"), np.uint8),
            np.frombuffer(customization, np.uint8),
            _PAD.tail(len(customization)),
        ]
    )


def _sponge(init: Array, padded: Array, output_size: int) -> Array:
    """§5.1's absorb-and-squeeze over an already rate-aligned byte stream:
    uint8 [B, N * 8] -> uint8 [B, output_size].

    All three rows reach this, because all three ARE this: one rate, one
    permutation, one schedule. What a row brings is its initial state, its
    squeeze count, and — for Ascon-CXOF128 — some bytes on the front of the
    stream.
    """
    b = padded.shape[0]
    # Blocks as (lo, hi) uint32 pairs, packed little-endian (§A.1):
    # [B, nblocks, 2], [..., 0] the low half.
    words = pack_le(padded.reshape(b, padded.shape[-1] // _RATE, _RATE))
    lo = fnp.broadcast_to(init[:, 0], (b, WORDS))
    hi = fnp.broadcast_to(init[:, 1], (b, WORDS))
    m = masks()

    # Absorbing (§5.1): every 64-bit block XORs into S0 — column 0 of each
    # half grid, patched by slice + concatenate so the other four words see
    # no op (the `keccak._patch_lane_zero` spelling and its reasoning).
    # Algorithm 5 defers the last block's permutation to the squeeze; same
    # schedule.
    #
    # Spelled here rather than through `extension.sponge.merge_into_rate`,
    # which is the same slice-and-concatenate and which the two Keccak sites
    # do share. The two families emit it in opposite order: Keccak packs its
    # block before merging, so the block's ops precede the state slice,
    # while this one slices S0 first. A helper takes its block as an
    # argument, so it can only have the one order — and moving either family
    # to the other's would be a lowering change with no value behind it.
    def absorb(state: Lane, i: int) -> Lane:
        lo, hi = state
        return (
            fnp.concatenate([lo[:, :1] ^ words[:, i, :1], lo[:, 1:]], axis=-1),
            fnp.concatenate([hi[:, :1] ^ words[:, i, 1:], hi[:, 1:]], axis=-1),
        )

    # Squeezing (§5.1): H_0..H_3 read S0 with the permutation between
    # reads; H = H_0 ‖ H_1 ‖ H_2 ‖ H_3 (eq. 63), each word written back
    # little-endian. The schedule owns "no permutation after the last
    # read" — the rule this used to spell by peeling the first read out of
    # the loop, and Keccak by guarding the permute.
    squeezed = absorb_squeeze(
        (lo, hi),
        blocks=padded.shape[-1] // _RATE,
        squeezes=squeeze_blocks(output_size, _RATE),
        absorb=absorb,
        permute=lambda state: permutation(state[0], state[1], m),
        read=lambda state: (state[0][:, 0], state[1][:, 0]),
    )
    digest = unpack_le(fnp.stack([half for word in squeezed for half in word], axis=-1))
    # The squeeze emits whole rate blocks, so a request that is not a
    # multiple of 8 overshoots. Sliced only when it does: Ascon-Hash256's
    # 32 bytes are four exact blocks, and an unconditional slice would put
    # an op in its region that was never there.
    if digest.shape[-1] != output_size:
        digest = digest[:, :output_size]
    return digest


def _digest_decomposition(output_size: int) -> Callable[..., Array]:
    """The three-operand body: `[init, msg, tail]`.

    `output_size` is the only thing that differs between the two rows that run
    it — it fixes the squeeze count — and the initial state arrives as an
    operand, so Ascon-Hash256 and Ascon-XOF128 are one body at two IVs rather
    than two transcriptions of §5.1.
    """

    def decomposition(init: Array, msg: Array, tail: Array, **_attrs: object) -> Array:
        return _sponge(init, padded_batch(msg, tail), output_size)

    return decomposition


def _cxof_decomposition(output_size: int) -> Callable[..., Array]:
    """The four-operand body: `[init, prefix, msg, tail]`.

    **The customization is not a second schedule.** §5.3 absorbs its
    length-encoded string as rate blocks with a permutation after each — the
    same thing the message absorb does — and `_customization_prefix` is
    rate-aligned, so putting those bytes on the front of the padded message and
    running the ordinary absorb IS Ascon-CXOF128. The published vectors in
    `ascon/testing/reference.py` hold that equivalence to the standard rather
    than to this paragraph.

    So what a fourth operand buys is not a stage but a buffer, and the region
    stays the one absorb loop the other two rows emit.
    """

    def decomposition(
        init: Array, prefix: Array, msg: Array, tail: Array, **_attrs: object
    ) -> Array:
        rows = msg.shape[0]
        # One three-way concatenate rather than `padded_batch` wrapped in a
        # second: the prefix goes in FRONT, so reusing that helper would emit
        # two concatenates into the region where one does.
        stream = fnp.concatenate(
            [
                fnp.broadcast_to(prefix, (rows, prefix.shape[0])),
                msg,
                fnp.broadcast_to(tail, (rows, tail.shape[0])),
            ],
            axis=-1,
        )
        return _sponge(init, stream, output_size)

    return decomposition


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

    return fused_region(
        _digest_decomposition(ASCON_HASH256_DIGEST_SIZE),
        _INITIAL_STATEd,
        msg,
        fnp.asarray(_PAD.tail(msg.shape[-1])),
        name=ASCON_HASH256_MARKER,
        version=ASCON_HASH256_MARKER_VERSION,
    )


@partial(frx.jit, static_argnames=("output_size",), inline=True)
def ascon_xof128_bytes(msg: Array, output_size: int) -> Array:
    """Whole-message Ascon-XOF128 over raw bytes as ONE marked region:
    uint8 [B, L] -> uint8 [B, output_size].

    Ascon-Hash256's region at a different initial state. §5.2's XOF runs the
    same rate, the same Algorithm 2 padding and the same absorb; what it drops
    is the fixed output length, which the IV encodes as 0 and which reaches
    this body as the squeeze count instead. That is why the two share
    `_digest_decomposition` rather than the XOF being a second transcription.

    Operands are the hash's, in the same positional order:

    ``[0] init``  uint32 [5, 2] — Ascon-p[12](IV ‖ 0^256) for the XOF IV,
                                  word i as (lo, hi) at [i, 0]/[i, 1]
    ``[1] msg``   uint8 [B, L]  — the unpadded message batch
    ``[2] tail``  uint8 [P]     — the Algorithm 2 padding for the static L

    `output_size` rides as an attribute rather than an operand: it fixes the
    region's SHAPE — how many squeeze permutations run and how wide the result
    is — and shape is what an attribute is for, where an operand is what
    determines a value. It is a static argument here for the same reason,
    since the squeeze is Python-unrolled.
    """
    return fused_region(
        _digest_decomposition(output_size),
        _XOF128_INITIAL_STATEd,
        msg,
        fnp.asarray(_PAD.tail(msg.shape[-1])),
        name=ASCON_XOF128_MARKER,
        version=ASCON_XOF128_MARKER_VERSION,
        output_size=output_size,
    )


def xof128(msg: ArrayLike, output_size: int) -> Array:
    """Ascon-XOF128 of a batch of equal-length messages, read out to
    `output_size` bytes. msg: uint8 [B, L] -> [B, output_size].

    Byte-identical to NIST SP 800-232 §5.2 per message. Traced or concrete, on
    the same terms as `digest`.
    """
    if output_size < 1:
        raise ValueError(f"output_size ({output_size}) must be >= 1")
    return ascon_xof128_bytes(device_message(msg), output_size)


@partial(frx.jit, static_argnames=("customization", "output_size"), inline=True)
def ascon_cxof128_bytes(msg: Array, customization: bytes, output_size: int) -> Array:
    """Whole-message Ascon-CXOF128 over raw bytes as ONE marked region:
    uint8 [B, L] -> uint8 [B, output_size].

    Ascon-XOF128's region at a third initial state, with §5.3's customization
    blocks absorbed ahead of the message. Same rate, same Algorithm 2 padding,
    same absorb and squeeze.

    Operands are the XOF's with one inserted, in the recognizer's positional
    ABI order:

    ``[0] init``    uint32 [5, 2] — Ascon-p[12](IV ‖ 0^256) for the CXOF IV,
                                    word i as (lo, hi) at [i, 0]/[i, 1]
    ``[1] prefix``  uint8 [Q]     — `Z0 ‖ Z ‖ pad`, the length-encoded
                                    customization blocks (`_customization_prefix`)
    ``[2] msg``     uint8 [B, L]  — the unpadded message batch
    ``[3] tail``    uint8 [P]     — the Algorithm 2 padding for the static L

    **The customization is a static argument, so a new one is a new trace.**
    It is part of *which hash this is* rather than per-call data — the
    `Blake3Keyed` rule — and it rides in the compiled program's constant pool
    the way `tail` does. A caller that needs a per-call customization has no
    fast path here; that would be a different marker with the prefix as a
    genuine operand, and no consumer has asked for one.

    It stays an OPERAND of the region nonetheless, not a literal folded into
    the body: the operand-ABI rule (docs/reference/conventions.md) is that a
    host array captured by a marked body is lifted into an unnamed operand
    ahead of the named ones, leaving no layout to write down.
    """
    return fused_region(
        _cxof_decomposition(output_size),
        _CXOF128_INITIAL_STATEd,
        fnp.asarray(_customization_prefix(customization)),
        msg,
        fnp.asarray(_PAD.tail(msg.shape[-1])),
        name=ASCON_CXOF128_MARKER,
        version=ASCON_CXOF128_MARKER_VERSION,
        output_size=output_size,
    )


def cxof128(msg: ArrayLike, customization: bytes, output_size: int) -> Array:
    """Ascon-CXOF128 of a batch of equal-length messages under
    `customization`, read out to `output_size` bytes.
    msg: uint8 [B, L] -> [B, output_size].

    Byte-identical to NIST SP 800-232 §5.3 per message. Traced or concrete, on
    the same terms as `digest` — the customization is a host `bytes`, so it
    never makes the message untraceable.

    **Not `xof128` with a setting.** An empty customization does not reduce to
    Ascon-XOF128: the two run different initial states, so they are different
    hashes at every input, including this one.
    """
    if output_size < 1:
        raise ValueError(f"output_size ({output_size}) must be >= 1")
    return ascon_cxof128_bytes(device_message(msg), bytes(customization), output_size)


def digest(msg: ArrayLike) -> fnp.ndarray:
    """Ascon-Hash256 of a batch of equal-length messages. msg: uint8 [B, L]
    -> [B, 32].

    Byte-identical to NIST SP 800-232 per message; the whole digest is
    emitted as the one name-routed `hash_frx.digest.ascon_hash256` marker
    (`ascon_hash256_bytes`).

    **Traced or concrete.** `msg` may be a tracer, so a consumer can hash
    inside its own `@jit` or `vmap` without reaching past the `ByteHash`
    seam: the padding is built from the static length and never reads the
    message (`_PAD`), which is the same property `sha256.digest`
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

    For batched hashing where the messages already live on the device. A
    strictly-sequential caller has no in-package alternative — `hashlib` has no
    Ascon either — so it pays this row's dispatch and its compile per distinct
    length."""

    digest_size = ASCON_HASH256_DIGEST_SIZE

    def __init__(self) -> None:
        super().__init__(FusionPath.from_routing(_routes_to_dedicated_emitter()))

    def digest(self, msg: ArrayLike) -> Array:
        return digest(msg)  # the module-level marker digest above


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_ascon_hash256: type[ByteHash] = AsconHash256


class AsconXof128(DeviceRow):
    """`ByteHash` for device Ascon-XOF128 at a fixed output length.

    **The length is a constructor parameter, not a weakened `digest_size`.**
    `AsconXof128(64)` is a different hash from `AsconXof128(32)` — not one hash
    asked for more bytes — so it rides in the value surface `__eq__`/`__hash__`
    cover and `digest_size` stays the concrete integer the seam promises. The
    family-wide rule, stated once in `docs/reference/conventions.md`; it is why
    the Keccak SHAKEs take no default either, and why this one does not.

    Not a mode flag on `AsconHash256`: the two run different initial states, so
    they are two hashes that happen to share a schedule, which is exactly what
    `_digest_decomposition` expresses.

    No plugin recognizes the marker yet, so `fusion_path` reads `GENERIC` on
    every backend today — the marker inlines, the bytes are the standard's, and
    an emitter landing flips the module flags and nothing here moves.
    """

    def __init__(self, output_size: int) -> None:
        if output_size < 1:
            raise ValueError(f"output_size ({output_size}) must be >= 1")
        self.digest_size = output_size
        super().__init__(FusionPath.from_routing(_routes_to_dedicated_emitter()))

    def digest(self, msg: ArrayLike) -> Array:
        return xof128(msg, self.digest_size)


if TYPE_CHECKING:
    _bh_ascon_xof128: type[ByteHash] = AsconXof128


class AsconCxof128(DeviceRow):
    """`ByteHash` for device Ascon-CXOF128 — the customized XOF (§5.3).

    Takes the customization string and the output length, both construction
    parameters. **The customization is `bytes` rather than an array** for the
    reason `Blake3Keyed`'s key is: the seam's `digest(msg)` takes a message and
    nothing else, so there is nowhere to put a per-call one — it is part of
    which hash this is, it rides in the compiled program's constant pool, and a
    new customization is a new trace.

    Both parameters are in the value surface `__eq__`/`__hash__` cover, so two
    customizations never share a compiled executable.

    **`AsconCxof128(b"", n)` is not `AsconXof128(n)`.** They run different
    initial states — version 4 against version 3 — so they disagree at every
    input, and an empty customization is a customization rather than its
    absence. That is why this is a row and not a keyword on the XOF.

    No plugin recognizes the marker yet, so `fusion_path` reads `GENERIC` on
    every backend today — the marker inlines, the bytes are the standard's, and
    an emitter landing flips the module flags and nothing here moves.
    """

    def __init__(self, customization: bytes, output_size: int) -> None:
        if output_size < 1:
            raise ValueError(f"output_size ({output_size}) must be >= 1")
        if len(customization) > MAX_CUSTOMIZATION_BYTES:
            raise ValueError(
                f"customization is {len(customization)} bytes; §5.3 allows at "
                f"most {MAX_CUSTOMIZATION_BYTES}"
            )
        self._customization = bytes(customization)
        self.digest_size = output_size
        super().__init__(FusionPath.from_routing(_routes_to_dedicated_emitter()))

    def digest(self, msg: ArrayLike) -> Array:
        return cxof128(msg, self._customization, self.digest_size)

    def _parameters(self) -> tuple[object, ...]:
        # Without the customization two rows differing only in it would compare
        # EQUAL and share a trace — the failure the base class documents, and
        # what `row_conformance_test` builds two instances to catch.
        return (*super()._parameters(), self._customization)


if TYPE_CHECKING:
    _bh_ascon_cxof128: type[ByteHash] = AsconCxof128

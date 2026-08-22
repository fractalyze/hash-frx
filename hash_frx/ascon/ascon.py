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
`_permutation` states why the grid, and not ten loose half arrays, is the
shape that survives lowering.

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
from frx import Array, lax
from frx.typing import ArrayLike

from hash_frx.byte_hash import device_message
from hash_frx.fusion import FusionPath, fused_region, routing
from hash_frx.word import pack_le, roll, split, unpack_le
from hash_frx.word64 import rotr64

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

U32 = fnp.uint32

# A 64-bit quantity as (low 32 bits, high 32 bits) of equal shape. The state
# is one such pair of uint32 [B, 5] grids — word i in column i.
_Lane = tuple[Array, Array]

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
_ROUNDS = 12  # §5.1: every phase runs Ascon-p[12]

# Round i of Ascon-p[rnd] XORs c_i = const_{16-rnd+i} into S2 (§3.2, eqs.
# 3-4; Table 5) — for rnd = 12 that is const_4..const_15, listed here in
# round order. Only the low 8 bits are ever set, so the XOR touches the low
# half alone, as Python-int scalar literals in the traced body (the
# `keccak._iota` arrangement: a scalar is a literal in the emitted module,
# never a host-materialised array, so the operand-lifting rule does not
# apply and the ABI stays three operands).
_ROUND_CONSTANTS = (
    0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B,
)  # fmt: skip

# The Σ_i rotation pairs (§3.4, eqs. 8-12): S_i ^= (S_i ⋙ r1) ^ (S_i ⋙ r2).
_SIGMA_ROTATIONS = ((19, 28), (61, 39), (1, 6), (10, 17), (7, 41))

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


# The three word-position masks the substitution layer applies, as uint32 [5]
# 0/1 grids (`_masks`): which words take the leading XOR of their predecessor,
# which the trailing one, and which is complemented.
_Masks = tuple[Array, Array, Array]


def _masks() -> _Masks:
    """The S-box layer's word-position masks, derived from `iota` on device.

    Host-built 0/1 vectors would be arrays the decomposition materialises,
    which `lax.composite` lifts into unnamed operands ahead of the declared
    ABI — so they are counted on device instead, the `iota` remedy of
    docs/reference/conventions.md (`blake3._counters` is the precedent).
    Computed once per digest and shared by every round.

    - ``pre``: words {0, 2, 4} — the even positions, `(i & 1) ^ 1`.
    - ``post``: words {0, 1, 3} — the positions where i + 1 keeps no bit of
      i (i + 1 a power of two or i = 0), `((i + 1) & i) == 0`.
    - ``word2``: word {2} alone, for the constant XOR and the complement.
    """
    idx = lax.iota(U32, 5)
    one = U32(1)
    pre = (idx & one) ^ one
    post = lax.convert_element_type(((idx + one) & idx) == 0, U32)
    word2 = lax.convert_element_type(idx == U32(2), U32)
    return pre, post, word2


def _substitution(lo: Array, hi: Array, masks: _Masks) -> _Lane:
    """p_S (§3.3): the 5-bit S-box across the five words, grid-wise.

    The substitution is 64 parallel S-box applications with word S_i
    supplying bit plane x_i (eq. 5) — the state is already bitsliced, so no
    per-bit extraction happens anywhere. This is the Figure 3 circuit with
    its word-crossing wires spelled as static rolls of the [B, 5] grid:

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
    pre, post, word2 = masks
    lo = lo ^ (roll(lo, 1, axis=-1) * pre)
    hi = hi ^ (roll(hi, 1, axis=-1) * pre)
    lo = lo ^ ((~roll(lo, -1, axis=-1)) & roll(lo, -2, axis=-1))
    hi = hi ^ ((~roll(hi, -1, axis=-1)) & roll(hi, -2, axis=-1))
    lo = lo ^ (roll(lo, 1, axis=-1) * post)
    hi = hi ^ (roll(hi, 1, axis=-1) * post)
    invert = word2 * U32(0xFFFFFFFF)
    return lo ^ invert, hi ^ invert


def _linear_diffusion(lo: Array, hi: Array) -> _Lane:
    """p_L (§3.4): S_i ^= (S_i ⋙ r1) ^ (S_i ⋙ r2), the Σ_i pairs — the one
    step where the halves interact, through the shared `word64.rotr64`. Each
    word carries its own rotation pair, so the grid splits into static
    columns and stacks back, the way Grøstl's ShiftBytes splits rows."""
    out_lo, out_hi = [], []
    for i, (r1, r2) in enumerate(_SIGMA_ROTATIONS):
        w = (lo[:, i], hi[:, i])
        a = rotr64(w, r1)
        b = rotr64(w, r2)
        out_lo.append(w[0] ^ a[0] ^ b[0])
        out_hi.append(w[1] ^ a[1] ^ b[1])
    return fnp.stack(out_lo, axis=-1), fnp.stack(out_hi, axis=-1)


def _permutation(lo: Array, hi: Array, masks: _Masks) -> _Lane:
    """Ascon-p[12] (§3) on the (lo, hi) uint32 [B, 5] grids: twelve rounds
    of p_C -> p_S -> p_L. The round loop is a Python-unrolled `for` — the
    count is static and small, and a `lax` loop would be a control-flow
    boundary (docs/reference/conventions.md). p_C touches only S2's low
    half: the constants are 8-bit, masked to word 2 like the complement.

    The grid is load-bearing for the lowering, not a style choice. Spelled
    over ten loose [B] half arrays the whole digest is one deep element-wise
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
    for rc in _ROUND_CONSTANTS:
        lo = lo ^ (masks[2] * U32(rc))
        lo, hi = _substitution(lo, hi, masks)
        lo, hi = _linear_diffusion(lo, hi)
    return lo, hi


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
    as scalar literals in the body (`_ROUND_CONSTANTS` states why that is
    lifting-safe), and the S-box masks are derived from `iota` on device
    (`_masks`); the three-operand count is pinned in `ascon_test`. `tail` is
    derivable from the static L — a recognizing emitter reads it rather than
    re-deriving it — and load-bearing for the inlined decomposition.
    """

    def decomposition(init: Array, msg: Array, tail: Array, **_attrs: object) -> Array:
        b = msg.shape[0]
        padded = fnp.concatenate(
            [msg, fnp.broadcast_to(tail, (b, tail.shape[0]))], axis=-1
        )
        # Blocks as (lo, hi) uint32 pairs, packed little-endian (§A.1):
        # [B, nblocks, 2], [..., 0] the low half.
        words = pack_le(padded.reshape(b, padded.shape[-1] // _RATE, _RATE))
        lo = fnp.broadcast_to(init[:, 0], (b, 5))
        hi = fnp.broadcast_to(init[:, 1], (b, 5))
        masks = _masks()
        # Absorbing (§5.1): every 64-bit block XORs into S0 — column 0,
        # patched by slice + concatenate so the other four words see no op
        # (the `keccak._patch_lane_zero` spelling and its reasoning) — each
        # block followed by the permutation. Algorithm 5 defers the last
        # block's permutation to the squeeze; same schedule.
        for i in range(padded.shape[-1] // _RATE):  # static, small
            lo = fnp.concatenate([lo[:, :1] ^ words[:, i, :1], lo[:, 1:]], axis=-1)
            hi = fnp.concatenate([hi[:, :1] ^ words[:, i, 1:], hi[:, 1:]], axis=-1)
            lo, hi = _permutation(lo, hi, masks)
        # Squeezing (§5.1): H_0..H_3 read S0 with the permutation between
        # reads; H = H_0 ‖ H_1 ‖ H_2 ‖ H_3 (eq. 63), each word written back
        # little-endian.
        h = [lo[:, 0], hi[:, 0]]
        for _ in range(3):
            lo, hi = _permutation(lo, hi, masks)
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


class AsconHash256:
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
        # Read per instance rather than pinned on the class: the emitter
        # switch is a property of the pin and the backend, and a value read at
        # import would pin the answer before anything could vary it.
        self.fusion_path = FusionPath.from_routing(_routes_to_dedicated_emitter())

    def digest(self, msg: ArrayLike) -> Array:
        return digest(msg)  # the module-level marker digest above

    def __eq__(self, other: object) -> bool:
        # By type, because Ascon-Hash256 is parameterless — the `Sha256`
        # form, stated there: `type(other) is not type(self)` rather than
        # `isinstance`, which is asymmetric under subclassing and blocks
        # Python's reflected-`__eq__` fallback.
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_ascon_hash256: type[ByteHash] = AsconHash256

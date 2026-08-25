# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""RIPEMD-160 over uint32 lanes, authored in frx — byte-identical to the
designers' specification (H. Dobbertin, A. Bosselaers, B. Preneel,
"RIPEMD-160: A Strengthened Version of RIPEMD", FSE 1996; the definitions are
published as pseudocode at
https://homes.esat.kuleuven.be/~bosselae/ripemd/rmd160.txt, which every
constant below cites; ISO/IEC 10118-3-registered). Bitcoin's HASH160 is
RIPEMD-160(SHA-256(x)) — P2PKH/P2SH addresses — which is the workload that
puts this hash next to `sha256` in this package.

A Merkle–Damgård ARX hash over a 5-word chain: two parallel lines (left and
right) of five 16-step rounds each walk the same 16 message words in different
orders with different rotations and added constants, and the block's result
cross-combines the two lines with the incoming chain. **Everything is
little-endian** — the byte-to-word packing (`word.pack_le`, shared with FIPS
202 and BLAKE3), the digest serialization, and the 64-bit length field in the
padding — the exact opposite byte order of SHA-2, and the documented trap: a
SHA-2-shaped reading of the padding or the packing produces a hash that is
wrong on every input longer than none.

Bulk-parallel by construction, like `sha256`: a batch of `B` equal-length
messages is hashed in one data-parallel call, every message advancing
independently through the shared schedule. The two lines' chains ride as
uint32 `[B]` lanes; every step is element-wise arithmetic on them plus a
static column read of the message words — the r/s schedules are static Python
tuples indexed by the unrolled Python loop, so no gather, dynamic index, or
table array ever reaches the body.

Contract: `digest(msg)` takes uint8 `[B, L]` and returns uint8 `[B, 20]`
digests. Length `L` is static, so the padding is data-independent: a host tail
built from the length alone (`_PAD`), concatenated on — which is what
lets `msg` itself be traced. Requires no x64; all arithmetic is uint32 (wraps
mod 2^32 in XLA).

No host `ByteHash` row ships from here, unlike SHA-2/SHA-3: OpenSSL 3 moved
RIPEMD-160 to the legacy provider, so `hashlib.new("ripemd160")` raises on
most current builds, and a host row that fails by default is worse than none
(issue #189). The differential partner is the testonly
`hash_frx.testing.host_ripemd160.HostRipemd160` over the pure-Python oracle.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import DeviceRow, device_message, padded_batch
from hash_frx.extension.pad import PadRule, Trailer
from hash_frx.fusion import FusionPath, fused_region, routing
from hash_frx.word import pack_le, rotl, unpack_le

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

U32 = fnp.uint32

RIPEMD160_MARKER = "hash_frx.digest.ripemd160"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# in `ripemd160_bytes`. XLA recognizes a marker by name + attributes and
# deliberately does not gate on the version, which exists so a contract change
# can stage without a rename once a recognizer ships (`hash_frx.fusion`); an
# ABI change ships as a NEW name, the way `sha256_bytes` did.
RIPEMD160_MARKER_VERSION = 1

RIPEMD160_DIGEST_SIZE = 20

# Whether the pinned Fractalyze XLA plugin ships a dedicated RIPEMD-160
# emitter, and on which backends. None exists yet — the pre-emitter half of
# carries the family-wide rationale), the posture the other emitterless
# digests hold: both flags flip together with the `frx>=` floor in
# `pyproject.toml` when an emitter lands, and `fusion_path_test`'s matrix
# law holds them to agree. The marker is emitted regardless — there is no
# per-block routing alternative for a whole-hash digest — and unrecognized
# it inlines its decomposition: right bytes, `GENERIC` fusion path.
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


_BLOCK = 64  # 512-bit message blocks, 16 words of 32 bits

# Initial chaining value h0..h4 (the designers' pseudocode, "initial value" —
# MD4's four words extended by a fifth). Host-built once and threaded through
# the marked region as an operand.
_H0 = np.array(
    [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0],
    dtype=np.uint32,
)
_H0d = fnp.asarray(_H0)

# Added constants K(j) / K'(j), one per round of sixteen: 0, then the integer
# parts of 2^30·sqrt(2, 3, 5, 7) on the left line; 2^30·cbrt(2, 3, 5, 7),
# then 0 on the right (the designers' pseudocode, "added constants"). Python
# ints deliberately: each enters the body as a scalar literal, never as a
# host-materialised array, so none of them disturbs the operand ABI.
_K_LEFT = (0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E)
_K_RIGHT = (0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000)

# Message-word selection r(j) / r'(j) and left-rotation amounts s(j) / s'(j),
# indexed by the flat 0..79 counter — the four tables of the designers'
# pseudocode, verbatim. Static Python tuples deliberately: the unrolled loop
# reads them at trace time, so a schedule entry becomes a static column slice
# or a shift amount rather than an indexed read of a table array (the
# no-gather rule, docs/reference/conventions.md).
_R_LEFT = (
    # fmt: off
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13,
    # fmt: on
)
_R_RIGHT = (
    # fmt: off
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11,
    # fmt: on
)
_S_LEFT = (
    # fmt: off
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6,
    # fmt: on
)
_S_RIGHT = (
    # fmt: off
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11,
    # fmt: on
)

# How this family pads, as the axes `extension/md.py` names.
# the designers' pseudocode — little-endian throughout, the opposite of SHA-2.
_PAD = PadRule(64, Trailer.BIT_LENGTH, big_endian=False)


def _f1(x: Array, y: Array, z: Array) -> Array:
    return x ^ y ^ z


def _f2(x: Array, y: Array, z: Array) -> Array:
    return (x & y) | (~x & z)


def _f3(x: Array, y: Array, z: Array) -> Array:
    return (x | ~y) ^ z


def _f4(x: Array, y: Array, z: Array) -> Array:
    return (x & z) | (y & ~z)


def _f5(x: Array, y: Array, z: Array) -> Array:
    return x ^ (y | ~z)


# The five nonlinear functions (the designers' pseudocode, "nonlinear
# functions at bit level: exor, mux, -, mux, -"), selected by the 16-step
# round: the left line walks them forward, the right line in reverse (the
# pseudocode's f(79-j)).
_F = (_f1, _f2, _f3, _f4, _f5)


def _compress(state: Array, x: Array) -> Array:
    """One block: chain [B, 5] + message words x [B, 16] (little-endian) ->
    chain [B, 5].

    The 80-step schedule of both lines is a Python-unrolled `for` — the count
    is static, and unrolling is what turns each r/s table entry into a static
    column slice / shift literal (a `fori_loop` would need a gather into the
    tables; a `lax` loop is also a control-flow boundary,
    docs/reference/conventions.md). The two chains ride as ten uint32 [B]
    lanes and re-stack only at the final combination, so the body stays
    element-wise arithmetic over compact carries.
    """
    al, bl, cl, dl, el = (state[:, i] for i in range(5))
    ar, br, cr, dr, er = al, bl, cl, dl, el
    for j in range(80):
        rnd = j // 16
        # T := rol_s(j)(A + f(j, B, C, D) + X[r(j)] + K(j)) + E;
        # A := E; E := D; D := rol_10(C); C := B; B := T (per line).
        t = (
            rotl(
                al + _F[rnd](bl, cl, dl) + x[:, _R_LEFT[j]] + U32(_K_LEFT[rnd]),
                _S_LEFT[j],
            )
            + el
        )
        al, bl, cl, dl, el = el, t, bl, rotl(cl, 10), dl
        t = (
            rotl(
                ar + _F[4 - rnd](br, cr, dr) + x[:, _R_RIGHT[j]] + U32(_K_RIGHT[rnd]),
                _S_RIGHT[j],
            )
            + er
        )
        ar, br, cr, dr, er = er, t, br, rotl(cr, 10), dr
    # T := h1 + C + D'; h1 := h2 + D + E'; h2 := h3 + E + A';
    # h3 := h4 + A + B'; h4 := h0 + B + C'; h0 := T — the cross-line
    # combination that rotates the register roles by one.
    h0, h1, h2, h3, h4 = (state[:, i] for i in range(5))
    return fnp.stack(
        [h1 + cl + dr, h2 + dl + er, h3 + el + ar, h4 + al + br, h0 + bl + cr],
        axis=1,
    )


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and a Merkle commit emits one digest call per tree level — so the
# uncached re-trace of the 160-step body would dominate the first-trace floor
# (cf. sha256_bytes and grostl256_bytes). `inline=True` splices the cached
# jaxpr into the enclosing trace, so the emitted module (one composite marker
# per digest) is unchanged.
@partial(frx.jit, inline=True)
def ripemd160_bytes(msg: Array) -> Array:
    """Whole-message RIPEMD-160 over raw bytes as ONE marked region:
    uint8 [B, L] -> uint8 [B, 20], padding, word packing, the compression
    chain and the digest serialization all inside the marker.

    A name-routed digest marker, so it is exempt from the generic
    single-kernel rule (`sha256.sha256_merkle_damgard` states the exemption)
    and the body may chain blocks; the 80 steps of each line are
    Python-unrolled regardless, the count being static. No plugin ships a
    RIPEMD-160 recognizer yet (`_DEDICATED_EMITTER_AVAILABLE`), so today the
    marker inlines its decomposition on every backend — identical bytes, no
    dedicated kernel — and an emitter landing changes the lowering, never the
    value.

    Operands are explicit in the recognizer's positional ABI order:

    ``[0] h0``    uint32 [5]   — the initial chaining value
    ``[1] msg``   uint8 [B, L] — the unpadded message batch
    ``[2] tail``  uint8 [P]    — the MD4-style padding for the static L

    Passing `h0` explicitly (rather than closing over the module constant) is
    the operand-ABI rule in docs/reference/conventions.md: a host-materialised
    array captured by the body would be lifted into an unnamed operand *ahead*
    of these, one per call site, leaving no layout to write down. The K
    constants stay scalar literals and the r/s schedules trace-time tuples, so
    neither ever becomes an operand. `tail` is derivable from the static L — a
    recognizing emitter reads it rather than re-deriving it — and load-bearing
    for the inlined decomposition.
    """

    def decomposition(h0: Array, msg: Array, tail: Array, **_attrs: object) -> Array:
        b = msg.shape[0]
        padded = padded_batch(msg, tail)
        words = pack_le(
            padded.reshape(b, padded.shape[-1] // _BLOCK, _BLOCK)
        )  # [B, nblocks, 16]
        state = fnp.broadcast_to(h0, (b, 5))
        for i in range(words.shape[1]):  # static, small
            state = _compress(state, words[:, i])
        return unpack_le(state)  # little-endian digest serialization: [B, 20]

    return fused_region(
        decomposition,
        _H0d,
        msg,
        fnp.asarray(_PAD.tail(msg.shape[-1])),
        name=RIPEMD160_MARKER,
        version=RIPEMD160_MARKER_VERSION,
    )


def digest(msg: ArrayLike) -> fnp.ndarray:
    """RIPEMD-160 of a batch of equal-length messages. msg: uint8 [B, L] ->
    [B, 20].

    Byte-identical to the designers' specification per message; the whole
    digest is emitted as the one name-routed `hash_frx.digest.ripemd160`
    marker (`ripemd160_bytes`).

    **Traced or concrete.** `msg` may be a tracer, so a consumer can hash
    inside its own `@jit` or `vmap` without reaching past the `ByteHash` seam:
    the padding is built from the static length and never reads the message
    (`_PAD`), which is the same property `sha256.digest` states.
    """
    msg = device_message(msg)
    return ripemd160_bytes(msg)


class Ripemd160(DeviceRow):
    """`ByteHash` for device RIPEMD-160 — `digest` runs the batch through the
    `hash_frx.digest.ripemd160` marker. No plugin recognizes that name yet, so
    `fusion_path` reads `GENERIC` on every backend today: the marker inlines,
    the bytes are the standard's, and an emitter landing flips the module
    flags and nothing here moves.

    For batched hashing where the messages already live on the device — the
    HASH160 batch (address derivation, Bitcoin-proof workloads) being the
    motivating case, downstream of a `sha256` batch that is already there. The
    strictly-sequential caller's alternative is the testonly
    `hash_frx.testing.host_ripemd160.HostRipemd160` — pure-Python, so unlike
    the SHA-2/SHA-3 host rows it does not ship (OpenSSL 3's legacy provider
    took `hashlib`'s binding away)."""

    digest_size = RIPEMD160_DIGEST_SIZE

    def __init__(self) -> None:
        super().__init__(FusionPath.from_routing(_routes_to_dedicated_emitter()))

    def digest(self, msg: ArrayLike) -> Array:
        return digest(msg)  # the module-level marker digest above


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_ripemd160: type[ByteHash] = Ripemd160

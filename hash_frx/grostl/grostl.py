# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Grøstl-256 over uint8 byte planes, authored in frx — byte-identical to the
final-round Grøstl specification ("Grøstl — a SHA-3 candidate", v2.0.1,
March 2, 2011, http://www.groestl.info/Groestl.pdf; the *tweaked* algorithm.
Section references below are to that document).

An AES-lineage Merkle–Damgård byte hash: the compression is built from two
fixed 512-bit AES-like permutations P and Q — AddRoundConstant, SubBytes (the
AES S-box), ShiftBytes, MixBytes over the Rijndael field, ten rounds each —
as f(h, m) = P(h ⊕ m) ⊕ Q(m) ⊕ h, with the output transformation
Ω(h) = trunc_256(P(h) ⊕ h). Its value in this package is the GF(2^8)
compression example bridging toward the binary-field direction; SHA-3 demand
belongs to the Keccak family.

Bulk-parallel by construction, like `sha256`: a batch of `B` equal-length
messages is hashed in one data-parallel call, every message advancing
independently through the shared P/Q schedule. The state rides as uint8
`[B, 8, 8]` (row, column) and every round step is element-wise, a static
slice, or a static roll — the S-box included, which is a bitsliced circuit
over the eight bit planes rather than a 256-entry lookup, so no gather ever
reaches the body (`_sbox` carries the circuit's provenance).

Contract: `digest(msg)` takes uint8 `[B, L]` and returns uint8 `[B, 32]`
digests (the trailing-256-bit truncation of Ω, spec section 3.3). Length `L`
is static, so the padding is data-independent: a host tail built from the
length alone (`_PAD`), concatenated on — which is what lets `msg`
itself be traced. Requires no x64; everything is uint8.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import DeviceRow, at_capacity, capacity, message_length
from hash_frx.extension.md import padded_region
from hash_frx.extension.pad import PadRule, Trailer
from hash_frx.fusion import FusionPath, fused_region, routing
from hash_frx.word import roll, unpack_be

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

U8 = fnp.uint8

GROSTL256_MARKER = "hash_frx.digest.grostl256"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# in `grostl256_bytes`. XLA recognizes a marker by name + attributes and
# deliberately does not gate on the version, which exists so a contract change
# can stage without a rename once a recognizer ships (`hash_frx.fusion`); an
# ABI change ships as a NEW name, the way `sha256_bytes` did.
GROSTL256_MARKER_VERSION = 1

GROSTL256_DIGEST_SIZE = 32

# Whether the pinned Fractalyze XLA plugin ships a dedicated Grøstl emitter.
# `keccak.permutation._DEDICATED_EMITTER_AVAILABLE` carries the family-wide
# rationale: the flag flips together with the `frx>=` floor in `pyproject.toml`,
# and `fusion_path_test`'s matrix law holds it and the backend tuple to agree.
# The marker is emitted regardless — there is no per-block routing alternative
# for a whole-hash digest — and unrecognized it inlines its decomposition:
# right bytes, `GENERIC` fusion path. That fallback is the expensive one for
# this hash rather than a mild one, because what the decomposition costs is
# fusion COUNT rather than op count: every boundary between the inlined rounds
# materializes a `[B, 8, 8]` state.
_DEDICATED_EMITTER_AVAILABLE = True

# Which backends carry that emitter — a different question from the pin, asked
# alongside it. The round core sits in `xla/codegen/emitters/grostl.{h,cc}`
# rather than inside the CPU emitter, so a GPU arm can be written without
# re-deriving it; a backend gaining one joins this tuple and nothing else here
# moves (the keccak/poseidon2 pattern).
_EMITTER_BACKENDS: tuple[str, ...] = ("cpu",)


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry this emitter
    (`fusion.routing`, which carries the rationale)."""
    return routing(_DEDICATED_EMITTER_AVAILABLE, _EMITTER_BACKENDS)


_BLOCK = 64  # ℓ = 512 bits: the state/message-block size for n ≤ 256 (§3.1)
_ROUNDS = 10  # §3.4.6: r = 10 for P_512 and Q_512

# ShiftBytes row rotations (left), σ per §3.4.4.
_SHIFT_P = (0, 1, 2, 3, 4, 5, 6, 7)
_SHIFT_Q = (1, 3, 5, 7, 0, 2, 4, 6)

# iv_256 = the 512-bit big-endian representation of 256: 00 … 00 01 00 (§3.5).
_IV = np.zeros(_BLOCK, dtype=np.uint8)
_IV[62] = 0x01


# The AddRoundConstant matrices C[i] for P_512 and Q_512, §3.4.2, as uint8
# [rounds, 8, 8] in the state's (row, column) layout. Both carry the one row
# formula (10·j) ⊕ i: P on row 0 over a zero matrix, Q as its complement on
# row 7 over an ff matrix. Host-built once and threaded through the marked
# region as operands.
_RC_ROW = (np.arange(8, dtype=np.uint8) << 4) ^ np.arange(
    _ROUNDS, dtype=np.uint8
).reshape(-1, 1)
_RC_P = np.zeros((_ROUNDS, 8, 8), dtype=np.uint8)
_RC_P[:, 0, :] = _RC_ROW
_RC_Q = np.full((_ROUNDS, 8, 8), 0xFF, dtype=np.uint8)
_RC_Q[:, 7, :] = 0xFF ^ _RC_ROW

_IVd = fnp.asarray(_IV)
_RC_Pd = fnp.asarray(_RC_P)
_RC_Qd = fnp.asarray(_RC_Q)

# How this family pads, as the axes `extension/md.py` names.
# Grøstl v2.0.1 §3.1 — the trailer counts BLOCKS, not bits.
_PAD = PadRule(64, Trailer.BLOCK_COUNT)


def _to_state(block: Array) -> Array:
    """uint8 [B, 64] -> uint8 [B, 8, 8] (row, column): byte k lands at row
    k mod 8, column k div 8 — the spec's column-wise mapping (§3.4.1). A
    reshape and an axis swap, no data-dependent movement."""
    return fnp.swapaxes(block.reshape(block.shape[0], 8, 8), -1, -2)


def _from_state(state: Array) -> Array:
    """The §3.4.1 mapping inverted: uint8 [B, 8, 8] -> uint8 [B, 64]."""
    return fnp.swapaxes(state, -1, -2).reshape(state.shape[0], _BLOCK)


def _sbox(x: Array) -> Array:
    """SubBytes (§3.4.3): the AES S-box on every byte, as a bitsliced circuit
    over the eight bit planes — no 256-entry table, so no gather (the table
    exists only in the testing oracle).

    The circuit is Boyar–Peralta's depth-16 forward AES S-box, transcribed
    gate for gate from J. Boyar, R. Peralta, "A depth-16 circuit for the AES
    S-box", ePrint 2011/332: Figure 5 (top linear, t1–t27), Figure 7 (shared
    nonlinear middle, m1–m63, with D = U7 in the forward direction) and
    Figure 8 (bottom linear, l0–l29 and the outputs). Their `+` is XOR, `x`
    AND, and `#` XNOR — on {0,1} planes XNOR is XOR-with-1, which is where the
    four `^ one` outputs come from. U0 is the *most significant* input bit and
    S0 the most significant output bit (transcription verified against the
    standard table for all 256 inputs in `grostl/testing/grostl_test.py`).

    Planes are extracted with shifts and masks and repacked with shifts and
    ors; every gate is an element-wise uint8 op, so the whole substitution
    stays data-parallel over the [B, 8, 8] state.
    """
    one = U8(1)
    u0 = (x >> U8(7)) & one
    u1 = (x >> U8(6)) & one
    u2 = (x >> U8(5)) & one
    u3 = (x >> U8(4)) & one
    u4 = (x >> U8(3)) & one
    u5 = (x >> U8(2)) & one
    u6 = (x >> U8(1)) & one
    u7 = x & one
    # Figure 5 — top linear transform.
    t1 = u0 ^ u3
    t2 = u0 ^ u5
    t3 = u0 ^ u6
    t4 = u3 ^ u5
    t5 = u4 ^ u6
    t6 = t1 ^ t5
    t7 = u1 ^ u2
    t8 = u7 ^ t6
    t9 = u7 ^ t7
    t10 = t6 ^ t7
    t11 = u1 ^ u5
    t12 = u2 ^ u5
    t13 = t3 ^ t4
    t14 = t6 ^ t11
    t15 = t5 ^ t11
    t16 = t5 ^ t12
    t17 = t9 ^ t16
    t18 = u3 ^ u7
    t19 = t7 ^ t18
    t20 = t1 ^ t19
    t21 = u6 ^ u7
    t22 = t7 ^ t21
    t23 = t2 ^ t22
    t24 = t2 ^ t10
    t25 = t20 ^ t17
    t26 = t3 ^ t16
    t27 = t1 ^ t12
    d = u7
    # Figure 7 — shared nonlinear middle (D = U7, forward direction).
    m1 = t13 & t6
    m2 = t23 & t8
    m3 = t14 ^ m1
    m4 = t19 & d
    m5 = m4 ^ m1
    m6 = t3 & t16
    m7 = t22 & t9
    m8 = t26 ^ m6
    m9 = t20 & t17
    m10 = m9 ^ m6
    m11 = t1 & t15
    m12 = t4 & t27
    m13 = m12 ^ m11
    m14 = t2 & t10
    m15 = m14 ^ m11
    m16 = m3 ^ m2
    m17 = m5 ^ t24
    m18 = m8 ^ m7
    m19 = m10 ^ m15
    m20 = m16 ^ m13
    m21 = m17 ^ m15
    m22 = m18 ^ m13
    m23 = m19 ^ t25
    m24 = m22 ^ m23
    m25 = m22 & m20
    m26 = m21 ^ m25
    m27 = m20 ^ m21
    m28 = m23 ^ m25
    m29 = m28 & m27
    m30 = m26 & m24
    m31 = m20 & m23
    m32 = m27 & m31
    m33 = m27 ^ m25
    m34 = m21 & m22
    m35 = m24 & m34
    m36 = m24 ^ m25
    m37 = m21 ^ m29
    m38 = m32 ^ m33
    m39 = m23 ^ m30
    m40 = m35 ^ m36
    m41 = m38 ^ m40
    m42 = m37 ^ m39
    m43 = m37 ^ m38
    m44 = m39 ^ m40
    m45 = m42 ^ m41
    m46 = m44 & t6
    m47 = m40 & t8
    m48 = m39 & d
    m49 = m43 & t16
    m50 = m38 & t9
    m51 = m37 & t17
    m52 = m42 & t15
    m53 = m45 & t27
    m54 = m41 & t10
    m55 = m44 & t13
    m56 = m40 & t23
    m57 = m39 & t19
    m58 = m43 & t3
    m59 = m38 & t22
    m60 = m37 & t20
    m61 = m42 & t1
    m62 = m45 & t4
    m63 = m41 & t2
    # Figure 8 — bottom linear transform; S1/S2/S6/S7 are the XNOR outputs.
    l0 = m61 ^ m62
    l1 = m50 ^ m56
    l2 = m46 ^ m48
    l3 = m47 ^ m55
    l4 = m54 ^ m58
    l5 = m49 ^ m61
    l6 = m62 ^ l5
    l7 = m46 ^ l3
    l8 = m51 ^ m59
    l9 = m52 ^ m53
    l10 = m53 ^ l4
    l11 = m60 ^ l2
    l12 = m48 ^ m51
    l13 = m50 ^ l0
    l14 = m52 ^ m61
    l15 = m55 ^ l1
    l16 = m56 ^ l0
    l17 = m57 ^ l1
    l18 = m58 ^ l8
    l19 = m63 ^ l4
    l20 = l0 ^ l1
    l21 = l1 ^ l7
    l22 = l3 ^ l12
    l23 = l18 ^ l2
    l24 = l15 ^ l9
    l25 = l6 ^ l10
    l26 = l7 ^ l9
    l27 = l8 ^ l10
    l28 = l11 ^ l14
    l29 = l11 ^ l17
    s0 = l6 ^ l24
    s1 = l16 ^ l26 ^ one
    s2 = l19 ^ l28 ^ one
    s3 = l6 ^ l21
    s4 = l20 ^ l22
    s5 = l25 ^ l29
    s6 = l13 ^ l27 ^ one
    s7 = l6 ^ l23 ^ one
    return (
        (s0 << U8(7))
        | (s1 << U8(6))
        | (s2 << U8(5))
        | (s3 << U8(4))
        | (s4 << U8(3))
        | (s5 << U8(2))
        | (s6 << U8(1))
        | s7
    )


def _xtime(x: Array) -> Array:
    """Multiply every byte by x in the Rijndael field — reduction polynomial
    x^8 + x^4 + x^3 + x + 1, least significant bit = the x^0 coefficient
    (§3.4.5): shift the byte up (uint8 `<<` drops the shifted-out bit) and
    fold the dropped bit back as 0x1b. The reduction rides a multiply by the
    0/1 high-bit plane rather than a select, staying element-wise uint8."""
    return (x << U8(1)) ^ ((x >> U8(7)) * U8(0x1B))


def _shift_bytes(state: Array, shifts: tuple[int, ...]) -> Array:
    """ShiftBytes (§3.4.4): rotate row i of the [B, 8, 8] state left by
    σ_i. Static row slices plus `word.roll` — a slice pair and a concatenate
    per row, never a gather."""
    rows = [roll(state[:, i, :], -shifts[i], axis=-1) for i in range(8)]
    return fnp.stack(rows, axis=1)


def _mix_bytes(state: Array) -> Array:
    """MixBytes (§3.4.5): left-multiply every state column by
    B = circ(02, 02, 03, 04, 05, 03, 05, 07) over the Rijndael field.

    B[i][j] = c[(j − i) mod 8] with c the circulant's first row, so
    out[i] = Σ_d c[d] ⊗ state[(i + d) mod 8]: an unrolled XOR fold of the
    constant-premultiplied state rolled up the row axis, one static roll per
    diagonal — no reduction, no gather, no per-column loop. The multiples
    02/03/04/05/07 are xtime chains computed once for the whole state
    (03 = 02 ⊕ 01, 05 = 04 ⊕ 01, 07 = 04 ⊕ 03)."""
    x1 = state
    x2 = _xtime(x1)
    x3 = x2 ^ x1
    x4 = _xtime(x2)
    x5 = x4 ^ x1
    x7 = x4 ^ x3
    # The premultiplied diagonals, in c's order — term d rolls up d rows
    # (`roll` is the identity at d = 0).
    diagonals = (x2, x2, x3, x4, x5, x3, x5, x7)
    out = diagonals[0]
    for d in range(1, 8):
        out = out ^ roll(diagonals[d], -d, axis=-2)
    return out


def _permutation(state: Array, rc: Array, shifts: tuple[int, ...]) -> Array:
    """P_512 or Q_512 (§3.4) on a [B, 8, 8] state: _ROUNDS rounds of
    AddRoundConstant → SubBytes → ShiftBytes → MixBytes. The two permutations
    differ only in their constants: `rc` is the round-constant operand
    ([rounds, 8, 8], indexed by a static round number) and `shifts` the static
    σ vector. The round loop is a Python-unrolled `for` — the count is static
    and small, and a `lax` loop would be a control-flow boundary
    (docs/reference/conventions.md)."""
    for r in range(_ROUNDS):
        state = state ^ rc[r]
        state = _sbox(state)
        state = _shift_bytes(state, shifts)
        state = _mix_bytes(state)
    return state


def _compress(h: Array, m: Array, rc_p: Array, rc_q: Array) -> Array:
    """f(h, m) = P(h ⊕ m) ⊕ Q(m) ⊕ h (§3.2, eq. 1), over serialized uint8
    [B, 64] chaining values and message blocks."""
    p = _from_state(_permutation(_to_state(h ^ m), rc_p, _SHIFT_P))
    q = _from_state(_permutation(_to_state(m), rc_q, _SHIFT_Q))
    return p ^ q ^ h


def _block_count_field(msg_bytes: Array) -> Array:
    """Grøstl v2.0.1 §3.1's 64-bit block count, big-endian: uint8 [8].

    The trailer this family strengthens with is the number of blocks the PADDED
    message occupies, not the bit length the other Merkle-Damgard rows write —
    `PadRule.tail` encodes the same value on the static path, and this is its
    counterpart for a traced length.

    The high word is zero rather than derived. The count rides as an int32, so
    a message is under 2^31 bytes and its block count under 2^25, which no
    32-bit half can overflow — where SHA-256's field has to split, because a
    byte count near 2^31 has a BIT length near 2^34 and one word would wrap.
    Stacking two words anyway is what makes the field eight bytes wide by
    construction rather than by an assumption about `unpack_be`'s shape.
    """
    blocks = _PAD.nblocks(msg_bytes.astype(fnp.uint32))
    return unpack_be(fnp.stack([fnp.uint32(0), blocks]))


def _runtime_padded_blocks(buf: Array, length: Array) -> Array:
    """The first `length` bytes of each row, padded to whole blocks: uint8
    [B, LMAX] plus an int32 scalar -> uint8 [B, NB * 64].

    What `_PAD.tail` builds from a static length, built from a traced one: the
    tail cannot be a host constant when the length is runtime data, so the
    region comes from `md.padded_region` — the same select `MdStream.finalize`
    pads with, widened from its two blocks to every block the buffer could
    need. `sha256._runtime_padded_words` is the sibling and differs only in
    packing its result into big-endian words; Grøstl compresses bytes, so this
    one hands the region back as it is.

    This is the marked region's decomposition rather than a path `digest` takes:
    where the marker is recognized the emitter replaces it. The speculation it
    costs — every block the buffer could need is compressed and the ones past
    the message selected away — is exactly the data-dependent-length cost the
    emitter exists to avoid paying, which is why this is written for
    correctness and the floor below refuses the wheels that would run it.
    """
    lmax = buf.shape[-1]
    pos = fnp.arange(_PAD.nblocks(lmax) * _PAD.block_size, dtype=fnp.int32)
    # Bytes at or past `length` are padding, so the message read is clamped into
    # range rather than guarded: every lane it could spoil is selected away.
    content = buf[:, fnp.clip(pos, 0, lmax - 1)]
    active = _PAD.nblocks(length) * fnp.int32(_PAD.block_size)
    return padded_region(_PAD, content, length, active, _block_count_field(length))


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and a Merkle commit emits one digest call per tree level — so the
# uncached re-trace of the 20-permutation body would dominate the first-trace
# floor (cf. sha256_bytes and poseidon2._permute_body). `inline=True` splices
# the cached jaxpr into the enclosing trace, so the emitted module (one
# composite marker per digest) is unchanged.
@partial(frx.jit, inline=True)
def grostl256_bytes(buf: Array, length: Array) -> Array:
    """Whole-message Grøstl-256 over the live prefix of a capacity buffer, as
    ONE marked region: uint8 [B, LMAX] with an int32 scalar byte count -> uint8
    [B, 32], padding, the compression chain and Ω all inside the marker.

    The digest is of `buf[:, :length]`; `LMAX` sizes the allocation and nothing
    else, since the emitter loops on `length` and never reads past it. That is
    the whole point of the form — one compiled kernel serves every length the
    buffer can hold, where a length carried in the message SHAPE compiles once
    per length.

    A name-routed digest marker, so it is exempt from the generic single-kernel
    rule (`sha256.sha256_merkle_damgard` states the exemption) and the body may
    chain blocks; the ten rounds of each permutation are Python-unrolled
    regardless, the count being static and small. Where the pinned plugin
    carries the emitter (`_EMITTER_BACKENDS`) the marker lowers to one kernel;
    elsewhere it inlines its decomposition — identical bytes, no dedicated
    kernel, `GENERIC` fusion path — the emitter changing the lowering, never the
    value.

    **This form REPLACES the static-tail one; there is no routing arm.** The
    decomposition derives its block count from runtime data, so it speculates
    every block the buffer could hold and is therefore *slower* than the
    static-tail form wherever the marker is not recognized — not merely
    un-fused. What keeps that interval from shipping is the `frx>=` floor in
    `pyproject.toml`, raised in the change that swapped this form in. A new
    plugin meeting an old producer needs no arm either: it still recognizes
    `tail u8[P]` and keeps routing.

    `length` rides as an int32 SCALAR — `np.int32`, not a Python int and not a
    device array, for the reasons `sha256.sha256_bytes` sets out: a Python int
    is weakly typed and an x64 build widens it to `s64[]`, which the recognizer
    declines, and a device scalar costs a transfer per call. It is one scalar
    for the whole batch, so every row of `buf` is hashed at the same length —
    the `ByteHash` seam is `uint8[B, L]`, equal-length by construction.

    Operands are explicit in the recognizer's positional ABI order:

    ``[0] iv``    uint8 [64]           — h_0 = iv_256 (§3.5)
    ``[1] rc_p``  uint8 [rounds, 8, 8] — P's AddRoundConstant matrices (§3.4.2)
    ``[2] rc_q``  uint8 [rounds, 8, 8] — Q's
    ``[3] buf``   uint8 [B, LMAX]      — the capacity buffer
    ``[4] len``   int32 []             — live bytes, the digest is of buf[:, :len]

    Operand 4 is what tells the two forms apart on the producer side as well as
    in the recognizer: `len s32[]` against `tail u8[P]` is disjoint in element
    type AND rank. `GROSTL256_MARKER_VERSION` stays 1 — the rewriter never
    reads `composite.version`, so it cannot select an operand form.

    Passing every table explicitly (rather than closing over the module
    constants) is the operand-ABI rule in docs/reference/conventions.md: a
    host-materialised array captured by the body would be lifted into an unnamed
    operand *ahead* of these, one per call site, leaving no layout to write down.
    """
    if buf.shape[-1] < 1:
        # The recognizer's floor, restated where it can still be a clear error.
        # It declines a zero-width buffer, which would leave the decomposition
        # to run — and that one indexes the message through a clamp with no byte
        # to clamp to. The empty message is `length = 0` in a buffer of at least
        # one byte, never a buffer of none.
        raise ValueError(
            f"buf must be uint8 [B, LMAX >= 1], got width {buf.shape[-1]}: an "
            "empty message is length 0 in a non-empty buffer"
        )

    def decomposition(
        iv: Array, rc_p: Array, rc_q: Array, msg: Array, ln: Array, **_attrs: object
    ) -> Array:
        padded = _runtime_padded_blocks(msg, ln)
        live = _PAD.nblocks(ln)
        h = fnp.broadcast_to(iv, (msg.shape[0], _BLOCK))
        # The block count is runtime data, so every block the buffer could need
        # is compressed and the ones past the message selected away. Static and
        # small, and never the routed path (`_runtime_padded_blocks` says why).
        for i in range(padded.shape[-1] // _BLOCK):
            block = padded[:, i * _BLOCK : (i + 1) * _BLOCK]
            h = fnp.where(i < live, _compress(h, block, rc_p, rc_q), h)
        # Ω(h) = trunc_256(P(h) ⊕ h): the trailing 32 bytes (§3.3).
        p = _from_state(_permutation(_to_state(h), rc_p, _SHIFT_P))
        return (p ^ h)[:, _BLOCK - GROSTL256_DIGEST_SIZE :]

    return fused_region(
        decomposition,
        _IVd,
        _RC_Pd,
        _RC_Qd,
        buf,
        length,
        name=GROSTL256_MARKER,
        version=GROSTL256_MARKER_VERSION,
    )


def digest(msg: ArrayLike) -> fnp.ndarray:
    """Grøstl-256 of a batch of equal-length messages. msg: uint8 [B, L] ->
    [B, 32].

    Byte-identical to the final-round specification per message; the whole
    digest is emitted as the one name-routed `hash_frx.digest.grostl256`
    marker (`grostl256_bytes`).

    Hashed out of a `byte_hash.capacity` buffer, so a host caller whose lengths
    vary compiles once per WIDTH rather than once per length: the live byte
    count reaches the marker as an operand and the bytes past it are never read.
    The widening runs on the host — on device it would itself be an eager op
    keyed on `L`, trading one compile per length for a cheaper one rather than
    removing it (`byte_hash.at_capacity` carries the measurement).

    **Traced or concrete.** `msg` may be a tracer, so a consumer can hash
    inside its own `@jit` or `vmap` without reaching past the `ByteHash` seam.
    The padding is built from the length and never reads the message, which is
    the property that allows it — the length is now the marker's operand rather
    than the message's shape, and `_runtime_padded_blocks` builds the same three
    regions in-graph that `_PAD.tail` built on the host.
    """
    length = message_length(msg)
    buf = at_capacity(msg, capacity(msg, _PAD.block_size))
    return grostl256_bytes(buf, np.int32(length))


class Grostl256(DeviceRow):
    """`ByteHash` for device Grøstl-256 — `digest` runs the batch through the
    `hash_frx.digest.grostl256` marker. `fusion_path` reads `DEDICATED` where
    the pinned plugin carries the emitter and `GENERIC` where it does not; the
    marker rides either way and the bytes are the standard's, so a backend
    gaining an arm moves the module flags and nothing here.

    For batched hashing where the messages already live on the device. The
    strictly-sequential caller's alternative is the testonly
    `grostl.testing.host_grostl256.HostGrostl256` — pure-Python, so unlike
    the SHA-2/SHA-3 host rows it does not ship (`hashlib` has no Grøstl)."""

    digest_size = GROSTL256_DIGEST_SIZE

    def __init__(self) -> None:
        super().__init__(FusionPath.from_routing(_routes_to_dedicated_emitter()))

    def digest(self, msg: ArrayLike) -> Array:
        return digest(msg)  # the module-level marker digest above


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_grostl256: type[ByteHash] = Grostl256

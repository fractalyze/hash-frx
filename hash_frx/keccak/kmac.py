# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""KMAC128 / KMAC256 and their XOF variants — the NIST-blessed Keccak MAC.

SP 800-185 section 4. KMAC is cSHAKE with the function name fixed to `"KMAC"`,
over a message that carries a length-encoded key in front of it and an encoding
of the requested output length behind it:

    newX = bytepad(encode_string(K), rate) ‖ X ‖ right_encode(L)
    KMAC(K, X, L, S) = cSHAKE(newX, L, "KMAC", S)

**It is not HMAC and cannot be built from it.** They share the word MAC and
nothing else: HMAC (`adapter/hmac.py`) is the ipad/opad two-pass construction
over a block-oriented hash, and KMAC is a sponge absorbing a length-encoded key.
There is no parameterization of one that yields the other, which is worth
writing down so it is not re-proposed — `adapter/hmac.py` carries the same note
from its side.

**`right_encode(L)` is what makes KMAC and KMACXOF two functions.** Section
4.3.1 defines the XOF form as the same construction with the encoded output
length set to *zero* rather than to the length actually requested. So
`Kmac128(k, output_size=32)` and `KmacXof128(k, output_size=32)` absorb
different final bytes and produce entirely different digests — this is not one
function truncated two ways, and it is the reason the two are separate types.
NIST publishes both sample sets over identical inputs, and
`testing/kmac_test.py` pins that all six pairs disagree.

**`L` is in bits.** `right_encode(8 * output_size)`, not `right_encode(32)` —
the same unit the rest of SP 800-185 counts in (`encodings.py`). A byte count
here is the failure mode that module documents: same shape, same block count,
different digest.

**The cSHAKE fallback is unreachable from here**, because `N` is the constant
`"KMAC"` and section 3.3's fallback needs *both* `N` and `S` empty. So every
KMAC absorbs the `0x04` domain byte whatever the customization is, and an empty
`S` is a customization rather than its absence — the opposite of what the same
argument does one layer down.

**The key is `bytes` on a row and an operand in the free functions.** A row is a
`ByteHash`, whose `digest(msg)` takes a message and nothing else, so the key is
part of *which hash this is* — `Blake3Keyed`'s and `Blake2bKeyed`'s arrangement,
with the two consequences those docstrings state: new key means a new value to
compare on, and secret material sits in a plain attribute nothing erases.

That is the wrong shape for the case a MAC actually has most often, which is a
key that varies per call. So the free functions below take the key as an
*operand*: `kmac128(key_array, msg, 32)` compiles once and re-keys without
re-tracing, because only the key's *length* reaches the encoding and a length is
a shape. The key's bytes never enter `left_encode`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import device_message
from hash_frx.keccak.byte_hashes import SHAKE128_RATE, SHAKE256_RATE, _KeccakHash
from hash_frx.keccak.cshake import CSHAKE_SUFFIX, _prefix_block
from hash_frx.keccak.encodings import (
    bytepad,
    bytepad_encoded_split,
    encode_string,
    right_encode,
)
from hash_frx.keccak.sponge import KeccakSponge

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

# Section 4.3's `N`. Fixed, and reserved to NIST by section 3.4 — this is one of
# the four names that section exists to allocate.
KMAC_NAME = b"KMAC"


def _framing(
    rate: int, output_size: int, customization: bytes, xof: bool
) -> tuple[bytes, bytes]:
    """§4.3's two constants: the cSHAKE prefix that opens the absorb, and the
    `right_encode(L)` that closes it.

    `N` is the constant "KMAC", so the prefix is never cSHAKE's empty-`N`-and-`S`
    fallback — the domain byte is `0x04` whatever the customization is. The tail
    is `right_encode(0)` for the XOF form, which is the whole of §4.3.1 and the
    whole difference between the two functions.

    `L` is in BITS. A byte count is another legal encoding of a plausible number,
    so the wrong unit is a self-consistent different hash (`encodings.py`).
    """
    return (
        _prefix_block(KMAC_NAME, customization, rate),
        right_encode(0 if xof else 8 * output_size),
    )


def _const(data: bytes, rows: int) -> Array:
    """Host bytes as one broadcast device row."""
    return fnp.broadcast_to(
        fnp.asarray(np.frombuffer(data, dtype=np.uint8)), (rows, len(data))
    )


def _bound_message(head: bytes, message: Array, tail: bytes) -> Array:
    """§4.3's `newX` when the key is known on the host: the whole head — cSHAKE
    prefix and key block together — collapses into one constant, so this is a
    single three-way concatenate that XLA constant-folds through the first rate
    block's permutation."""
    rows = message.shape[0]
    return fnp.concatenate([_const(head, rows), message, _const(tail, rows)], axis=-1)


def _operand_message(
    prefix: bytes, key: ArrayLike, message: Array, tail: bytes, rate: int
) -> Array:
    """§4.3's `newX` when the key is an operand, so one compiled program serves
    every key value.

    `bytepad(encode_string(K), rate)` is split around the key by
    `encodings.bytepad_encoded_split`, which owns that arithmetic and is pinned
    against `bytepad` itself — only the key's *length* reaches either encoding,
    and a length is a shape.
    """
    rows = message.shape[0]
    operand = fnp.asarray(key, dtype=fnp.uint8)
    if operand.ndim == 1:
        operand = operand[None, :]
    if operand.ndim != 2:
        raise ValueError(f"key must be [K] or [B, K], got shape {operand.shape}")
    operand = fnp.broadcast_to(operand, (rows, operand.shape[1]))
    framing, fill = bytepad_encoded_split(operand.shape[1], rate)

    parts = [_const(prefix + framing, rows), operand]
    # A key whose framed block already lands on a rate boundary asks for no
    # fill; emitting a zero-width operand would put a dead broadcast in the
    # lowered module for XLA to drop.
    if fill:
        parts.append(fnp.zeros((rows, fill), dtype=fnp.uint8))
    parts += [message, _const(tail, rows)]
    return fnp.concatenate(parts, axis=-1)


def _kmac_with_operand_key(
    rate: int,
    key: ArrayLike,
    msg: ArrayLike,
    output_size: int,
    customization: bytes,
    xof: bool,
) -> Array:
    """The body the four free functions share.

    Builds its own `KeccakSponge` because there is no row to hang one on — the
    reason `sponge.py` gives for building `KeccakF1600` per call rather than at
    import applies here too, and it is ~2 us of host work against a ~500 us
    eager digest.
    """
    if isinstance(key, (bytes, bytearray, memoryview)):
        raise ValueError(
            "key must be an array operand, not bytes: a host key belongs on a "
            "row (Kmac128(key, ...)), where it is part of which hash this is. "
            "Passing bytes here would bake it into the compiled program and "
            "re-compile per key, which is what this surface exists to avoid"
        )
    message = device_message(msg)
    prefix, tail = _framing(rate, output_size, customization, xof)
    return KeccakSponge(rate=rate, suffix=CSHAKE_SUFFIX, output_size=output_size).hash(
        _operand_message(prefix, key, message, tail, rate)
    )


def kmac128(
    key: ArrayLike,
    msg: ArrayLike,
    output_size: int,
    customization: bytes = b"",
) -> Array:
    """KMAC128 with the key as an operand: uint8 `[B, L]` -> `[B, output_size]`.

    `key` may be `[K]` (shared by the batch) or `[B, K]` (per message), and may
    be a tracer — one compiled program serves every key value of a given length.
    """
    return _kmac_with_operand_key(
        SHAKE128_RATE, key, msg, output_size, customization, xof=False
    )


def kmac256(
    key: ArrayLike,
    msg: ArrayLike,
    output_size: int,
    customization: bytes = b"",
) -> Array:
    """KMAC256 with the key as an operand — `kmac128`'s 136-byte-rate sibling."""
    return _kmac_with_operand_key(
        SHAKE256_RATE, key, msg, output_size, customization, xof=False
    )


def kmac_xof128(
    key: ArrayLike,
    msg: ArrayLike,
    output_size: int,
    customization: bytes = b"",
) -> Array:
    """KMACXOF128 — `kmac128` with the encoded output length set to zero, which
    makes it a different function rather than the same one read further."""
    return _kmac_with_operand_key(
        SHAKE128_RATE, key, msg, output_size, customization, xof=True
    )


def kmac_xof256(
    key: ArrayLike,
    msg: ArrayLike,
    output_size: int,
    customization: bytes = b"",
) -> Array:
    """KMACXOF256 — `kmac256`'s XOF form (section 4.3.1)."""
    return _kmac_with_operand_key(
        SHAKE256_RATE, key, msg, output_size, customization, xof=True
    )


class _Kmac(_KeccakHash):
    """The shared body of the four KMAC rows.

    A subclass supplies `_rate` — the rate of the cSHAKE it is built on, and the
    `w` both `bytepad`s fill — and `_xof`, which selects §4.3.1's
    `right_encode(0)` over §4.3's `right_encode(L)`. Both are class attributes,
    and so is `_suffix`: KMAC's domain byte is cSHAKE's `0x04` unconditionally,
    because `N` is the constant "KMAC" and §3.3's both-empty fallback needs an
    empty name. That is why this row does NOT need `_CShake`'s per-instance
    suffix — there the byte is a function of the arguments, here it is a
    function of the type.

    With a key bound to the instance, *every* byte outside the message is fixed
    at construction — the cSHAKE prefix, the key block and the trailing length
    encoding. So the head and tail are built once here rather than per `digest`,
    which is the arrangement `_CShake` states for its own prefix, and the hash
    is one three-way concatenate over `_KeccakHash`'s own sponge.
    """

    _rate: int
    _xof: bool
    _suffix = CSHAKE_SUFFIX

    def __init__(
        self,
        key: bytes,
        customization: bytes = b"",
        *,
        output_size: int,
    ) -> None:
        self._key = bytes(key)
        self._customization = bytes(customization)
        super().__init__(output_size)
        prefix, self._tail = _framing(
            self._rate, output_size, self._customization, self._xof
        )
        # `bytepad(encode_string(K), rate)` straight from `encodings`, not
        # re-derived: the operand path reaches the same bytes through
        # `bytepad_encoded_split`, and `encodings_test` pins the two together.
        self._head = prefix + bytepad(encode_string(self._key), self._rate)

    def digest(self, msg: ArrayLike) -> Array:
        return self._sponge.hash(
            _bound_message(self._head, device_message(msg), self._tail)
        )

    def _parameters(self) -> tuple[object, ...]:
        # The key and the customization are both `which hash this is`, so both
        # are the jit cache key. Two rows that compare equal share a compiled
        # program — for a MAC, that is a key crossing a cache hit.
        return (*super()._parameters(), self._key, self._customization)


class Kmac128(_Kmac):
    """`ByteHash` for device KMAC128 at a fixed output length.

    The key rides on the instance, so this is a MAC that has been *bound* to a
    key. For a key that varies per call — the ordinary MAC case — use
    `kmac128`, where the key is an operand.
    """

    _rate = SHAKE128_RATE
    _xof = False


class Kmac256(_Kmac):
    """`ByteHash` for device KMAC256 at a fixed output length."""

    _rate = SHAKE256_RATE
    _xof = False


class KmacXof128(_Kmac):
    """`ByteHash` for device KMACXOF128 — section 4.3.1's arbitrary-output form.

    Not `Kmac128` read to a different length: the encoded output length absorbed
    at the end is zero here and `8 * output_size` there, so the two disagree at
    every length including the one they were both asked for.
    """

    _rate = SHAKE128_RATE
    _xof = True


class KmacXof256(_Kmac):
    """`ByteHash` for device KMACXOF256 — `Kmac256`'s XOF form."""

    _rate = SHAKE256_RATE
    _xof = True


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md). Named individually
    # because mypy rejects re-annotating one name.
    _bh_kmac128: type[ByteHash] = Kmac128
    _bh_kmac256: type[ByteHash] = Kmac256
    _bh_kmac_xof128: type[ByteHash] = KmacXof128
    _bh_kmac_xof256: type[ByteHash] = KmacXof256

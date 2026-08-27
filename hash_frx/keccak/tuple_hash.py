# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""TupleHash128 / TupleHash256 and their XOF forms — hashing a *sequence* of
strings without the ambiguity a concatenation would carry.

SP 800-185 section 5. TupleHash is cSHAKE with the function name fixed to
`"TupleHash"`, over each element wrapped in `encode_string` and the requested
output length appended:

    z    = encode_string(X[0]) ‖ encode_string(X[1]) ‖ …
    newX = z ‖ right_encode(L)
    TupleHash(X, L, S) = cSHAKE(newX, L, "TupleHash", S)

**The problem it solves, and the reason this is not a `ByteHash`.** Hashing
`("ab", "c")` by concatenating first gives the same bytes as `("a", "bc")` and
as `("abc")` — three different inputs, one digest, and a consumer that builds a
commitment or a transcript that way has a collision it did not design. Length
prefixes fix it, and getting them right is exactly the fiddly part
(`encodings.py`). All three of those tuples produce different digests here, and
`testing/tuple_hash_test.py` asserts that directly.

That is also why this takes a sequence rather than a message. `ByteHash`'s
`digest` is `uint8 [B, L] -> [B, digest_size]` — one flat message per row — and
flattening a tuple into it would destroy the only thing TupleHash provides. So
these are `Row`s and not `ByteHash`es, the position `adapter/hmac.py` holds for
the neighbouring reason: it keys a byte hash rather than being one, this hashes
a tuple of them. Neither implements the seam, both keep the seam's equality
contract, and neither carries a conformance pin — see the note at the bottom.

**`right_encode(L)` splits TupleHash from TupleHashXOF**, exactly as it splits
KMAC from KMACXOF (section 5.3.1, and `kmac.py` for the same argument): the XOF
form encodes zero rather than the length asked for, so the two are different
functions and not one read to different lengths. NIST publishes both over
identical tuples, and all six pairs disagree.

**Every element's length is static, so the framing is host bytes.** A batch's
shape is fixed at trace time, so `encode_string`'s prefix for each element is a
constant and the whole hash is one concatenate of alternating constants and
operands — the elements themselves never leave the device and never reach
`left_encode`. The tuple's *arity* and the per-element *lengths* are part of the
traced program; their *contents* are not.
"""

from __future__ import annotations

from collections.abc import Sequence

import frx.numpy as fnp
from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import DeviceRow, device_message
from hash_frx.keccak.byte_hashes import SHAKE128_RATE, SHAKE256_RATE
from hash_frx.keccak.cshake import CSHAKE_SUFFIX, const_rows, derived_framing
from hash_frx.keccak.encodings import left_encode
from hash_frx.keccak.permutation import KeccakF1600
from hash_frx.keccak.sponge import KeccakSponge

# Section 5.3's `N`. Fixed, and reserved to NIST by section 3.4.
TUPLE_HASH_NAME = b"TupleHash"


def _tuple_message(prefix: bytes, strings: Sequence[ArrayLike], tail: bytes) -> Array:
    """Section 5.3's `newX`, prefixed by cSHAKE's own block.

    Each element contributes `left_encode(8 * len)` — a host constant read off
    its shape — followed by the element itself as an operand. So the result is
    one concatenate over `2n + 2` pieces, alternating constants and batches,
    and no element byte is ever read on the host.

    """
    if not strings:
        raise ValueError(
            "strings must be non-empty: TupleHash of no tuples is not a "
            "construction SP 800-185 section 5 defines"
        )
    batches = [device_message(s) for s in strings]
    rows = batches[0].shape[0]
    for i, batch in enumerate(batches):
        if batch.shape[0] != rows:
            raise ValueError(
                f"every element must carry the same batch size: element {i} has "
                f"{batch.shape[0]} rows, element 0 has {rows}"
            )

    parts: list[Array] = [const_rows(prefix, rows)]
    for batch in batches:
        parts.append(const_rows(left_encode(8 * batch.shape[1]), rows))
        parts.append(batch)
    parts.append(const_rows(tail, rows))
    return fnp.concatenate(parts, axis=-1)


class _TupleHash(DeviceRow):
    """The shared body of the four TupleHash rows.

    A subclass supplies `_rate` — the rate of the cSHAKE it is built on — and
    `_xof`, which selects section 5.3.1's `right_encode(0)` over section 5.3's
    `right_encode(L)`.

    The cSHAKE prefix and the trailing length encoding are fixed by
    `(customization, output_size)`, so both are built once here. What cannot be
    is the per-element framing: it is a function of the shapes `hash` is called
    with, and those belong to the call.

    `fusion_path` is derived from `KeccakF1600` for the reason `_KeccakHash`
    gives — so it cannot disagree with the routing `KeccakSponge.hash` actually
    takes. The whole hash is one `keccak_sponge` region: the framing is
    constants concatenated outside it, exactly as cSHAKE's prefix is.
    """

    _rate: int
    _xof: bool

    def __init__(self, customization: bytes = b"", *, output_size: int) -> None:
        self.digest_size = output_size
        self._customization = bytes(customization)
        self._prefix, self._tail = derived_framing(
            TUPLE_HASH_NAME, self._customization, self._rate, output_size, self._xof
        )
        self._sponge = KeccakSponge(
            rate=self._rate, suffix=CSHAKE_SUFFIX, output_size=output_size
        )
        super().__init__(KeccakF1600().fusion_path)

    def hash(self, strings: Sequence[ArrayLike]) -> Array:
        """Hash a tuple of equal-batch byte strings: a sequence of uint8
        `[B, L_i]` -> uint8 `[B, digest_size]`.

        Named `hash` rather than `digest` because it is not the seam's shape —
        the argument is a sequence, and that is the whole construction.
        """
        message = _tuple_message(self._prefix, strings, self._tail)
        return self._sponge.hash(message)

    def _parameters(self) -> tuple[object, ...]:
        # The customization is `which hash this is`, so it is the jit cache key
        # as much as the output length. The tuple's shape is not a parameter of
        # the row — it belongs to the call, and reaches the trace through the
        # operand avals.
        return (*super()._parameters(), self._customization)


def _tuple_hash(
    rate: int,
    strings: Sequence[ArrayLike],
    output_size: int,
    customization: bytes,
    xof: bool,
) -> Array:
    """Section 5.3 over the Keccak sponge, for the free functions.

    Builds its own `KeccakSponge` because there is no row to hang one on, which
    is the arrangement `kmac.py` states for the same reason.
    """
    prefix, tail = derived_framing(
        TUPLE_HASH_NAME, customization, rate, output_size, xof
    )
    message = _tuple_message(prefix, strings, tail)
    return KeccakSponge(rate=rate, suffix=CSHAKE_SUFFIX, output_size=output_size).hash(
        message
    )


def tuple_hash128(
    strings: Sequence[ArrayLike],
    output_size: int,
    customization: bytes = b"",
) -> Array:
    """TupleHash128 of a sequence of uint8 `[B, L_i]` batches."""
    return _tuple_hash(SHAKE128_RATE, strings, output_size, customization, xof=False)


def tuple_hash256(
    strings: Sequence[ArrayLike],
    output_size: int,
    customization: bytes = b"",
) -> Array:
    """TupleHash256 — `tuple_hash128`'s 136-byte-rate sibling."""
    return _tuple_hash(SHAKE256_RATE, strings, output_size, customization, xof=False)


def tuple_hash_xof128(
    strings: Sequence[ArrayLike],
    output_size: int,
    customization: bytes = b"",
) -> Array:
    """TupleHashXOF128 — section 5.3.1's form, with the encoded output length
    set to zero. A different function from `tuple_hash128`, not the same one
    read further."""
    return _tuple_hash(SHAKE128_RATE, strings, output_size, customization, xof=True)


def tuple_hash_xof256(
    strings: Sequence[ArrayLike],
    output_size: int,
    customization: bytes = b"",
) -> Array:
    """TupleHashXOF256 — `tuple_hash256`'s XOF form."""
    return _tuple_hash(SHAKE256_RATE, strings, output_size, customization, xof=True)


class TupleHash128(_TupleHash):
    """Unambiguous hashing of a sequence of strings, at SHAKE128's rate.

    `TupleHash128(output_size=32).hash([a, b])` is not `Shake128(32)` over
    `a ‖ b`, and that is the point: the split is part of what is hashed.
    """

    _rate = SHAKE128_RATE
    _xof = False


class TupleHash256(_TupleHash):
    """`TupleHash128`'s 136-byte-rate sibling, for a 256-bit security level."""

    _rate = SHAKE256_RATE
    _xof = False


class TupleHashXof128(_TupleHash):
    """Section 5.3.1's arbitrary-output form at SHAKE128's rate.

    Not `TupleHash128` read to a different length: the encoded output length is
    zero here and `8 * output_size` there, so the two disagree at every length
    including the one they were both asked for.
    """

    _rate = SHAKE128_RATE
    _xof = True


class TupleHashXof256(_TupleHash):
    """`TupleHash256`'s XOF form."""

    _rate = SHAKE256_RATE
    _xof = True


# No seam-conformance pin, and — like `adapter/hmac.py`'s — that is a decision.
#
# `docs/reference/conventions.md` asks every implementation module to end with
# `_: type[<Seam>] = <Class>`. `type[ByteHash]` is the annotation that is
# definitely wrong here: it would assert these are byte hashes, and the whole
# argument of this module is that a byte hash takes ONE message, while
# TupleHash's input is a sequence whose division is part of the hash. Pinning
# against `ByteHash` would claim the seam that flattening destroys.
#
# Nor is a `TupleByteHash` Protocol declared to pin against. There is one
# implementation and no consumer asking for the abstraction, which is the
# `Mac`-Protocol argument `hmac.py` makes and declines for the same reason.
# What holds these rows instead is `testing/rows.py`, which registers them so
# `row_conformance_test` asserts the equality contract they DO implement, and
# `testing/tuple_hash_test.py`, which pins the construction against the
# published SP 800-185 vectors.

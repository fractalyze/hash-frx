# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every `Row` this package ships, as one table the conformance suite walks.

The seam's contract is stated once in `byte_hash.py` and implemented by all but
one row in this table, and until now nothing held them to it at once — each family
tested its own rows, so a rule could hold everywhere it was checked and be
missing from the row that shipped last. That is not hypothetical: the two seam
sweeps that became #211 and #215 both had to be re-applied to SM3 and BLAKE2s
afterwards, because those rows merged in parallel with the sweeps.

A row joins this table when it ships. `row_conformance_test` fails if it and
the package disagree, so joining is not optional.

**One table, and every bucket derived from it.** `Row` states the equality
contract and `ByteHash` the digest one, so not every row is a byte hash: `Hmac`
keys a hash rather than being one, and `byte_hash.Row` records that it declares
no `fusion_path` precisely so such an adapter can share the equality contract.
That split decides which cases a row reaches, and it is read off the row rather
than written down twice — a hand-kept second list is one a row can be left off,
which is the failure this file exists to prevent.

`variants` are further instances of the same type, each differing from `make()`
in ONE parameter. They are what make the equality cases mean anything: without
them "two instances compare equal" is satisfied by a row whose `__eq__` ignores
the parameter that distinguishes them. Why that matters is stated once, on
`byte_hash.Row`.

It is a tuple rather than a single alternative because six rows have two
distinguishing parameters. `Blake3Keyed` and friends take `(key, output_size)`,
and with only the key varied the `*super()._parameters()` half of their override
is never exercised: deleting it left the suite green. The two adapter rows are
the same shape — `Mgf1` over `(hash, output_size)` and `Hmac` over
`(hash, block_size)` — and for them the untested half is the hash, which is the
half that decides what the row actually computes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from hash_frx import (
    AsconCxof128,
    AsconHash256,
    AsconXof128,
    Blake2b,
    Blake2bKeyed,
    Blake2s,
    Blake2sKeyed,
    Blake3,
    Blake3DeriveKey,
    Blake3Keyed,
    CShake128,
    CShake256,
    Grostl256,
    Hmac,
    Keccak256,
    Kmac128,
    Kmac256,
    KmacXof128,
    KmacXof256,
    Mgf1,
    Ripemd160,
    Sha3_224,
    Sha3_256,
    Sha3_384,
    Sha3_512,
    Sha256,
    Sha384,
    Sha512,
    Sha512_256,
    Shake128,
    Shake256,
    Sm3,
    TupleHash128,
    TupleHash256,
    TupleHashXof128,
    TupleHashXof256,
)

_KEY_A = bytes(range(32))
_KEY_B = bytes(range(1, 33))


class RowCase(NamedTuple):
    """One shipped row, plus how to build siblings that must not equal it."""

    name: str
    make: Callable[[], Any]
    variants: tuple[Callable[[], Any], ...] = ()


# Every shipped row, in one table. The conformance suite asserts the law over
# it — every row returns an `Array` — rather than over a list of rows anyone
# labelled by hand.
ALL_ROWS: tuple[RowCase, ...] = (
    RowCase("Sha256", Sha256),
    RowCase("Sha512", Sha512),
    RowCase("Sha384", Sha384),
    RowCase("Sha512_256", Sha512_256),
    RowCase("Sm3", Sm3),
    RowCase("Ripemd160", Ripemd160),
    RowCase("Grostl256", Grostl256),
    RowCase("AsconHash256", AsconHash256),
    RowCase("AsconXof128", lambda: AsconXof128(32), (lambda: AsconXof128(64),)),
    RowCase(
        "AsconCxof128",
        lambda: AsconCxof128(b"", 32),
        (lambda: AsconCxof128(b"z", 32), lambda: AsconCxof128(b"", 64)),
    ),
    RowCase(
        "Blake2s", Blake2s, (lambda: Blake2s(20), lambda: Blake2s(32, person=b"p"))
    ),
    RowCase(
        "Blake2b", Blake2b, (lambda: Blake2b(20), lambda: Blake2b(64, person=b"p"))
    ),
    # The keyed rows carry FOUR parameters, so each gets a variant per
    # parameter: a row equal to another it should differ from is that other's
    # jit cache key, and `person` is the half a size-only variant leaves
    # unexercised — which is the trap the BLAKE3 keyed rows below record.
    RowCase(
        "Blake2sKeyed",
        lambda: Blake2sKeyed(_KEY_A),
        (
            lambda: Blake2sKeyed(_KEY_B),
            lambda: Blake2sKeyed(_KEY_A, 20),
            lambda: Blake2sKeyed(_KEY_A, salt=b"s"),
            lambda: Blake2sKeyed(_KEY_A, person=b"p"),
        ),
    ),
    RowCase(
        "Blake2bKeyed",
        lambda: Blake2bKeyed(_KEY_A),
        (
            lambda: Blake2bKeyed(_KEY_B),
            lambda: Blake2bKeyed(_KEY_A, 20),
            lambda: Blake2bKeyed(_KEY_A, salt=b"s"),
            lambda: Blake2bKeyed(_KEY_A, person=b"p"),
        ),
    ),
    RowCase("Sha3_224", Sha3_224),
    RowCase("Sha3_256", Sha3_256),
    RowCase("Sha3_384", Sha3_384),
    RowCase("Sha3_512", Sha3_512),
    RowCase("Keccak256", Keccak256),
    RowCase("Shake128", lambda: Shake128(32), (lambda: Shake128(64),)),
    RowCase("Shake256", lambda: Shake256(64), (lambda: Shake256(32),)),
    # Each parameter varied alone: varying two at once passes while either
    # is ignored, and the customization is exactly what must not be.
    RowCase(
        "CShake128",
        lambda: CShake128(b"s", b"n", output_size=32),
        (
            lambda: CShake128(b"t", b"n", output_size=32),
            lambda: CShake128(b"s", b"m", output_size=32),
            lambda: CShake128(b"s", b"n", output_size=64),
        ),
    ),
    RowCase(
        "CShake256",
        lambda: CShake256(b"s", b"n", output_size=64),
        (
            lambda: CShake256(b"t", b"n", output_size=64),
            lambda: CShake256(b"s", b"m", output_size=64),
            lambda: CShake256(b"s", b"n", output_size=32),
        ),
    ),
    # The key is varied alone alongside the other two: two KMAC rows that
    # compare equal share a compiled program, and for a MAC that is a key
    # crossing a cache hit.
    RowCase(
        "Kmac128",
        lambda: Kmac128(_KEY_A, b"s", output_size=32),
        (
            lambda: Kmac128(_KEY_B, b"s", output_size=32),
            lambda: Kmac128(_KEY_A, b"t", output_size=32),
            lambda: Kmac128(_KEY_A, b"s", output_size=64),
        ),
    ),
    RowCase(
        "Kmac256",
        lambda: Kmac256(_KEY_A, b"s", output_size=64),
        (
            lambda: Kmac256(_KEY_B, b"s", output_size=64),
            lambda: Kmac256(_KEY_A, b"t", output_size=64),
            lambda: Kmac256(_KEY_A, b"s", output_size=32),
        ),
    ),
    RowCase(
        "KmacXof128",
        lambda: KmacXof128(_KEY_A, b"s", output_size=32),
        (
            lambda: KmacXof128(_KEY_B, b"s", output_size=32),
            lambda: KmacXof128(_KEY_A, b"t", output_size=32),
            lambda: KmacXof128(_KEY_A, b"s", output_size=64),
        ),
    ),
    RowCase(
        "KmacXof256",
        lambda: KmacXof256(_KEY_A, b"s", output_size=64),
        (
            lambda: KmacXof256(_KEY_B, b"s", output_size=64),
            lambda: KmacXof256(_KEY_A, b"t", output_size=64),
            lambda: KmacXof256(_KEY_A, b"s", output_size=32),
        ),
    ),
    # TupleHash rows keep the equality contract without implementing the seam —
    # their input is a sequence, so `row_conformance_test`'s `BYTE_HASH_ROWS`
    # half correctly passes them by, and this half still holds their cache key.
    RowCase(
        "TupleHash128",
        lambda: TupleHash128(b"s", output_size=32),
        (
            lambda: TupleHash128(b"t", output_size=32),
            lambda: TupleHash128(b"s", output_size=64),
        ),
    ),
    RowCase(
        "TupleHash256",
        lambda: TupleHash256(b"s", output_size=64),
        (
            lambda: TupleHash256(b"t", output_size=64),
            lambda: TupleHash256(b"s", output_size=32),
        ),
    ),
    RowCase(
        "TupleHashXof128",
        lambda: TupleHashXof128(b"s", output_size=32),
        (
            lambda: TupleHashXof128(b"t", output_size=32),
            lambda: TupleHashXof128(b"s", output_size=64),
        ),
    ),
    RowCase(
        "TupleHashXof256",
        lambda: TupleHashXof256(b"s", output_size=64),
        (
            lambda: TupleHashXof256(b"t", output_size=64),
            lambda: TupleHashXof256(b"s", output_size=32),
        ),
    ),
    RowCase("Blake3", Blake3, (lambda: Blake3(16),)),
    RowCase(
        "Blake3Keyed",
        lambda: Blake3Keyed(_KEY_A),
        (lambda: Blake3Keyed(_KEY_B), lambda: Blake3Keyed(_KEY_A, 16)),
    ),
    RowCase(
        "Blake3DeriveKey",
        lambda: Blake3DeriveKey("ctx a"),
        (lambda: Blake3DeriveKey("ctx b"), lambda: Blake3DeriveKey("ctx a", 16)),
    ),
    # The two adapter rows. Both subclass `Row` directly rather than either
    # base, so both derive into a different set of buckets below than any
    # family row does.
    RowCase(
        "Mgf1",
        lambda: Mgf1(Sha256(), 32),
        # Both parameters, one variant each. The length alone would leave
        # `_parameters`' `self._byte_hash` half unexercised and the hash alone
        # its `*super()._parameters()` half — the trap the BLAKE3 keyed rows
        # above record, and the live one here: `Mgf1(Sha256(), 32)` comparing
        # equal to `Mgf1(Sha512(), 32)` serves one mask's trace for the other's.
        (lambda: Mgf1(Sha512(), 32), lambda: Mgf1(Sha256(), 64)),
    ),
    RowCase(
        "Hmac",
        lambda: Hmac(Sha256()),
        # SM3 shares SHA-256's 64-byte block, so this pair differs in the hash
        # and nothing else; the second differs in the block and nothing else.
        # Varying both at once (`Hmac(Sha512())`, block 128) would pass while
        # comparing on either one alone.
        (lambda: Hmac(Sm3()), lambda: Hmac(Sha256(), 128)),
    ),
)

# Byte hashes: the rows `digest` is asked of.
#
# On `digest` and not `isinstance(row, ByteHash)`, so that
# `RowSeamTest.test_satisfies_the_protocol` still asserts something: `digest` is
# the weakest member of the Protocol, so what stays under test is that
# `digest_size` and `fusion_path` are PRESENT — which a row on neither base has
# to supply itself, and which is the shape an adapter row gets wrong. Deriving
# on the Protocol would make that case assert its own selection criterion.
BYTE_HASH_ROWS: tuple[RowCase, ...] = tuple(
    c for c in ALL_ROWS if hasattr(c.make(), "digest")
)

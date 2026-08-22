# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every `ByteHash` row this package ships, as one list the conformance suite
walks.

The seam's contract is stated once in `byte_hash.py` and implemented thirty-two
times, and until now nothing held all thirty-two to it at once — each family
tested its own rows, so a rule could hold everywhere it was checked and be
missing from the row that shipped last. That is not hypothetical: the two seam
sweeps that became #211 and #215 both had to be re-applied to SM3 and BLAKE2s
afterwards, because those rows merged in parallel with the sweeps.

A row joins this list when it ships. `row_conformance_test` fails if the list
and the package disagree, so joining is not optional.

`variants` are further instances of the same type, each differing from `make()`
in ONE parameter. They are what make the equality cases mean anything: without
them "two instances compare equal" is satisfied by a row whose `__eq__` ignores
the parameter that distinguishes them. Why that matters is stated once, on
`byte_hash.Row`.

It is a tuple rather than a single alternative because four rows have two
distinguishing parameters. `Blake3Keyed` and friends take `(key, output_size)`,
and with only the key varied the `*super()._parameters()` half of their override
is never exercised: deleting it left the suite green.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from hash_frx import (
    AsconHash256,
    Blake2b,
    Blake2s,
    Blake3,
    Blake3DeriveKey,
    Blake3Keyed,
    Grostl256,
    HostBlake2b,
    HostBlake2s,
    HostBlake3,
    HostBlake3DeriveKey,
    HostBlake3Keyed,
    HostSha3_256,
    HostSha3_512,
    HostSha256,
    HostSha384,
    HostSha512,
    HostSha512_256,
    HostShake128,
    HostShake256,
    HostSm3,
    Keccak256,
    Ripemd160,
    Sha3_256,
    Sha3_512,
    Sha256,
    Sha384,
    Sha512,
    Sha512_256,
    Shake128,
    Shake256,
    Sm3,
)
from hash_frx.byte_hash import DeviceRow, HostRow

_KEY_A = bytes(range(32))
_KEY_B = bytes(range(1, 33))


class RowCase(NamedTuple):
    """One shipped row, plus how to build siblings that must not equal it."""

    name: str
    make: Callable[[], Any]
    variants: tuple[Callable[[], Any], ...] = ()


# Every shipped row, in one table. Which of the two tuples below a row lands in
# is NOT restated here: it derives from `DeviceRow`/`HostRow`, so the conformance
# suite asserts "a `DeviceRow` returns an `Array`" — the actual law — rather than
# "the rows I happened to label device do".
ALL_ROWS: tuple[RowCase, ...] = (
    RowCase("Sha256", Sha256),
    RowCase("Sha512", Sha512),
    RowCase("Sha384", Sha384),
    RowCase("Sha512_256", Sha512_256),
    RowCase("Sm3", Sm3),
    RowCase("Ripemd160", Ripemd160),
    RowCase("Grostl256", Grostl256),
    RowCase("AsconHash256", AsconHash256),
    RowCase("Blake2s", Blake2s, (lambda: Blake2s(20),)),
    RowCase("Blake2b", Blake2b, (lambda: Blake2b(20),)),
    RowCase("Sha3_256", Sha3_256),
    RowCase("Sha3_512", Sha3_512),
    RowCase("Keccak256", Keccak256),
    RowCase("Shake128", lambda: Shake128(32), (lambda: Shake128(64),)),
    RowCase("Shake256", lambda: Shake256(64), (lambda: Shake256(32),)),
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
    RowCase("HostSha256", HostSha256),
    RowCase("HostSha512", HostSha512),
    RowCase("HostSha384", HostSha384),
    RowCase("HostSha512_256", HostSha512_256),
    RowCase("HostSm3", HostSm3),
    RowCase("HostBlake2s", HostBlake2s, (lambda: HostBlake2s(20),)),
    RowCase("HostBlake2b", HostBlake2b, (lambda: HostBlake2b(20),)),
    RowCase("HostSha3_256", HostSha3_256),
    RowCase("HostSha3_512", HostSha3_512),
    RowCase("HostShake128", lambda: HostShake128(32), (lambda: HostShake128(64),)),
    RowCase("HostShake256", lambda: HostShake256(64), (lambda: HostShake256(32),)),
    RowCase("HostBlake3", HostBlake3, (lambda: HostBlake3(16),)),
    RowCase(
        "HostBlake3Keyed",
        lambda: HostBlake3Keyed(_KEY_A),
        (lambda: HostBlake3Keyed(_KEY_B), lambda: HostBlake3Keyed(_KEY_A, 16)),
    ),
    RowCase(
        "HostBlake3DeriveKey",
        lambda: HostBlake3DeriveKey("ctx a"),
        (
            lambda: HostBlake3DeriveKey("ctx b"),
            lambda: HostBlake3DeriveKey("ctx a", 16),
        ),
    ),
)

# Device rows: traceable, `digest` returns an `Array`, `fusion_path` derived per
# (row, backend). Host rows: never traceable, `digest` returns `np.ndarray`,
# `fusion_path` the one legitimate constant. Both read off the class, so a row
# cannot be filed under the wrong one.
DEVICE_ROWS: tuple[RowCase, ...] = tuple(
    c for c in ALL_ROWS if issubclass(c.make().__class__, DeviceRow)
)
HOST_ROWS: tuple[RowCase, ...] = tuple(
    c for c in ALL_ROWS if issubclass(c.make().__class__, HostRow)
)

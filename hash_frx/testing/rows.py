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

`variant` is a second instance of the same type with DIFFERENT parameters, or
`None` for a row that has none. It is what makes the equality cases mean
something: without it "two instances compare equal" is satisfied by a row whose
`__eq__` ignores the parameter that distinguishes them, which is exactly the bug
`_Blake3Hash` warns about in its own docstring.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from hash_frx.ascon.ascon import AsconHash256
from hash_frx.blake2b.blake2b import Blake2b
from hash_frx.blake2b.byte_hashes import HostBlake2b
from hash_frx.blake2s import Blake2s, HostBlake2s
from hash_frx.blake3.byte_hashes import (
    Blake3,
    Blake3DeriveKey,
    Blake3Keyed,
    HostBlake3,
    HostBlake3DeriveKey,
    HostBlake3Keyed,
)
from hash_frx.grostl.grostl import Grostl256
from hash_frx.keccak.byte_hashes import (
    HostSha3_256,
    HostSha3_512,
    HostShake128,
    HostShake256,
    Keccak256,
    Sha3_256,
    Sha3_512,
    Shake128,
    Shake256,
)
from hash_frx.ripemd160 import Ripemd160
from hash_frx.sha256 import HostSha256, Sha256
from hash_frx.sha512 import (
    HostSha384,
    HostSha512,
    HostSha512_256,
    Sha384,
    Sha512,
    Sha512_256,
)
from hash_frx.sm3 import HostSm3, Sm3

_KEY_A = bytes(range(32))
_KEY_B = bytes(range(1, 33))


class RowCase(NamedTuple):
    """One shipped row, plus how to build a differing sibling of it."""

    name: str
    make: Callable[[], Any]
    variant: Callable[[], Any] | None


def _case(name: str, make: Callable[[], Any], variant: Any = None) -> RowCase:
    return RowCase(name, make, variant)


# Device rows: traceable, `digest` returns an `Array`, `fusion_path` is derived
# per (row, backend) from the family's routing gate.
DEVICE_ROWS: tuple[RowCase, ...] = (
    _case("Sha256", Sha256),
    _case("Sha512", Sha512),
    _case("Sha384", Sha384),
    _case("Sha512_256", Sha512_256),
    _case("Sm3", Sm3),
    _case("Ripemd160", Ripemd160),
    _case("Grostl256", Grostl256),
    _case("AsconHash256", AsconHash256),
    _case("Blake2s", Blake2s, lambda: Blake2s(20)),
    _case("Blake2b", Blake2b, lambda: Blake2b(20)),
    _case("Sha3_256", Sha3_256),
    _case("Sha3_512", Sha3_512),
    _case("Keccak256", Keccak256),
    _case("Shake128", lambda: Shake128(32), lambda: Shake128(64)),
    _case("Shake256", lambda: Shake256(64), lambda: Shake256(32)),
    _case("Blake3", Blake3, lambda: Blake3(16)),
    _case("Blake3Keyed", lambda: Blake3Keyed(_KEY_A), lambda: Blake3Keyed(_KEY_B)),
    _case(
        "Blake3DeriveKey",
        lambda: Blake3DeriveKey("ctx a"),
        lambda: Blake3DeriveKey("ctx b"),
    ),
)

# Host rows: never traceable, `digest` returns `np.ndarray`, `fusion_path` is
# the one legitimate constant (`HOST` on every backend).
HOST_ROWS: tuple[RowCase, ...] = (
    _case("HostSha256", HostSha256),
    _case("HostSha512", HostSha512),
    _case("HostSha384", HostSha384),
    _case("HostSha512_256", HostSha512_256),
    _case("HostSm3", HostSm3),
    _case("HostBlake2s", HostBlake2s, lambda: HostBlake2s(20)),
    _case("HostBlake2b", HostBlake2b, lambda: HostBlake2b(20)),
    _case("HostSha3_256", HostSha3_256),
    _case("HostSha3_512", HostSha3_512),
    _case("HostShake128", lambda: HostShake128(32), lambda: HostShake128(64)),
    _case("HostShake256", lambda: HostShake256(64), lambda: HostShake256(32)),
    _case("HostBlake3", HostBlake3, lambda: HostBlake3(16)),
    _case(
        "HostBlake3Keyed",
        lambda: HostBlake3Keyed(_KEY_A),
        lambda: HostBlake3Keyed(_KEY_B),
    ),
    _case(
        "HostBlake3DeriveKey",
        lambda: HostBlake3DeriveKey("ctx a"),
        lambda: HostBlake3DeriveKey("ctx b"),
    ),
)

ALL_ROWS: tuple[RowCase, ...] = DEVICE_ROWS + HOST_ROWS

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A host Keccak-256, as the differential partner for the device one.

**Testonly, unlike its shipped siblings, and the asymmetry is deliberate.**
`HostSha3_256` and the `HostShake*` pair ship because `hashlib` implements those
functions in C: at `B=1` they cost ~1.7 us against ~941 us for a device
dispatch, so a strictly sequential caller genuinely wants them. Python's standard
library has no Keccak-256 — NIST standardised the changed padding, not the
original — so a shipped sibling would carry a pure-Python permutation at ~264 us,
two orders of magnitude off what a `Host*` name promises.

The consequence is worth stating rather than discovering: a `testonly` target
cannot be depended on by shipped code, so a sequential Keccak-256 caller has no
host path here. Giving it one is a dependency decision (`pycryptodome`,
`eth-hash`) rather than a reason to promote this.

Built on `reference.sponge`, which `reference_test` anchors against `hashlib` on
SHA3-256 and SHAKE. That anchor covers this too: Keccak-256 differs from SHA3-256
only in the domain byte, so an anchored SHA-3 proves the permutation, the absorb,
the multi-rate padding, and the squeeze — everything except the one constant,
which `keccak256_test` pins against the published vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hash_frx.keccak.byte_hashes import (
    KECCAK256_DIGEST_SIZE,
    KECCAK256_RATE,
    KECCAK256_SUFFIX,
    _HostKeccak,
)
from hash_frx.keccak.testing.reference import sponge

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash


class HostKeccak256(_HostKeccak):
    """`ByteHash` for host Keccak-256 over the plain-Python reference sponge.

    Inherits the batch loop and the value identity from `_HostKeccak`, which
    exists to be extended this way; only the per-message hash differs.
    """

    def __init__(self) -> None:
        super().__init__(KECCAK256_DIGEST_SIZE)

    def _hash_one(self, data: bytes) -> bytes:
        return sponge(data, KECCAK256_RATE, KECCAK256_SUFFIX, self.digest_size)


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_host_keccak256: type[ByteHash] = HostKeccak256

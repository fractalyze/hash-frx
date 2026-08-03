# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A host Keccak-256, as the differential partner for the device one.

**Testonly, unlike its FIPS 202 siblings, and the asymmetry is the point.**
`HostSha3_256` and the `HostShake*` pair ship because `hashlib` implements those
functions in C: at `B=1` they cost ~1.7 us against ~941 us for a device dispatch,
so a strictly sequential caller genuinely wants them. Python's standard library
has no Keccak-256 — NIST standardised the changed padding, not the original —
so a shipped sibling would have to carry a pure-Python permutation at ~264 us,
two orders of magnitude off what a `Host*` name promises. That is a test oracle
wearing a production name, so it stays in `testing/`.

Built on `reference.sponge`, which `reference_test` anchors against `hashlib` on
SHA3-256 and SHAKE. The anchor covers this one too: Keccak-256 differs from
SHA3-256 only in the domain byte, so an anchored SHA-3 proves the permutation,
the absorb, the multi-rate padding, and the squeeze — everything except the one
constant, which `keccak256_test` pins against the published Ethereum vectors.

If a consumer ever needs a fast host Keccak-256, that is a dependency decision
(`pycryptodome`, `eth-hash`) rather than a reason to promote this.
"""

from __future__ import annotations

import numpy as np
from frx.typing import ArrayLike

from hash_frx.keccak.fips202 import (
    KECCAK256_DIGEST_SIZE,
    KECCAK256_RATE,
    KECCAK256_SUFFIX,
)
from hash_frx.keccak.testing.reference import sponge


class HostKeccak256:
    """`ByteHash` for host Keccak-256 over the plain-Python reference sponge."""

    digest_size = KECCAK256_DIGEST_SIZE
    has_dedicated_fusion = False

    def digest(self, msg: ArrayLike) -> np.ndarray:
        rows = np.ascontiguousarray(np.asarray(msg, dtype=np.uint8))  # [B, L]
        out = np.empty((rows.shape[0], self.digest_size), dtype=np.uint8)
        for i, row in enumerate(rows):
            out[i] = np.frombuffer(
                sponge(row.tobytes(), KECCAK256_RATE, KECCAK256_SUFFIX, out.shape[1]),
                dtype=np.uint8,
            )
        return out

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HostKeccak256)

    def __hash__(self) -> int:
        return hash(HostKeccak256)

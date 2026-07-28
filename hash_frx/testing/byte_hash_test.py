# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ByteHash Protocol is structural and hash-agnostic.

Seam-level only: a duck-typed double stands in for a real hash, so this runs
before any concrete byte hash exists. The differential cases that pin a concrete
implementation against its standard live with that implementation.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from absl.testing import absltest
from frx.typing import ArrayLike

from hash_frx.byte_hash import ByteHash


class _HostSha256Double:
    """A minimal host `ByteHash`, standing in for a real implementation."""

    digest_size = 32
    has_dedicated_fusion = False

    def digest(self, msg: ArrayLike) -> np.ndarray:
        rows = np.ascontiguousarray(np.asarray(msg, dtype=np.uint8))
        out = np.empty((rows.shape[0], self.digest_size), dtype=np.uint8)
        for i, row in enumerate(rows):
            out[i] = np.frombuffer(hashlib.sha256(row.tobytes()).digest(), np.uint8)
        return out

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _HostSha256Double)

    def __hash__(self) -> int:
        return hash(_HostSha256Double)


class ByteHashProtocolTest(absltest.TestCase):
    def test_duck_typed_impl_satisfies_protocol(self) -> None:
        self.assertIsInstance(_HostSha256Double(), ByteHash)

    def test_consumer_reads_digest_size_without_naming_a_hash(self) -> None:
        h: ByteHash = _HostSha256Double()
        msgs = np.zeros((4, 13), dtype=np.uint8)  # L is static
        self.assertEqual(np.asarray(h.digest(msgs)).shape, (4, h.digest_size))

    def test_fusion_flag_is_the_substrate_axis(self) -> None:
        # The seam's whole point: substrate lives on the hash as a value, not in a
        # class name a consumer would have to branch on.
        self.assertIsInstance(_HostSha256Double().has_dedicated_fusion, bool)

    def test_value_identity_keeps_the_seam_re_trace_safe(self) -> None:
        # A param-free hash compares by type, so two freshly built instances are
        # one pytree aux value rather than two — identity equality here does not
        # error, it silently re-traces the enclosing jit zone per instance.
        self.assertEqual(_HostSha256Double(), _HostSha256Double())
        self.assertEqual(hash(_HostSha256Double()), hash(_HostSha256Double()))


if TYPE_CHECKING:
    # mypy-enforced seam conformance — the pin every instance module carries,
    # exercised here so the seam is known pinnable before one exists.
    _bh: type[ByteHash] = _HostSha256Double


if __name__ == "__main__":
    absltest.main()

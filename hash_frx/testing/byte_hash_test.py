# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ByteHash Protocol is structural and hash-agnostic.

Seam-level only: a duck-typed double stands in for a real hash, so this runs
before any concrete byte hash exists. The differential cases that pin a concrete
implementation against its standard live with that implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from absl.testing import absltest
from frx.typing import ArrayLike

from hash_frx.byte_hash import ByteHash, device_message, host_digest
from hash_frx.fusion import FusionPath


class _ByteHashDouble:
    """A shape-correct `ByteHash` standing in for a real hash.

    Returns zeros deliberately: this file tests the Protocol's *shape*, and the
    cases that pin a concrete hash against its standard live with that hash.
    Computing a real digest here would duplicate `HostSha256` and imply a
    fidelity nothing in this file checks.
    """

    digest_size = 32
    fusion_path = FusionPath.HOST

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return np.zeros((np.asarray(msg).shape[0], self.digest_size), dtype=np.uint8)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ByteHashDouble)

    def __hash__(self) -> int:
        return hash(_ByteHashDouble)


class ByteHashProtocolTest(absltest.TestCase):
    def test_duck_typed_impl_satisfies_protocol(self) -> None:
        self.assertIsInstance(_ByteHashDouble(), ByteHash)

    def test_consumer_reads_digest_size_without_naming_a_hash(self) -> None:
        h: ByteHash = _ByteHashDouble()
        msgs = np.zeros((4, 13), dtype=np.uint8)  # L is static
        self.assertEqual(np.asarray(h.digest(msgs)).shape, (4, h.digest_size))

    def test_fusion_flag_is_the_substrate_axis(self) -> None:
        # The seam's whole point: substrate lives on the hash as a value, not in a
        # class name a consumer would have to branch on.
        self.assertIsInstance(_ByteHashDouble().fusion_path, FusionPath)

    def test_value_identity_keeps_the_seam_re_trace_safe(self) -> None:
        # A param-free hash compares by type, so two freshly built instances are
        # one pytree aux value rather than two — identity equality here does not
        # error, it silently re-traces the enclosing jit zone per instance.
        self.assertEqual(_ByteHashDouble(), _ByteHashDouble())
        self.assertEqual(hash(_ByteHashDouble()), hash(_ByteHashDouble()))


if TYPE_CHECKING:
    # mypy-enforced seam conformance — the pin every instance module carries,
    # exercised here so the seam is known pinnable before one exists.
    _bh: type[ByteHash] = _ByteHashDouble


class DeviceMessageTest(absltest.TestCase):
    """The rank check every device row shares (#215).

    A 1-D message is the common miss — a single message is `B = 1`, not a bare
    `[L]` — and without the seam check it surfaced from inside a marked
    region's trace as a reshape or concatenate error naming neither the seam
    nor the rank.

    Only the rejection is asserted here, and that is the point: the check runs
    before the conversion, so it needs no backend, and this file must keep
    running with none — it is the seam's test, held to a double rather than to
    any implementation. What the accepting path returns is pinned by every
    family's own digest tests, which have a substrate to run on.
    """

    def test_rejects_a_message_that_is_not_a_batch(self) -> None:
        for bad in [np.zeros(8, dtype=np.uint8), np.zeros((1, 2, 3), dtype=np.uint8)]:
            with self.subTest(ndim=np.ndim(bad)):
                with self.assertRaisesRegex(ValueError, "2-D uint8"):
                    device_message(bad)
                # The host door has to answer the same call the same way (#235),
                # and this is the cheap backend-free target to say so in — the
                # all-rows suite covers it too, at `size = "large"`.
                with self.assertRaisesRegex(ValueError, "2-D uint8"):
                    host_digest(lambda b: b"", 1, bad)


if __name__ == "__main__":
    absltest.main()

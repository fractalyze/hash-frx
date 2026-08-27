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

from hash_frx.byte_hash import (
    ByteHash,
    device_message,
    message_length,
)
from hash_frx.fusion import FusionPath


class _ByteHashDouble:
    """A shape-correct `ByteHash` standing in for a real hash.

    Returns zeros deliberately: this file tests the Protocol's *shape*, and the
    cases that pin a concrete hash against its standard live with that hash.
    Computing a real digest here would duplicate a concrete row and imply a
    fidelity nothing in this file checks.

    It returns `np.ndarray` where a shipped row returns an `Array`: this target
    carries no device plugin, so reaching `frx.numpy` here fails the GPU leg
    (`docs/reference/conventions.md`). The Protocol's shape is what is under
    test, and that is substrate-free.
    """

    digest_size = 32
    fusion_path = FusionPath.GENERIC

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

    def test_fusion_path_is_a_value_on_the_hash(self) -> None:
        # The seam's whole point: routing lives on the hash as a value, not in a
        # class name a consumer would have to branch on, so `is_one_kernel` is
        # readable without knowing which hash this is.
        self.assertIsInstance(_ByteHashDouble().fusion_path, FusionPath)
        self.assertFalse(_ByteHashDouble().fusion_path.is_one_kernel)

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


class MessageLengthTest(absltest.TestCase):
    """Reading a batch's width through the seam.

    Backend-free assertions only, for the reason `DeviceMessageTest` states: the
    rank check runs before any conversion, so it belongs in the seam's own test.
    """

    def test_reports_the_batch_width(self) -> None:
        self.assertEqual(message_length(np.zeros((4, 37), dtype=np.uint8)), 37)

    def test_reads_the_length_through_the_seam_rank_check(self) -> None:
        # Reading `.shape[-1]` off a 1-D message would silently take the batch
        # for a length, so this door answers like every other seam door.
        with self.assertRaisesRegex(ValueError, "2-D uint8"):
            message_length(np.zeros(8, dtype=np.uint8))


if __name__ == "__main__":
    absltest.main()

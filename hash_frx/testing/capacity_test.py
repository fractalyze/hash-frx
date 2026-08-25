# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The capacity ladder — the buffer width a runtime-length marker hashes out of.

Parameterized on the block size rather than pinned to SHA-256's, because that is
the only per-family term in the policy and the six tail-operand rows do not
agree on it: 64 bytes for SHA-256, SM3, RIPEMD-160 and Grostl-256, 128 for
SHA-512 and BLAKE2b.

Separate from `byte_hash_test` deliberately: that target is pure numpy and takes
no GPU plugin wheels, and the device branch here needs a materialized array.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.byte_hash import at_capacity, capacity


class CapacityTest(parameterized.TestCase):
    """`capacity` — the width compilation is keyed on."""

    @parameterized.parameters(
        (64, 0, 64),
        (64, 1, 64),
        (64, 64, 64),
        (64, 65, 128),
        (64, 100, 128),
        (64, 1000, 1024),
        (128, 0, 128),
        (128, 100, 128),
        (128, 128, 128),
        (128, 129, 256),
        (128, 1000, 1024),
    )
    def test_a_host_message_widens_to_the_next_power_of_two(
        self, block_size: int, length: int, want: int
    ) -> None:
        # Floored at one block, so short messages share a width rather than each
        # compiling their own. The floor is the only thing the block size moves:
        # above it the ladder is the same rungs for every family.
        msg = np.zeros((1, length), np.uint8)
        self.assertEqual(capacity(msg, block_size), want)

    @parameterized.parameters(1, 65, 100, 1000)
    def test_a_device_message_keeps_its_own_extent(self, length: int) -> None:
        # Widening one would be a dispatched device op that buys the caller
        # nothing, so only the marker changes for a batch already materialized.
        # The block size does not enter: there is no host copy to round up.
        msg = fnp.asarray(np.zeros((1, length), dtype=np.uint8))
        self.assertEqual(capacity(msg, 64), length)
        self.assertEqual(capacity(msg, 128), length)

    def test_an_empty_device_message_still_clears_the_recognizer_floor(self) -> None:
        # `LMAX >= 1`: the emitter's clamped message-side index needs a byte to
        # land on even when no byte is live.
        msg = fnp.asarray(np.zeros((1, 0), dtype=np.uint8))
        self.assertEqual(capacity(msg, 64), 1)

    @parameterized.parameters(64, 128)
    def test_the_width_is_a_power_of_two_at_or_above_the_block(
        self, block_size: int
    ) -> None:
        # The property the ladder exists for, over a range no table enumerates:
        # every width is a power of two, never below the block, and never more
        # than one doubling past the message.
        for length in range(0, 4096, 13):
            width = capacity(np.zeros((1, length), np.uint8), block_size)
            self.assertEqual(width & (width - 1), 0)
            self.assertGreaterEqual(width, block_size)
            self.assertGreaterEqual(width, length)
            self.assertLess(width, max(2 * length, block_size + 1))


class AtCapacityTest(parameterized.TestCase):
    """`at_capacity` — the widening the policy above sizes."""

    @parameterized.parameters(0, 1, 63, 64, 65)
    def test_a_host_message_is_widened_with_zeros(self, length: int) -> None:
        # The bytes past `L` are never read — the marker takes the live count as
        # an operand — so they are left zero and nothing derives from them.
        msg = np.arange(length, dtype=np.uint8).reshape(1, length) + 1
        buf = at_capacity(msg, 128)
        self.assertEqual(buf.shape, (1, 128))
        np.testing.assert_array_equal(np.asarray(buf)[:, :length], msg)
        np.testing.assert_array_equal(
            np.asarray(buf)[:, length:], np.zeros((1, 128 - length), np.uint8)
        )

    def test_a_width_equal_to_the_length_copies_nothing(self) -> None:
        msg = np.arange(32, dtype=np.uint8).reshape(1, 32)
        np.testing.assert_array_equal(np.asarray(at_capacity(msg, 32)), msg)

    def test_a_device_message_at_its_own_width_is_handed_back_untouched(self) -> None:
        # The branch a device-resident caller takes on every call: a converting
        # call here would return the identical object and still pay a full eager
        # dispatch.
        msg = fnp.asarray(np.zeros((2, 48), dtype=np.uint8))
        self.assertIs(at_capacity(msg, 48), msg)

    def test_a_device_message_widens_in_graph(self) -> None:
        msg = fnp.asarray(np.arange(2 * 10, dtype=np.uint8).reshape(2, 10))
        buf = at_capacity(msg, 64)
        self.assertEqual(buf.shape, (2, 64))
        np.testing.assert_array_equal(np.asarray(buf)[:, :10], np.asarray(msg))
        self.assertEqual(int(np.asarray(buf)[:, 10:].sum()), 0)

    def test_a_width_below_the_message_is_rejected(self) -> None:
        # A capacity is a buffer the message must fit in; a smaller one would
        # truncate it into a digest of the wrong bytes rather than fail.
        with self.assertRaisesRegex(ValueError, "must be >= the message length"):
            at_capacity(np.zeros((1, 64), dtype=np.uint8), 32)

    def test_a_1_d_message_is_rejected_at_both_doors(self) -> None:
        # Both read the length through the seam's rank check, so a 1-D message
        # fails here rather than as a confusing unpack downstream (#235).
        with self.assertRaisesRegex(ValueError, "2-D uint8"):
            capacity(np.zeros(8, dtype=np.uint8), 64)
        with self.assertRaisesRegex(ValueError, "2-D uint8"):
            at_capacity(np.zeros(8, dtype=np.uint8), 64)


if __name__ == "__main__":
    absltest.main()

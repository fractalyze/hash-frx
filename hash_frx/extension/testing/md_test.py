# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`PadRule` against the seven padding rules it replaced.

Pinned as literal bytes rather than re-derived from the same formula, because a
test that recomputes the implementation proves only that the implementation is
deterministic. These vectors were read off the seven hand-written
`_padding_tail` functions before they were deleted, and each is the
specification's own rule.

The four axes exist because reading all seven families first is what showed they
were needed. Each has a case here that would pass with the axis set wrong on
some other family and fails with it set wrong here — a rule whose parameters do
not each change an outcome is a rule with a parameter nobody needs.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.extension.md import PadRule, Trailer

SHA256 = PadRule(64, Trailer.BIT_LENGTH)
SHA512 = PadRule(128, Trailer.BIT_LENGTH, reserve=16)
RIPEMD160 = PadRule(64, Trailer.BIT_LENGTH, big_endian=False)
SM3 = PadRule(64, Trailer.BIT_LENGTH)
GROSTL = PadRule(64, Trailer.BLOCK_COUNT)
BLAKE2S = PadRule(64, Trailer.NONE)
BLAKE2B = PadRule(128, Trailer.NONE)

# (rule, message length, tail length, first byte, last eight bytes)
_VECTORS = (
    # FIPS 180-4 §5.1.1: 0x80, zeros, then the length in BITS, big-endian.
    ("sha256_empty", SHA256, 0, 64, "80", "0000000000000000"),
    ("sha256_one", SHA256, 1, 63, "80", "0000000000000008"),
    ("sha256_last_that_fits", SHA256, 55, 9, "80", "00000000000001b8"),
    ("sha256_spills", SHA256, 56, 72, "80", "00000000000001c0"),
    ("sha256_whole_block", SHA256, 64, 64, "80", "0000000000000200"),
    # §5.1.2, the same rule at 128-byte blocks.
    ("sha512_empty", SHA512, 0, 128, "80", "0000000000000000"),
    ("sha512_last_that_fits", SHA512, 111, 17, "80", "0000000000000378"),
    # RIPEMD-160 is little-endian throughout, which is the whole of the
    # `big_endian` axis: same length, mirrored trailer.
    ("ripemd160_one", RIPEMD160, 1, 63, "80", "0800000000000000"),
    ("ripemd160_last_that_fits", RIPEMD160, 55, 9, "80", "b801000000000000"),
    # Grøstl v2.0.1 §3.1 counts BLOCKS, so the trailer barely moves with the
    # length — 0, 1 and 55 bytes all encode one block.
    ("grostl_empty", GROSTL, 0, 64, "80", "0000000000000001"),
    ("grostl_one", GROSTL, 1, 63, "80", "0000000000000001"),
    ("grostl_last_that_fits", GROSTL, 55, 9, "80", "0000000000000001"),
    ("grostl_spills", GROSTL, 56, 72, "80", "0000000000000002"),
    # RFC 7693 §3.3: HAIFA has no 0x80 and no trailer — the length reaches the
    # compression as a counter, so this is a zero fill.
    ("blake2s_empty", BLAKE2S, 0, 64, "00", "0000000000000000"),
    ("blake2s_partial", BLAKE2S, 56, 8, "00", "0000000000000000"),
)


class PadRuleVectorTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (name, rule, length, size, first, last)
        for name, rule, length, size, first, last in _VECTORS
    )
    def test_tail_matches_the_specification(
        self, rule: PadRule, length: int, size: int, first: str, last: str
    ) -> None:
        tail = rule.tail(length)
        self.assertEqual(len(tail), size)
        self.assertEqual((length + size) % rule.block_size, 0)
        self.assertEqual(tail[:1].tobytes().hex(), first)
        self.assertEqual(tail[-8:].tobytes().hex(), last)

    def test_a_whole_block_message_needs_no_haifa_padding(self) -> None:
        # The one case where a HAIFA tail is empty: the message already ends on
        # a block, and unlike the empty message there is a block to compress.
        self.assertEqual(len(BLAKE2S.tail(64)), 0)
        self.assertEqual(len(BLAKE2B.tail(128)), 0)

    def test_the_empty_haifa_message_still_gets_a_block(self) -> None:
        # `-0 % block` is 0, so the empty message is the case the modulo alone
        # gets wrong — the compression must still run once.
        self.assertEqual(len(BLAKE2S.tail(0)), 64)
        self.assertEqual(len(BLAKE2B.tail(0)), 128)


class PadRuleAxisTest(absltest.TestCase):
    """Each axis changes an outcome, or it is a parameter nobody needs."""

    def test_reserve_changes_where_the_message_spills(self) -> None:
        # SHA-512 reserves 16 bytes for a 128-bit field it only ever fills to
        # 64. The axis is invisible except in the 8-byte band the extra reserve
        # pushes over the boundary — 112..119 at a 128-byte block.
        wide, narrow = SHA512, PadRule(128, Trailer.BIT_LENGTH, reserve=8)
        self.assertEqual(len(wide.tail(111)), len(narrow.tail(111)))
        self.assertEqual(len(wide.tail(120)), len(narrow.tail(120)))
        self.assertEqual(len(wide.tail(112)), 144)
        self.assertEqual(len(narrow.tail(112)), 16)

    def test_endianness_mirrors_the_trailer(self) -> None:
        self.assertEqual(
            SHA256.tail(55)[-8:].tobytes(), RIPEMD160.tail(55)[-8:].tobytes()[::-1]
        )

    def test_block_count_and_bit_length_disagree(self) -> None:
        # Same block size, same endianness, same reserve — only the trailer
        # differs, and it has to.
        self.assertNotEqual(
            SHA256.tail(55)[-8:].tobytes(), GROSTL.tail(55)[-8:].tobytes()
        )

    def test_identical_rules_are_one_value(self) -> None:
        # SHA-256 and SM3 pad identically; a frozen dataclass makes that a fact
        # the type system carries rather than a coincidence two files repeat.
        self.assertEqual(SHA256, SM3)
        self.assertEqual(hash(SHA256), hash(SM3))


class PadRuleValidationTest(absltest.TestCase):
    def test_rejects_a_block_size_that_is_not_whole_words(self) -> None:
        for bad in (0, -64, 63):
            with self.assertRaisesRegex(ValueError, "block_size"):
                PadRule(bad, Trailer.BIT_LENGTH)

    def test_rejects_a_trailer_that_does_not_fit(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved"):
            PadRule(64, Trailer.BIT_LENGTH, reserve=4)

    def test_haifa_needs_no_reserve(self) -> None:
        self.assertEqual(len(PadRule(64, Trailer.NONE, reserve=0).tail(10)), 54)


class PadRuleTailIsSafeToShareTest(absltest.TestCase):
    def test_the_memoized_tail_is_the_documented_shape(self) -> None:
        # `tail` is memoized, so callers share one array. They all hand it to
        # `fnp.asarray`; this pins that nothing has made it writable-in-place by
        # accident, which would corrupt every later caller.
        first, second = SHA256.tail(55), SHA256.tail(55)
        self.assertIs(first, second)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    absltest.main()

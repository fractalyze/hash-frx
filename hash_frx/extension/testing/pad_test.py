# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`PadRule` and `SpongePad` against the nine padding rules they replaced.

Pinned as literal bytes rather than re-derived from the same formula, because a
test that recomputes the implementation proves only that the implementation is
deterministic. These vectors were read off the nine hand-written
`_padding_tail` functions before they were deleted, and each is the
specification's own rule.

The four axes exist because reading all seven families first is what showed they
were needed. Each has a case here that would pass with the axis set wrong on
some other family and fails with it set wrong here — a rule whose parameters do
not each change an outcome is a rule with a parameter nobody needs.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from hash_frx.extension.pad import PadRule, SpongePad, Trailer, haifa_counter

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
    def test_the_memoized_tail_is_read_only(self) -> None:
        # `tail` is memoized, so callers share one array — and the cache keys by
        # VALUE, so equal rules share an entry too. A writable array here means
        # one caller can change another family's padding.
        tail = SHA256.tail(55)
        self.assertIs(SHA256.tail(55), tail)
        self.assertFalse(tail.flags.writeable)
        with self.assertRaises(ValueError):
            tail[0] = 0

    def test_equal_rules_share_the_cached_tail(self) -> None:
        # SHA-256 and SM3 are the same rule by value, which is what makes the
        # read-only guarantee load-bearing rather than tidy.
        self.assertIs(SHA256.tail(55), SM3.tail(55))


class HaifaCounterTest(absltest.TestCase):
    """RFC 7693's §3.2/§3.3 split, which is where a wrong BLAKE2 digest hides."""

    def test_interior_blocks_count_a_full_block(self) -> None:
        # §3.2: an interior block reports the bytes through its own end.
        self.assertEqual(haifa_counter(0, 3, 200, 64), (64, False))
        self.assertEqual(haifa_counter(1, 3, 200, 64), (128, False))

    def test_the_final_block_reports_the_true_length(self) -> None:
        # §3.3, and the whole point: 200 bytes is 3 blocks of 64 with 8 bytes of
        # zero pad, and the pad must NOT be counted. Reporting 192 here — the
        # padded length — is a wrong digest for every message that is not a
        # block multiple, and right for every one that is.
        self.assertEqual(haifa_counter(2, 3, 200, 64), (200, True))

    def test_a_block_multiple_agrees_either_way(self) -> None:
        # Which is why the bug is easy to ship: at a block multiple the padded
        # and true lengths coincide, so the cases that would catch it are
        # exactly the ones a round-numbered test misses.
        self.assertEqual(haifa_counter(1, 2, 128, 64), (128, True))

    def test_the_empty_message_still_has_one_final_block(self) -> None:
        # HAIFA pads the empty message to a whole block, so there is a block to
        # compress and it reports zero bytes.
        self.assertEqual(haifa_counter(0, 1, 0, 64), (0, True))

    def test_blake2b_uses_the_wider_block(self) -> None:
        self.assertEqual(haifa_counter(0, 2, 200, 128), (128, False))
        self.assertEqual(haifa_counter(1, 2, 200, 128), (200, True))


# The two sponge rules `SpongePad` replaced. FIPS 202 section 6 fixes the Keccak
# suffixes; SP 800-232 Algorithm 2 fixes Ascon's.
SHA3_256_PAD = SpongePad(rate=136, head=0x06)
SHAKE256_PAD = SpongePad(rate=136, head=0x1F)
KECCAK256_PAD = SpongePad(rate=136, head=0x01)
ASCON_PAD = SpongePad(rate=8, head=0x01, final_bit=False)


class SpongePadVectorTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # FIPS 202 section 5.1: `suffix ‖ 0x00* ‖ 0x80`, never empty.
        ("sha3_256_empty", SHA3_256_PAD, 0, 136, 0x06, 0x80),
        ("sha3_256_one", SHA3_256_PAD, 1, 135, 0x06, 0x80),
        # One byte short of a block: the two ends land on the SAME byte and it
        # becomes `head | 0x80` — the standard's single-byte pad rather than a
        # special case, and the case a rule that wrote the two ends
        # independently would get wrong.
        ("sha3_256_single_byte_pad", SHA3_256_PAD, 135, 1, 0x86, 0x86),
        # A rate-aligned message gains a WHOLE padding block.
        ("sha3_256_whole_block", SHA3_256_PAD, 136, 136, 0x06, 0x80),
        ("shake256_one", SHAKE256_PAD, 1, 135, 0x1F, 0x80),
        ("shake256_single_byte_pad", SHAKE256_PAD, 135, 1, 0x9F, 0x9F),
        # Keccak-256 differs from SHA3-256 in exactly this byte, and in nothing
        # else — the padding NIST changed on standardisation.
        ("keccak256_one", KECCAK256_PAD, 1, 135, 0x01, 0x80),
        # Ascon has no trailing bit at all (`final_bit=False`), which is the one
        # axis separating the two sponge rules: the last byte stays zero, and at
        # a one-byte pad the head is NOT ORed with 0x80.
        ("ascon_empty", ASCON_PAD, 0, 8, 0x01, 0x00),
        ("ascon_one", ASCON_PAD, 1, 7, 0x01, 0x00),
        ("ascon_single_byte_pad", ASCON_PAD, 7, 1, 0x01, 0x01),
        ("ascon_whole_block", ASCON_PAD, 8, 8, 0x01, 0x00),
    )
    def test_tail_matches_the_specification(
        self,
        pad: SpongePad,
        length: int,
        size: int,
        first: int,
        last: int,
    ) -> None:
        tail = pad.tail(length)
        self.assertEqual(tail.shape, (size,))
        self.assertEqual(int(tail[0]), first)
        self.assertEqual(int(tail[-1]), last)
        # Everything between the two ends is zero.
        self.assertTrue(bool((tail[1:-1] == 0).all()))

    def test_the_pad_is_never_empty(self) -> None:
        # What makes the block count a function of the length alone, and what a
        # bare `-length % rate` gets wrong on an aligned message.
        for length in range(0, 3 * 136 + 1):
            self.assertGreater(SHA3_256_PAD.tail(length).size, 0)

    def test_the_padded_length_is_always_whole_blocks(self) -> None:
        for pad in (SHA3_256_PAD, ASCON_PAD):
            for length in range(0, 3 * pad.rate + 1):
                self.assertEqual((length + pad.tail(length).size) % pad.rate, 0)


class SpongePadAxisTest(absltest.TestCase):
    """Each axis changes an outcome, or it is a parameter nobody needs."""

    def test_the_head_is_the_domain_separation(self) -> None:
        # SHA3-256 and Keccak-256 differ in this byte and nothing else.
        self.assertNotEqual(int(SHA3_256_PAD.tail(1)[0]), int(KECCAK256_PAD.tail(1)[0]))
        self.assertEqual(SHA3_256_PAD.tail(1).size, KECCAK256_PAD.tail(1).size)

    def test_the_final_bit_changes_the_last_byte(self) -> None:
        with_bit = SpongePad(rate=8, head=0x01)
        without = SpongePad(rate=8, head=0x01, final_bit=False)
        self.assertEqual(int(with_bit.tail(1)[-1]), 0x80)
        self.assertEqual(int(without.tail(1)[-1]), 0x00)

    def test_the_final_bit_ors_into_a_single_byte_pad(self) -> None:
        # The case the two axes interact on: one byte of pad, so head and
        # trailing bit share it.
        self.assertEqual(int(SpongePad(rate=8, head=0x06).tail(7)[0]), 0x86)
        self.assertEqual(
            int(SpongePad(rate=8, head=0x06, final_bit=False).tail(7)[0]), 0x06
        )

    def test_the_rate_sets_the_boundary(self) -> None:
        self.assertEqual(SpongePad(rate=136, head=0x06).tail(1).size, 135)
        self.assertEqual(SpongePad(rate=72, head=0x06).tail(1).size, 71)


class SpongePadValidationTest(absltest.TestCase):
    def test_rejects_a_head_that_claims_the_padding_bit(self) -> None:
        # Bit 7 belongs to `pad10*1`'s trailing 1. A head that set it would
        # collide on a one-byte pad, where `|= 0x80` on a host tail and
        # `^ 0x80` on a traced block disagree — a different digest, not an
        # error.
        with self.assertRaisesRegex(ValueError, "head"):
            SpongePad(rate=136, head=0x80)

    def test_a_rule_without_the_final_bit_may_use_the_whole_byte(self) -> None:
        # The bit is only reserved where there is a trailing 1 to reserve it
        # for, so the rejection above must not be unconditional.
        self.assertEqual(
            int(SpongePad(rate=8, head=0x80, final_bit=False).tail(1)[0]), 0x80
        )

    def test_rejects_a_non_positive_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "rate"):
            SpongePad(rate=0, head=0x06)

    def test_rejects_a_head_that_is_not_a_byte(self) -> None:
        with self.assertRaisesRegex(ValueError, "head"):
            SpongePad(rate=8, head=0x100, final_bit=False)


class SpongePadTailIsSafeToShareTest(absltest.TestCase):
    def test_the_memoized_tail_is_read_only(self) -> None:
        # Keyed by VALUE, so two equal rules share one entry — a caller writing
        # through the array would change the other's padding, silently and
        # everywhere after. `PadRuleTailIsSafeToShareTest` states the same for
        # the MD rule.
        tail = SHA3_256_PAD.tail(1)
        with self.assertRaises(ValueError):
            tail[0] = 0

    def test_equal_rules_share_the_cached_tail(self) -> None:
        # SHAKE256 and SHA3-256 run the same rate but different heads, so they
        # must NOT share; two spellings of one rule must.
        self.assertIsNot(SHA3_256_PAD.tail(1), SHAKE256_PAD.tail(1))
        self.assertIs(SHA3_256_PAD.tail(1), SpongePad(rate=136, head=0x06).tail(1))


# `nblocks` is what a RUNTIME-LENGTH path sizes its block loop from, so it has to
# hold for every rule rather than for the trailer-carrying ones alone. The
# vectors below are literal, per this file's rule that a test recomputing the
# implementation proves only that the implementation is deterministic; the
# invariant test that follows then ties them to `tail`, which the vectors at the
# top of this file already pin.


class NblocksTest(parameterized.TestCase):
    """Block counts, literal per this file's rule, for the two shapes that do not
    use the trailer formula."""

    @parameterized.named_parameters(
        # `pad10*1` never pads to nothing, so a rate-aligned message gains a
        # whole block and the count is `length // rate + 1` everywhere.
        ("ascon_empty", ASCON_PAD, 0, 1),
        ("ascon_partial", ASCON_PAD, 7, 1),
        ("ascon_exact_block", ASCON_PAD, 8, 2),
        ("ascon_two_blocks", ASCON_PAD, 63, 8),
        ("sha3_empty", SHA3_256_PAD, 0, 1),
        ("sha3_one_short", SHA3_256_PAD, 135, 1),
        ("sha3_exact_block", SHA3_256_PAD, 136, 2),
        # HAIFA zero-fills and carries the length in a counter, so a
        # block-aligned message needs NO padding — the one family whose block
        # count is not the trailer formula. The empty message still runs the
        # compression once.
        ("blake2s_empty", BLAKE2S, 0, 1),
        ("blake2s_partial", BLAKE2S, 1, 1),
        ("blake2s_exact_block", BLAKE2S, 64, 1),
        ("blake2s_spills", BLAKE2S, 65, 2),
        ("blake2s_two_exact", BLAKE2S, 128, 2),
        ("blake2b_exact_block", BLAKE2B, 128, 1),
        ("blake2b_spills", BLAKE2B, 129, 2),
    )
    def test_nblocks(
        self, rule: PadRule | SpongePad, length: int, expected: int
    ) -> None:
        self.assertEqual(rule.nblocks(length), expected)


class NblocksAgreesWithTailTest(parameterized.TestCase):
    """`nblocks` and `tail` are two views of one rule, and this is the contract
    between them: the padded message is exactly `nblocks` whole blocks.

    Restating either one per call site is what let a family's batch digest and
    its streaming finalize disagree about where the length field goes, so the
    agreement is pinned here rather than assumed.
    """

    @parameterized.named_parameters(
        ("sha256", SHA256, 64),
        ("sha512", SHA512, 128),
        ("ripemd160", RIPEMD160, 64),
        ("sm3", SM3, 64),
        ("grostl", GROSTL, 64),
        ("blake2s", BLAKE2S, 64),
        ("blake2b", BLAKE2B, 128),
        ("ascon", ASCON_PAD, 8),
        ("sha3_256", SHA3_256_PAD, 136),
    )
    def test_padded_length_is_whole_blocks(
        self, rule: PadRule | SpongePad, block_size: int
    ) -> None:
        for length in range(0, 300):
            padded = length + len(rule.tail(length))
            self.assertEqual(
                padded,
                rule.nblocks(length) * block_size,
                f"{rule} disagrees with its own tail at length {length}",
            )


if __name__ == "__main__":
    absltest.main()

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""RFC 7693 §2.8's parameter block — the byte layout, and the constant it
generalizes.

The block is where every BLAKE2 parameter that is not the message enters the
hash, so a wrong offset here is a wrong digest with no other symptom: the hash
stays self-consistent, every structural test passes, and only a published
vector disagrees. Two things are held here that a vector cannot see.

**The generalization is pinned against what it replaced.** Both families
carried `0x01010000 ^ digest_size` as the whole parameter block while they
shipped unkeyed. `test_unkeyed_reproduces_the_original_constant` asserts the
builder still produces exactly that at every digest size in both families —
which is the check that catches a transposed offset in the half of the block
an unkeyed hash leaves zero, where a keyed vector would not look.

**Sequential mode is asserted, not assumed.** Fanout and depth are 1 and the
whole tree group is 0; those are the values that make this a sequential hash
rather than a tree one, and they are written as constants, so a test is the
only thing holding them.

Host-only and substrate-free — the module under test pulls no frx, which is
half of why it can be read without a device (`extension/pad.py` keeps the same
property for the same reason).
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from hash_frx import blake2_params
from hash_frx.blake2_params import (
    BLAKE2B_WORD_BYTES,
    BLAKE2S_WORD_BYTES,
    block_size,
    max_field_size,
    param_block,
    param_words,
    salt_size,
)

_FAMILIES = (("blake2b", BLAKE2B_WORD_BYTES), ("blake2s", BLAKE2S_WORD_BYTES))


class WidthTest(parameterized.TestCase):
    @parameterized.named_parameters(*_FAMILIES)
    def test_the_widths_are_the_standards(self, word_bytes: int) -> None:
        # §2.1/§2.8's numbers, spelled rather than derived, so a change to the
        # derivation has to come past them: BLAKE2b is a 64-byte block with a
        # 16-byte salt and a 64-byte maximum digest; BLAKE2s is half of each.
        expected = {
            BLAKE2B_WORD_BYTES: (64, 16, 64),
            BLAKE2S_WORD_BYTES: (32, 8, 32),
        }[word_bytes]
        self.assertEqual(
            (
                block_size(word_bytes),
                salt_size(word_bytes),
                max_field_size(word_bytes),
            ),
            expected,
        )

    @parameterized.named_parameters(*_FAMILIES)
    def test_the_block_is_eight_words(self, word_bytes: int) -> None:
        # The block is XORed word-for-word into an eight-word state (§2.5), so
        # any other length is a state the XOR cannot reach.
        self.assertLen(param_block(word_bytes, 1), 8 * word_bytes)
        self.assertLen(param_words(word_bytes, 1), 8)


class UnkeyedConstantTest(parameterized.TestCase):
    """The pin against what the builder replaced."""

    @parameterized.named_parameters(*_FAMILIES)
    def test_unkeyed_reproduces_the_original_constant(self, word_bytes: int) -> None:
        # `0x01010000 ^ digest_size` into h[0] and nothing anywhere else was
        # the entire parameter block while both families shipped unkeyed. At
        # EVERY digest size, because the length is the one field that varied.
        for size in range(1, max_field_size(word_bytes) + 1):
            words = param_words(word_bytes, size)
            self.assertEqual(
                words[0],
                0x01010000 ^ size,
                f"h[0] moved at digest_size={size}",
            )
            self.assertEqual(
                words[1:],
                (0,) * 7,
                f"an unkeyed hash gained a nonzero word at digest_size={size}",
            )


class LayoutTest(parameterized.TestCase):
    """Where each field lands, read off the bytes rather than off the words —
    §2.8 defines the block as a byte layout and the word split is downstream."""

    @parameterized.named_parameters(*_FAMILIES)
    def test_the_named_bytes_are_where_2_8_puts_them(self, word_bytes: int) -> None:
        block = param_block(word_bytes, 48 % max_field_size(word_bytes) + 1, 7)
        self.assertEqual(block[0], 48 % max_field_size(word_bytes) + 1)  # nn
        self.assertEqual(block[1], 7)  # kk
        self.assertEqual(block[2], 1)  # fanout, sequential
        self.assertEqual(block[3], 1)  # depth, sequential

    @parameterized.named_parameters(*_FAMILIES)
    def test_the_tree_group_is_sequential_mode(self, word_bytes: int) -> None:
        # Leaf length, node offset, node depth and inner length — every byte
        # from 4 up to the salt — are 0 for a sequential hash. This is the
        # assertion that stands in for the tree mode this package does not
        # implement: the fields exist at the right offsets and are zero.
        block = param_block(word_bytes, 32, 16, b"S" * salt_size(word_bytes))
        salt_off = block_size(word_bytes) // 2
        self.assertEqual(block[4:salt_off], bytes(salt_off - 4))

    @parameterized.named_parameters(*_FAMILIES)
    def test_salt_and_person_land_in_their_own_halves(self, word_bytes: int) -> None:
        # The two fields are adjacent and the same width, which is exactly the
        # arrangement a transposition survives silently — so they are given
        # distinguishable values and read back by offset.
        n = salt_size(word_bytes)
        block = param_block(word_bytes, 32, 0, b"\xa1" * n, b"\xb2" * n)
        salt_off = block_size(word_bytes) // 2
        self.assertEqual(block[salt_off : salt_off + n], b"\xa1" * n)
        self.assertEqual(block[salt_off + n :], b"\xb2" * n)

    @parameterized.named_parameters(*_FAMILIES)
    def test_a_short_salt_is_zero_padded_not_left_aligned_elsewhere(
        self, word_bytes: int
    ) -> None:
        # RFC 7693 fixes the field width, so a 1-byte salt occupies byte 0 of
        # the field and zeros after it — the behavior `hashlib` also has, and
        # the reason a caller cannot use salt length as a parameter.
        n = salt_size(word_bytes)
        block = param_block(word_bytes, 32, 0, b"\xa1")
        salt_off = block_size(word_bytes) // 2
        self.assertEqual(block[salt_off : salt_off + n], b"\xa1" + bytes(n - 1))


class RejectionTest(parameterized.TestCase):
    """What the builder refuses. Each is a caller error the standard has no
    representation for, so refusing beats truncating or wrapping."""

    @parameterized.named_parameters(*_FAMILIES)
    def test_an_oversized_salt_or_person_is_rejected(self, word_bytes: int) -> None:
        n = salt_size(word_bytes)
        with self.assertRaisesRegex(ValueError, "salt"):
            param_block(word_bytes, 32, 0, b"x" * (n + 1))
        with self.assertRaisesRegex(ValueError, "person"):
            param_block(word_bytes, 32, 0, b"", b"x" * (n + 1))

    @parameterized.named_parameters(*_FAMILIES)
    def test_an_out_of_range_digest_or_key_is_rejected(self, word_bytes: int) -> None:
        cap = max_field_size(word_bytes)
        with self.assertRaisesRegex(ValueError, "digest_size"):
            param_block(word_bytes, 0)
        with self.assertRaisesRegex(ValueError, "digest_size"):
            param_block(word_bytes, cap + 1)
        with self.assertRaisesRegex(ValueError, "key_size"):
            param_block(word_bytes, 32, cap + 1)
        with self.assertRaisesRegex(ValueError, "key_size"):
            param_block(word_bytes, 32, -1)


class SubstrateTest(absltest.TestCase):
    def test_the_module_pulls_no_frx(self) -> None:
        # The property that lets a parameter block be read and tested without
        # a device — `extension/pad.py`'s, for the same reason. A future
        # `fnp` import here would initialize a backend on a host-only path.
        self.assertNotIn("frx", blake2_params.__dict__)


if __name__ == "__main__":
    absltest.main()

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SP 800-185 §2.3's encodings, held to the standard on their own.

These four functions are the correctness surface of cSHAKE, KMAC and TupleHash,
and a bug in any of them reaches all three at once. So they are pinned here,
with no construction above them: a failure in this file names the encoding, and
a failure in a construction's vectors with this file green is the construction's
own.

Three things are asserted that a construction's published vector would also
catch, but only after the fact and without saying which mistake happened:

**The units.** `encode_string` counts bits and `bytepad`'s `w` counts bytes. The
two are adjacent, both plausible for either function, and crossing them changes
no shape and no block count — `test_encode_string_counts_bits_not_bytes` and
`test_bytepad_w_counts_bytes_not_bits` are the checks that make the difference
visible before a digest is involved.

**Parsing, rather than bytes.** The functions exist so that a reader can find
the boundary between a length and what it measures. `_parse_left` and
`_parse_right` are readers written from §2.3.1's description rather than from
the encoder, so the round-trip asserts the property the standard is *for*, not
that the encoder agrees with itself.

**The 2^2040 bound is structural.** It is where the one-byte count `n` runs
out, so both sides of the boundary are pinned rather than the limit being taken
on trust.

Host-only and substrate-free — the module under test pulls no frx, so this
suite takes neither the frx requirement nor the GPU plugin, which is itself part
of what it asserts (`blake2_params_test.py` keeps the same property).
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from hash_frx.keccak.encodings import (
    MAX_ENCODE_BITS,
    MAX_ENCODE_EXCLUSIVE,
    bytepad,
    encode_string,
    left_encode,
    right_encode,
)

# The lengths at which §2.3.1's `n` steps up, and the last value before each.
# `n` is the count of base-256 digits, so it grows exactly at a power of 256.
_BOUNDARIES = (0, 1, 255, 256, 257, 65535, 65536, (1 << 24) - 1, 1 << 24)


def _parse_left(data: bytes) -> tuple[int, bytes]:
    """Read a `left_encode` off the front, returning `(x, the rest)`.

    Written from §2.3.1's description of what the encoding is *for* — the
    leading byte says how many follow — rather than from `left_encode`. A reader
    derived from the encoder would agree with a wrong encoder.
    """
    n = data[0]
    return int.from_bytes(data[1 : 1 + n], "big"), data[1 + n :]


def _parse_right(data: bytes) -> tuple[int, bytes]:
    """Read a `right_encode` off the end, returning `(x, what preceded it)`."""
    n = data[-1]
    return int.from_bytes(data[-1 - n : -1], "big"), data[: -1 - n]


class SpecExampleTest(absltest.TestCase):
    """The three worked examples §2.3 states in prose.

    The document writes them as bit strings in FIPS 202's LSB-first notation, so
    `10000000` is the byte `0x01` and `00000000` is `0x00` — the transcription
    that is easiest to get backwards, and the reason these are spelled as bytes
    here.
    """

    def test_left_encode_zero(self) -> None:
        # §2.3.1: "left_encode(0) will yield 10000000 00000000" — the count 1,
        # then the single zero byte it counts.
        self.assertEqual(left_encode(0), b"\x01\x00")

    def test_right_encode_zero(self) -> None:
        # §2.3.1: "right_encode(0) will yield 00000000 10000000" — the same two
        # bytes as left_encode(0), in the other order.
        self.assertEqual(right_encode(0), b"\x00\x01")

    def test_encode_string_empty(self) -> None:
        # §2.3.2: encode_string("") yields "10000000 00000000" — left_encode(0)
        # with nothing after it.
        self.assertEqual(encode_string(b""), b"\x01\x00")


class UnitTest(parameterized.TestCase):
    """Bits against bytes — the one confusion this module exists to prevent."""

    def test_encode_string_counts_bits_not_bytes(self) -> None:
        # §2.3.2 defines encode_string over bit strings, so a four-byte string
        # is length 32. A byte count would give `01 04 ...`: same length, same
        # shape, different hash. "KMAC" is the literal §4.1 function name, so
        # this is the exact prefix KMAC will absorb.
        self.assertEqual(encode_string(b"KMAC"), b"\x01\x20KMAC")
        self.assertNotEqual(encode_string(b"KMAC"), b"\x01\x04KMAC")

    @parameterized.parameters(0, 1, 2, 7, 8, 31, 32, 136, 168, 1000)
    def test_encode_string_prefix_is_the_bit_length(self, size: int) -> None:
        # Across every length, the parsed prefix is 8 * len(s) and what follows
        # is the string unchanged.
        s = bytes(i % 256 for i in range(size))
        bit_length, rest = _parse_left(encode_string(s))
        self.assertEqual(bit_length, 8 * size)
        self.assertEqual(rest, s)

    def test_bytepad_w_counts_bytes_not_bits(self) -> None:
        # §2.3.3 divides the bit length by 8 before taking the modulus, so `w`
        # is a byte count. At SHAKE128's rate the result is 168 bytes, not 21
        # (168 bits) and not 1344 (168 * 8).
        self.assertLen(bytepad(b"", 168), 168)


class LeftRightTest(parameterized.TestCase):
    """`left_encode` and `right_encode` against §2.3.1."""

    @parameterized.parameters(*_BOUNDARIES)
    def test_left_encode_parses_from_the_front(self, x: int) -> None:
        # The property §2.3.1 claims: a reader that takes the leading count
        # recovers x and lands exactly at the end.
        value, rest = _parse_left(left_encode(x))
        self.assertEqual(value, x)
        self.assertEmpty(rest)

    @parameterized.parameters(*_BOUNDARIES)
    def test_right_encode_parses_from_the_end(self, x: int) -> None:
        value, rest = _parse_right(right_encode(x))
        self.assertEqual(value, x)
        self.assertEmpty(rest)

    def test_both_parse_back_over_a_dense_range(self) -> None:
        # Dense rather than sampled across the first two width classes, because
        # an off-by-one in `n` shows up at exactly one value (255 -> 256) and a
        # sampled sweep can step over it.
        for x in range(0, 1200):
            self.assertEqual(_parse_left(left_encode(x))[0], x)
            self.assertEqual(_parse_right(right_encode(x))[0], x)

    @parameterized.named_parameters(
        ("zero", 0, 1),
        ("one", 1, 1),
        ("max_one_byte", 255, 1),
        ("min_two_bytes", 256, 2),
        ("max_two_bytes", 65535, 2),
        ("min_three_bytes", 65536, 3),
    )
    def test_the_width_steps_at_each_power_of_256(self, x: int, width: int) -> None:
        # §2.3.1: n is the smallest positive integer with 2^(8n) > x, so it
        # grows at 256, 65536, ... and never at a round decimal.
        self.assertLen(left_encode(x), width + 1)
        self.assertLen(right_encode(x), width + 1)
        self.assertEqual(left_encode(x)[0], width)
        self.assertEqual(right_encode(x)[-1], width)

    @parameterized.parameters(*_BOUNDARIES)
    def test_the_two_share_a_body_and_differ_in_where_the_count_sits(
        self, x: int
    ) -> None:
        # The same base-256 digits either way — the encodings differ in the
        # position of the count and in nothing else, which is what makes them
        # one function with two parse directions rather than two encodings.
        left, right = left_encode(x), right_encode(x)
        self.assertEqual(left[1:], right[:-1])
        self.assertEqual(left[:1], right[-1:])

    def test_the_body_is_big_endian(self) -> None:
        # x = Σ 2^(8(n-i)) * x_i for i = 1..n (§2.3.1) puts the most significant
        # digit first. Little-endian would give `02 02 01` here.
        self.assertEqual(left_encode(0x0102), b"\x02\x01\x02")


class BoundTest(absltest.TestCase):
    """The 2^2040 validity condition, and why it is that number."""

    def test_the_largest_encodable_integer_fills_the_count_byte(self) -> None:
        # 2^2040 - 1 is 255 bytes of 0xFF, and 255 is the largest count a single
        # byte holds. That is the whole derivation of the limit.
        x = MAX_ENCODE_EXCLUSIVE - 1
        encoded = left_encode(x)
        self.assertEqual(encoded[0], 255)
        self.assertLen(encoded, 256)
        self.assertEqual(_parse_left(encoded)[0], x)

    def test_one_past_the_limit_is_rejected(self) -> None:
        # The next integer needs a 256-byte body, which the count cannot name.
        with self.assertRaisesRegex(ValueError, "2\\*\\*2040"):
            left_encode(MAX_ENCODE_EXCLUSIVE)
        with self.assertRaisesRegex(ValueError, "2\\*\\*2040"):
            right_encode(MAX_ENCODE_EXCLUSIVE)

    def test_the_limit_is_where_the_count_byte_runs_out(self) -> None:
        # Stated as an identity rather than a literal, so the constant and its
        # reason cannot drift apart.
        self.assertEqual(MAX_ENCODE_BITS, 8 * 255)

    def test_a_negative_integer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 <= x"):
            left_encode(-1)
        with self.assertRaisesRegex(ValueError, "0 <= x"):
            right_encode(-1)


class BytepadTest(parameterized.TestCase):
    """§2.3.3, at the rates the constructions above actually use."""

    @parameterized.named_parameters(
        ("shake128_rate", 168), ("shake256_rate", 136), ("small", 8), ("one", 1)
    )
    def test_the_result_is_a_whole_number_of_w_byte_blocks(self, w: int) -> None:
        # The point of the function: what follows the padded prefix starts on a
        # block boundary of the sponge it is absorbed into.
        for size in range(0, 3 * w + 1):
            self.assertEqual(len(bytepad(bytes(size), w)) % w, 0)

    @parameterized.named_parameters(("shake128_rate", 168), ("shake256_rate", 136))
    def test_it_opens_with_left_encode_of_w(self, w: int) -> None:
        # §2.3.3 prepends the encoded block size. The prefix names it, which is
        # what lets a reader re-derive where the padding ends.
        padded = bytepad(b"payload", w)
        self.assertEqual(padded[: len(left_encode(w))], left_encode(w))
        parsed, rest = _parse_left(padded)
        self.assertEqual(parsed, w)
        self.assertEqual(rest[: len(b"payload")], b"payload")

    def test_the_fill_is_zeros(self) -> None:
        # §2.3.3 fills with 00000000, so everything past the prefix and X is
        # zero.
        w = 136
        padded = bytepad(b"abc", w)
        head = left_encode(w) + b"abc"
        self.assertEqual(padded[: len(head)], head)
        self.assertEqual(padded[len(head) :], bytes(w - len(head)))

    def test_an_exact_fit_gains_no_extra_block(self) -> None:
        # The fill is `-len(z) % w`, which is 0 when z already fills whole
        # blocks. A fill written as `w - len(z) % w` would add a block here, and
        # at every other exact fit.
        w = 136
        exact = bytes(w - len(left_encode(w)))
        self.assertLen(bytepad(exact, w), w)
        self.assertLen(bytepad(exact + bytes(w), w), 2 * w)

    def test_a_longer_x_spans_the_blocks_it_needs(self) -> None:
        w = 136
        self.assertLen(bytepad(bytes(w), w), 2 * w)

    @parameterized.parameters(0, -1)
    def test_a_non_positive_w_is_rejected(self, w: int) -> None:
        # §2.3.3's validity condition is w > 0; at w = 0 the modulus is a
        # division by zero rather than a padding rule.
        with self.assertRaisesRegex(ValueError, "must be positive"):
            bytepad(b"", w)


class PublishedIntermediateTest(absltest.TestCase):
    """NIST's KMAC sample values, which publish the encodings themselves.

    The samples do not only give a digest — they print `Encoded K`, `Encoded N`,
    `Encoded S`, `Right_encoded L` and the `bytepad` results as bytes, at both
    rates. That makes them a published vector for *this layer*, so the encodings
    are anchored against the standard's own numbers before any sponge is
    involved, and a later construction failure is definitely not from here.

    Both samples key on the same 32-byte K and the same function name, and
    differ in rate (168 against 136), requested output (256 bits against 512)
    and customization (empty against a string) — so between them they move every
    argument these four functions take.
    """

    # KMAC_samples.pdf, "Length of Key is 256-bits": 40 41 ... 5F.
    _KEY = bytes(range(0x40, 0x60))
    # The §4.1 function name, and the customization of the 256-bit sample.
    _NAME = b"KMAC"
    _CUSTOM = b"My Tagged Application"

    def _padded(self, prefix: bytes, w: int) -> bytes:
        """A published `bytepad` result, spelled as its prefix and its width so
        the 168 or 136 zero bytes need not be transcribed."""
        return prefix + bytes(-len(prefix) % w)

    def test_the_key_encoding_is_a_bit_count(self) -> None:
        # `Encoded K` is printed as `02 01 00 40 41 ...` — left_encode(256) for
        # a 32-byte key. This single line is what settles bits against bytes:
        # a byte count would print `01 20`, which is the same length and a
        # different value.
        self.assertEqual(encode_string(self._KEY), bytes.fromhex("020100") + self._KEY)

    def test_the_key_block_at_both_rates(self) -> None:
        # `byte_padded stuff`, whose leading bytes are left_encode(w): `01 A8`
        # at KMAC128's 168 and `01 88` at KMAC256's 136. Here 168 is a BYTE
        # count — the same `01 A8` that appears below as a BIT count.
        encoded = bytes.fromhex("020100") + self._KEY
        self.assertEqual(
            bytepad(encode_string(self._KEY), 168),
            self._padded(bytes.fromhex("01A8") + encoded, 168),
        )
        self.assertEqual(
            bytepad(encode_string(self._KEY), 136),
            self._padded(bytes.fromhex("0188") + encoded, 136),
        )

    def test_the_output_length_encoding(self) -> None:
        # `Right_encoded L` for the two requested output lengths, both in bits:
        # 256 -> `01 00 02`, 512 -> `02 00 02`. The trailing byte is the count
        # and the leading pair is the value, which is the whole difference
        # between this and left_encode.
        self.assertEqual(right_encode(256), bytes.fromhex("010002"))
        self.assertEqual(right_encode(512), bytes.fromhex("020002"))

    def test_the_function_name_and_customization_encodings(self) -> None:
        # `Encoded N` and `Encoded S`. The customization is 21 bytes, so its
        # prefix is left_encode(168) — printed as `01 A8`, the same two bytes
        # the 168-byte rate produces above. Same bytes, different unit, in one
        # published sample: the clearest statement of the trap this module
        # documents.
        self.assertEqual(encode_string(self._NAME), bytes.fromhex("01204B4D4143"))
        self.assertEqual(encode_string(b""), bytes.fromhex("0100"))
        self.assertEqual(
            encode_string(self._CUSTOM), bytes.fromhex("01A8") + self._CUSTOM
        )
        self.assertLen(self._CUSTOM, 21)

    def test_the_name_and_customization_block_at_both_rates(self) -> None:
        # `bytepad data` — what cSHAKE absorbs ahead of the message. The empty
        # customization still contributes its `01 00`, which is why cSHAKE with
        # an empty N and S is a decision in the construction rather than
        # something the encodings produce for free.
        self.assertEqual(
            bytepad(encode_string(self._NAME) + encode_string(b""), 168),
            self._padded(bytes.fromhex("01A801204B4D41430100"), 168),
        )
        self.assertEqual(
            bytepad(encode_string(self._NAME) + encode_string(self._CUSTOM), 136),
            self._padded(bytes.fromhex("018801204B4D414301A8") + self._CUSTOM, 136),
        )


if __name__ == "__main__":
    absltest.main()

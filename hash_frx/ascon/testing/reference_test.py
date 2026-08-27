# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The oracles are anchored to the published record before anything is held to them.

`ascon_test` checks the frx Ascon-Hash256 against `reference.py`, which only
means something if the oracle is right. Oracle and KAT transcription were both
written here, so the anchors are drawn from the outside: the KAT vectors
(three independent sources, provenance on `KAT_VECTORS`), and — for the
initialization, which the oracle derives by permuting IV ‖ 0^256 rather than
copying — the precomputed state NIST SP 800-232 publishes in Table 12. A
wrong constant, S-box entry, rotation pair, byte order or padding rule does
not survive both: every one of them feeds all twelve rounds of every
permutation call.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from hash_frx.ascon.testing.reference import (
    CXOF_INITIAL_STATE,
    CXOF_IV,
    CXOF_KAT_VECTORS,
    INITIAL_STATE,
    IV,
    KAT_VECTORS,
    MAX_CUSTOMIZATION_BYTES,
    RATE,
    SBOX,
    XOF_INITIAL_STATE,
    XOF_IV,
    XOF_KAT_VECTORS,
    ascon_cxof128,
    ascon_hash256,
    ascon_xof128,
    customization_prefix,
    pad,
    permutation,
)


class ReferenceAnchorTest(parameterized.TestCase):
    @parameterized.parameters(*((len(m), m, d) for m, d in KAT_VECTORS))
    def test_matches_the_sp800_232_kat(
        self, _length: int, msg: bytes, digest_hex: str
    ) -> None:
        self.assertEqual(ascon_hash256(msg).hex(), digest_hex)

    def test_initial_state_matches_the_published_precomputation(self) -> None:
        # The oracle derives Ascon-p[12](IV ‖ 0^256); SP 800-232 §A.3 /
        # Table 12 publishes the result, transcribed here. Passing pins the
        # whole permutation — constants, S-box, rotations — to the standard
        # independently of any digest vector.
        self.assertEqual(
            INITIAL_STATE,
            (
                0x9B1E5494E934D681,
                0x4BC3A01E333751D2,
                0xAE65396C6B34B81A,
                0x3C7FD4A4D56A4DB3,
                0x1A5C464906C5976D,
            ),
        )

    def test_sbox_is_a_permutation_with_the_published_anchors(self) -> None:
        # The corner entries of Table 6's two printed rows, plus the
        # permutation property — what a mistyped or shifted row breaks first.
        self.assertCountEqual(SBOX, range(32))
        self.assertEqual(SBOX[0x00], 0x04)
        self.assertEqual(SBOX[0x0F], 0x1C)
        self.assertEqual(SBOX[0x10], 0x1E)
        self.assertEqual(SBOX[0x1F], 0x17)

    def test_padding_is_01_then_zeros_to_the_rate(self) -> None:
        # Algorithm 2 at byte level (§A.2): append 0x01 then zeros — never
        # empty, so a rate-aligned message gains a whole 8-byte block; and in
        # the little-endian convention the marker is the byte 0x01, not the
        # big-endian 0x80 a CAESAR-era transcription would write.
        self.assertEqual(pad(0), b"\x01" + b"\x00" * 7)
        self.assertEqual(pad(7), b"\x01")
        self.assertEqual(pad(8), b"\x01" + b"\x00" * 7)
        for length in (0, 1, 7, 8, 9, 15, 16, 64):
            self.assertEqual((length + len(pad(length))) % 8, 0)


class XofReferenceAnchorTest(parameterized.TestCase):
    """The XOF oracle against the same published record as the hash's."""

    @parameterized.parameters(*((len(m), m, d) for m, d in XOF_KAT_VECTORS))
    def test_matches_the_reference_implementation_kat(
        self, _length: int, msg: bytes, out_hex: str
    ) -> None:
        self.assertEqual(ascon_xof128(msg, len(out_hex) // 2).hex(), out_hex)

    def test_a_short_read_is_a_prefix_of_a_long_one(self) -> None:
        # An XOF's defining property, and what separates a genuine squeeze from
        # a fixed digest truncated: asking for fewer bytes must not change the
        # bytes returned. It also pins the rate-boundary truncation, since 8 is
        # the block and the reads below straddle it.
        full = ascon_xof128(b"abc", 64)
        for n in (1, 7, 8, 9, 32, 63, 64):
            self.assertEqual(ascon_xof128(b"abc", n), full[:n])

    def test_the_xof_is_not_the_hash(self) -> None:
        # The two differ only in the IV, so a transcription that reused
        # Ascon-Hash256's initial state would pass every structural case and
        # fail here.
        self.assertNotEqual(ascon_xof128(b"abc", 32), ascon_hash256(b"abc"))
        self.assertNotEqual(XOF_INITIAL_STATE, INITIAL_STATE)

    def test_the_initial_state_derives_from_the_documented_iv(self) -> None:
        # §B Table 13's layout: Ascon-Hash256's IV with the version byte 3 and
        # a zero output length, which is how "arbitrary output" is encoded.
        self.assertEqual(XOF_IV, 0x0000080000CC0003)
        self.assertEqual(XOF_INITIAL_STATE, tuple(permutation([XOF_IV, 0, 0, 0, 0])))


class CxofReferenceAnchorTest(parameterized.TestCase):
    """Ascon-CXOF128 (§5.3), held to the published record on both axes."""

    @parameterized.parameters(
        *((len(m), len(z), m, z, d) for m, z, d in CXOF_KAT_VECTORS)
    )
    def test_matches_the_reference_implementation_kat(
        self, _mlen: int, _zlen: int, msg: bytes, customization: bytes, out_hex: str
    ) -> None:
        self.assertEqual(
            ascon_cxof128(msg, customization, len(out_hex) // 2).hex(), out_hex
        )

    def test_every_iv_derives_from_one_field_assembly(self) -> None:
        # §B Table 13 assembles an IV from version, round counts, output length
        # and rate: v ‖ 0^8 ‖ a ‖ b ‖ t ‖ r/8 ‖ 0^16, read little-endian. The
        # three rows differ ONLY in the version byte and the output length, so
        # rebuilding all three from that one rule is what makes the CXOF
        # constant a reading of the standard rather than a copy of a copy — a
        # mistyped nibble in it is a mistyped nibble in the two shipped IVs too.
        def iv(version: int, output_bits: int) -> int:
            raw = (
                bytes([version, 0, (12 << 4) + 12])
                + output_bits.to_bytes(2, "little")
                + bytes([RATE, 0, 0])
            )
            return int.from_bytes(raw, "little")

        self.assertEqual(iv(2, 256), IV)
        self.assertEqual(iv(3, 0), XOF_IV)
        self.assertEqual(iv(4, 0), CXOF_IV)
        self.assertEqual(CXOF_IV, 0x0000080000CC0004)
        self.assertEqual(CXOF_INITIAL_STATE, tuple(permutation([CXOF_IV, 0, 0, 0, 0])))

    def test_an_empty_customization_is_not_the_plain_xof(self) -> None:
        # The IVs differ, so CXOF128 with no customization is a DIFFERENT hash
        # from XOF128 rather than the same one. This is what an implementation
        # folding CXOF into XOF as an optional keyword gets wrong, and it is
        # wrong at every input including this one.
        self.assertNotEqual(ascon_cxof128(b"abc", b"", 32), ascon_xof128(b"abc", 32))
        self.assertNotEqual(CXOF_INITIAL_STATE, XOF_INITIAL_STATE)

    def test_the_prefix_is_a_bit_length_then_the_string_then_the_pad(self) -> None:
        # Z0 is the BIT count in eight little-endian bytes — not a byte count,
        # and not big-endian. Both mistakes keep the block count and fail every
        # vector, so pinning the bytes says which one happened.
        self.assertEqual(customization_prefix(b""), (0).to_bytes(8, "little") + pad(0))
        self.assertEqual(
            customization_prefix(b"abc"),
            (24).to_bytes(8, "little") + b"abc" + pad(3),
        )

    def test_the_prefix_is_rate_aligned_at_every_admissible_length(self) -> None:
        # Which is what lets the caller prepend these bytes to the padded
        # message and run ONE absorb loop: if any length left a partial block
        # the two stages could not be one stream.
        for n in range(MAX_CUSTOMIZATION_BYTES + 1):
            self.assertEqual(len(customization_prefix(bytes(n))) % RATE, 0)

    def test_a_customization_over_the_cap_is_refused(self) -> None:
        # §5.3 bounds |Z| at 2048 bits, which is what makes the length field a
        # fixed eight bytes; past it the encoding has no defined meaning.
        customization_prefix(bytes(MAX_CUSTOMIZATION_BYTES))
        with self.assertRaisesRegex(ValueError, "at most"):
            customization_prefix(bytes(MAX_CUSTOMIZATION_BYTES + 1))

    def test_a_short_read_is_a_prefix_of_a_long_one(self) -> None:
        full = ascon_cxof128(b"abc", b"z", 64)
        for n in (1, 7, 8, 9, 32, 63, 64):
            self.assertEqual(ascon_cxof128(b"abc", b"z", n), full[:n])

    def test_the_customization_changes_the_digest(self) -> None:
        # Two customizations of the SAME length, so nothing about the block
        # count or the padding differs — only the absorbed bytes.
        self.assertNotEqual(
            ascon_cxof128(b"abc", b"aaaa", 32), ascon_cxof128(b"abc", b"aaab", 32)
        )


if __name__ == "__main__":
    absltest.main()

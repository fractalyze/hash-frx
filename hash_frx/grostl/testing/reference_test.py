# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The oracle is anchored to the published record before anything is held to it.

`grostl_test` checks the frx Grøstl-256 against `reference.py`, which only
means something if the oracle is right. Oracle and KAT transcription were both
written here, so the anchors are drawn from the outside: the final-round KAT
vectors (three independent sources, provenance on `KAT_VECTORS`), and — for
the S-box, which the oracle derives rather than transcribes — the published
FIPS 197 table's spot values. A wrong shift vector, MixBytes row, round
constant or padding rule does not survive the vectors: every one of them
feeds all ten rounds of both permutations.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from hash_frx.grostl.testing.reference import (
    AES_SBOX,
    KAT_VECTORS,
    grostl256,
    pad,
)


class ReferenceAnchorTest(parameterized.TestCase):
    @parameterized.parameters(*((len(m), m, d) for m, d in KAT_VECTORS))
    def test_matches_the_final_round_kat(
        self, _length: int, msg: bytes, digest_hex: str
    ) -> None:
        self.assertEqual(grostl256(msg).hex(), digest_hex)

    def test_sbox_spot_values_match_the_published_table(self) -> None:
        # The oracle derives the S-box from the FIPS 197 definition; these pin
        # the derivation to the published table (FIPS 197 Figure 7): the two
        # affine fixed inputs' images, the worked section 5.1.1 example 53 ->
        # ed, and the last entry.
        self.assertEqual(AES_SBOX[0x00], 0x63)
        self.assertEqual(AES_SBOX[0x01], 0x7C)
        self.assertEqual(AES_SBOX[0x53], 0xED)
        self.assertEqual(AES_SBOX[0xFF], 0x16)

    def test_sbox_is_a_permutation(self) -> None:
        self.assertCountEqual(AES_SBOX, range(256))

    def test_padding_appends_the_block_count(self) -> None:
        # The length field counts 512-bit blocks of the PADDED message (spec
        # section 3.6), not message bits — the mistake a SHA-2-shaped reading
        # makes. 56 bytes is the boundary: the 9 padding bytes no longer fit
        # the first block, so the count is 2 there and 1 just below.
        self.assertEqual(pad(b"")[-8:], (1).to_bytes(8, "big"))
        self.assertEqual(pad(bytes(55))[-8:], (1).to_bytes(8, "big"))
        self.assertEqual(pad(bytes(56))[-8:], (2).to_bytes(8, "big"))
        self.assertEqual(pad(bytes(64))[-8:], (2).to_bytes(8, "big"))
        for length in (0, 1, 55, 56, 63, 64, 65, 128):
            padded = pad(bytes(length))
            self.assertEqual(len(padded) % 64, 0)
            self.assertEqual(padded[length], 0x80)


if __name__ == "__main__":
    absltest.main()

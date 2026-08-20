# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The oracle is anchored to the published record before anything is held to it.

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
    INITIAL_STATE,
    KAT_VECTORS,
    SBOX,
    ascon_hash256,
    pad,
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


if __name__ == "__main__":
    absltest.main()

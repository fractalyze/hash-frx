# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The oracle is anchored to the published record before anything is held to it.

`ripemd160_test` checks the frx RIPEMD-160 against `ripemd160_reference.py`,
which only means something if the oracle is right. Oracle and vector
transcription were both written here, so the anchors are drawn from the
outside: the designers' nine published vectors (provenance on `VECTORS`),
including the million-"a" record — cheap on the oracle, unlike the device
digest, which would unroll ~15k compression blocks into one graph. A wrong
table entry, added constant, line combination or padding rule does not survive
them: every one feeds all eighty steps of both lines.

Where the interpreter's OpenSSL still carries RIPEMD-160 (OpenSSL 3 moved it
to the legacy provider, so most modern builds do not), `hashlib` serves as a
third, independently-built cross-check at lengths the published vectors skip.
"""

from __future__ import annotations

import hashlib

from absl.testing import absltest, parameterized

from hash_frx.testing.ripemd160_reference import (
    MILLION_A_DIGEST,
    VECTORS,
    pad,
    ripemd160,
)


def _hashlib_has_ripemd160() -> bool:
    try:
        hashlib.new("ripemd160")
    except ValueError:  # unsupported hash type: the legacy-provider gap
        return False
    return True


class ReferenceAnchorTest(parameterized.TestCase):
    @parameterized.parameters(*((len(m), m, d) for m, d in VECTORS))
    def test_matches_the_designers_vectors(
        self, _length: int, msg: bytes, digest_hex: str
    ) -> None:
        self.assertEqual(ripemd160(msg).hex(), digest_hex)

    def test_matches_the_million_a_vector(self) -> None:
        self.assertEqual(ripemd160(b"a" * 1000000).hex(), MILLION_A_DIGEST)

    def test_padding_appends_the_bit_length_little_endian(self) -> None:
        # The 64-bit field is the message BIT length in LITTLE-endian byte
        # order (MD4's padding) — each the opposite of SHA-2, and the
        # documented trap. 55 bytes is the one-block boundary: the 0x80 byte
        # plus the 8-byte field still fit; 56 no longer does.
        self.assertEqual(pad(b"")[-8:], (0).to_bytes(8, "little"))
        self.assertEqual(pad(bytes(55))[-8:], (55 * 8).to_bytes(8, "little"))
        self.assertEqual(len(pad(bytes(55))), 64)
        self.assertEqual(len(pad(bytes(56))), 128)
        for length in (0, 1, 55, 56, 63, 64, 65, 128):
            padded = pad(bytes(length))
            self.assertEqual(len(padded) % 64, 0)
            self.assertEqual(padded[length], 0x80)

    @absltest.skipUnless(
        _hashlib_has_ripemd160(),
        "hashlib lacks ripemd160 (OpenSSL 3 legacy provider)",
    )
    def test_matches_hashlib_where_available(self) -> None:
        # Opportunistic third source: an OpenSSL-lineage implementation the
        # oracle shares nothing with, at every length through the two-block
        # range rather than the vectors' handful.
        for length in range(0, 130):
            msg = bytes((length + i) % 256 for i in range(length))
            self.assertEqual(
                ripemd160(msg),
                hashlib.new("ripemd160", msg).digest(),
                msg=f"length {length}",
            )


if __name__ == "__main__":
    absltest.main()

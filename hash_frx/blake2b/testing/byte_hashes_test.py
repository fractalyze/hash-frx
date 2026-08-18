# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""HostBlake2b: the published vectors, the wrapper's own obligations, the seam.

The row wraps `hashlib.blake2b`, so what needs proving is not BLAKE2b — the
KATs pin that the wrapper actually computes BLAKE2b at the declared length —
but everything the wrapper adds around it: the shared host loop reaching every
batch row, the length riding the value surface, the range check, and the seam
contract (a host row returns `np.ndarray` and reads `HOST`). A differential
against `hashlib` at unpublished lengths would prove nothing here — the row IS
`hashlib` — so that test waits for the device row, whose reference this row
will be.
"""

from __future__ import annotations

import hashlib

import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.blake2b.byte_hashes import MAX_DIGEST_SIZE, HostBlake2b
from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath

# RFC 7693 Appendix A ("abc", BLAKE2b-512) and the empty-message vector from
# the BLAKE2 reference test vectors (github.com/BLAKE2/BLAKE2, testvectors/):
# the two published anchors that catch a wrapper mis-wiring `digest_size` or
# the message bytes.
_KAT_ABC_512 = bytes.fromhex(
    "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1"
    "7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
)
_KAT_EMPTY_512 = bytes.fromhex(
    "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419"
    "d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"
)


def _rows(*messages: bytes) -> np.ndarray:
    return np.array([list(m) for m in messages], dtype=np.uint8)


class KatTest(absltest.TestCase):
    def test_abc_matches_rfc_7693(self) -> None:
        digest = HostBlake2b().digest(_rows(b"abc"))
        self.assertEqual(bytes(digest[0]), _KAT_ABC_512)

    def test_empty_message_matches_the_published_vector(self) -> None:
        digest = HostBlake2b().digest(np.zeros((1, 0), dtype=np.uint8))
        self.assertEqual(bytes(digest[0]), _KAT_EMPTY_512)


class WrapperTest(parameterized.TestCase):
    def test_every_batch_row_is_hashed_independently(self) -> None:
        messages = (b"abc", b"abd", b"ab\x00")
        digests = HostBlake2b().digest(_rows(*messages))
        for i, message in enumerate(messages):
            self.assertEqual(
                bytes(digests[i]),
                hashlib.blake2b(message, digest_size=MAX_DIGEST_SIZE).digest(),
            )

    @parameterized.parameters(1, 20, 32, 48, 64)
    def test_the_declared_length_reaches_the_hash(self, size: int) -> None:
        # Truncating BLAKE2b-512 is the WRONG bytes at every shorter length —
        # the parameter block folds `digest_size` into the IV — so agreement
        # with `hashlib` at the same length is what proves the length was
        # passed through rather than sliced off.
        digest = HostBlake2b(size).digest(_rows(b"abc"))
        self.assertEqual(digest.shape, (1, size))
        self.assertEqual(
            bytes(digest[0]), hashlib.blake2b(b"abc", digest_size=size).digest()
        )

    @parameterized.parameters(0, MAX_DIGEST_SIZE + 1)
    def test_out_of_range_length_is_refused_at_construction(self, size: int) -> None:
        with self.assertRaises(ValueError):
            HostBlake2b(size)


class SeamTest(absltest.TestCase):
    def test_satisfies_the_byte_hash_protocol(self) -> None:
        self.assertIsInstance(HostBlake2b(), ByteHash)

    def test_a_host_row_is_host_and_returns_a_host_array(self) -> None:
        row = HostBlake2b()
        self.assertIs(row.fusion_path, FusionPath.HOST)
        self.assertIsInstance(row.digest(_rows(b"abc")), np.ndarray)

    def test_the_length_rides_the_value_surface(self) -> None:
        # `HostBlake2b(32)` is a different hash from `HostBlake2b(64)`, not one
        # hash asked for fewer bytes — the SHAKE/BLAKE3 rule.
        self.assertEqual(HostBlake2b(32), HostBlake2b(32))
        self.assertEqual(hash(HostBlake2b(32)), hash(HostBlake2b(32)))
        self.assertNotEqual(HostBlake2b(32), HostBlake2b(64))
        self.assertNotEqual(HostBlake2b(32), object())


if __name__ == "__main__":
    absltest.main()

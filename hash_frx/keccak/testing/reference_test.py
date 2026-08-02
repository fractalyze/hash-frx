# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The oracle is anchored to hashlib before anything is held to it.

`permutation_test` checks the frx Keccak-f[1600] against `reference.py`, which
only means something if the reference is right. Both were written here, so
agreement between them would also be the signature of one misreading of FIPS 202
applied twice.

`hashlib` is the independent party: it implements the same standard without
sharing a line with this repo. Driving the reference permutation through a
sponge and matching SHA3-256 and SHAKE128 exercises all 24 rounds and every step
mapping — a wrong rotation offset or round constant does not survive it.
"""

from __future__ import annotations

import hashlib

from absl.testing import absltest

from hash_frx.keccak.testing.reference import keccak_f1600, sponge

# The published Keccak-f[1600] value for the all-zero state (the Keccak team's
# intermediate-value files). Pins the permutation directly, not through a sponge.
_ZERO_STATE_FIRST_LANES = (
    0xF1258F7940E1DDE7,
    0x84D5CCF933C0478A,
    0xD598261EA65AA9EE,
)


class ReferenceAnchorTest(absltest.TestCase):
    def test_permutation_of_the_zero_state_matches_the_published_vector(self) -> None:
        got = keccak_f1600([0] * 25)
        self.assertEqual(tuple(got[:3]), _ZERO_STATE_FIRST_LANES)

    def test_sha3_256_matches_hashlib(self) -> None:
        # Lengths straddle the 136-byte rate so the multi-block path and both
        # padding edges (a full final block, and a block needing one pad byte)
        # are exercised, not just a single-block message.
        for message in (
            b"",
            b"abc",
            b"a" * 135,
            b"a" * 136,
            b"a" * 137,
            b"a" * 272,
            bytes(range(256)),
        ):
            with self.subTest(length=len(message)):
                self.assertEqual(
                    sponge(message, 136, 0x06, 32).hex(),
                    hashlib.sha3_256(message).hexdigest(),
                )

    def test_shake128_matches_hashlib(self) -> None:
        # A different rate (168) and a squeeze longer than one block, so the
        # squeeze-side permute is covered too.
        for message in (b"", b"abc", b"a" * 200):
            with self.subTest(length=len(message)):
                self.assertEqual(
                    sponge(message, 168, 0x1F, 512).hex(),
                    hashlib.shake_128(message).hexdigest(512),
                )


if __name__ == "__main__":
    absltest.main()

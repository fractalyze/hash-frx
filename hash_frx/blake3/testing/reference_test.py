# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The oracle is anchored to the official vectors before anything is held to it.

`compress_test` checks the frx compression function against `reference.py`, which
only means something if the reference is right. Both were written here, so
agreement between them would equally be the signature of one misreading of the
spec applied twice.

The vectors, their input rule, and why they reach a compression function at all
live in `vectors.py`.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from hash_frx.blake3.testing.reference import derive_key, hash_tree, hash_xof, keyed_xof
from hash_frx.blake3.testing.vectors import (
    ALL_LENGTHS,
    CONTEXT,
    EXTENDED,
    EXTENDED_DERIVE_KEY,
    EXTENDED_KEYED,
    KEY,
    official_input,
)

_WIDTH = 131


class ReferenceAnchorTest(parameterized.TestCase):
    @parameterized.parameters(*ALL_LENGTHS)
    def test_official_vectors(self, length: int, expected: str) -> None:
        # Every published length: the single-block ones pin the compression
        # function itself, the multi-block ones add the flag placement and
        # the chaining-value hand-off, and the multi-chunk ones add the
        # parent-node tree. `vectors.py` is where the sets are told apart.
        self.assertEqual(hash_tree(official_input(length)).hex(), expected)

    @parameterized.parameters(*EXTENDED)
    def test_official_extended_vectors(self, length: int, expected: str) -> None:
        # `hash_xof` reaches plumbing `hash_tree` cannot — the output-block
        # counter threaded to a root's last compression and no further — so it
        # needs its own anchor here. Held only against `rows.xof` it would be
        # two implementations of section 2.6 agreeing, which is exactly the
        # agreement this module exists to refuse.
        self.assertEqual(hash_xof(official_input(length), _WIDTH).hex(), expected)

    @parameterized.parameters(*EXTENDED_KEYED)
    def test_official_keyed_vectors(self, length: int, expected: str) -> None:
        # The mode reaches every compression rather than one, so a reading that
        # set `KEYED_HASH` on the root alone — or that opened the parents from
        # the IV while the chunks took the key — still produces well-formed
        # bytes at every length. The whole table is what tells them apart: the
        # single-block rows carry no parent at all, so only the multi-chunk ones
        # can catch a parent opening from the wrong key.
        got = keyed_xof(KEY, official_input(length), _WIDTH)
        self.assertEqual(got.hex(), expected)

    @parameterized.parameters(*EXTENDED_DERIVE_KEY)
    def test_official_derive_key_vectors(self, length: int, expected: str) -> None:
        # The two-pass mode has two flags and two keys, and swapping either pair
        # is invisible below this anchor. It also pins the first pass being read
        # at 32 bytes: a longer read would change nothing about the context hash
        # and everything about the key taken from it.
        got = derive_key(CONTEXT, official_input(length), _WIDTH)
        self.assertEqual(got.hex(), expected)


if __name__ == "__main__":
    absltest.main()

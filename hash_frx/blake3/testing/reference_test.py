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

from hash_frx.blake3.testing.reference import hash_single_chunk
from hash_frx.blake3.testing.vectors import SINGLE_CHUNK, official_input


class ReferenceAnchorTest(parameterized.TestCase):
    @parameterized.parameters(*SINGLE_CHUNK)
    def test_official_vectors(self, length: int, expected: str) -> None:
        # The single-block lengths pin the compression function itself, with
        # nothing built on it in between; the multi-block ones add the
        # CHUNK_START / CHUNK_END placement and the chaining-value hand-off.
        # `vectors.py` is where the two are told apart.
        self.assertEqual(hash_single_chunk(official_input(length)).hex(), expected)

    def test_a_second_chunk_is_out_of_scope(self) -> None:
        # Tree mode belongs to the multi-chunk work; the oracle refuses rather
        # than silently hashing only part of the input.
        with self.assertRaises(ValueError):
            hash_single_chunk(official_input(1025))


if __name__ == "__main__":
    absltest.main()

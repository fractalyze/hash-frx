# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The oracle is anchored to the official vectors before anything is held to it.

`compress_test` checks the frx compression function against `reference.py`, which
only means something if the reference is right. Both were written here, so
agreement between them would equally be the signature of one misreading of the
spec applied twice.

BLAKE3 has no standard-library implementation to serve as the third party the way
`hashlib` does for SHA-3, so the anchor is the BLAKE3 team's own published test
vectors. They pin whole hashes rather than compression intermediates — but for an
input of at most one 1024-byte chunk a BLAKE3 hash *is* this compression function
chained over that chunk's blocks with no tree above it, so the vectors reach the
compression function directly. Inputs of at most 64 bytes reach it in a single
call.

The vectors and their input rule live in `vectors.py`.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from hash_frx.blake3.testing.reference import hash_single_chunk
from hash_frx.blake3.testing.vectors import (
    MULTI_BLOCK,
    SINGLE_BLOCK,
    official_input,
)


class ReferenceAnchorTest(parameterized.TestCase):
    @parameterized.parameters(*SINGLE_BLOCK)
    def test_single_block_vectors(self, length: int, expected: str) -> None:
        # One compression call, so these pin the compression function itself
        # rather than anything built on it.
        self.assertEqual(hash_single_chunk(official_input(length)).hex(), expected)

    @parameterized.parameters(*MULTI_BLOCK)
    def test_multi_block_vectors(self, length: int, expected: str) -> None:
        # Several blocks chained within one chunk: the CHUNK_START / CHUNK_END
        # flag placement and the chaining-value hand-off, which a single-block
        # vector cannot reach.
        self.assertEqual(hash_single_chunk(official_input(length)).hex(), expected)

    def test_a_second_chunk_is_out_of_scope(self) -> None:
        # Tree mode belongs to the multi-chunk work; the oracle refuses rather
        # than silently hashing only part of the input.
        with self.assertRaises(ValueError):
            hash_single_chunk(official_input(1025))


if __name__ == "__main__":
    absltest.main()

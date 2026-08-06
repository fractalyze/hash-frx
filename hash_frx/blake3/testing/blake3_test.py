# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Single-chunk BLAKE3 — values, batching, and lowering.

The published vectors are the anchor rather than a second implementation, and
this module is the one place they pin a whole `digest` end to end: every length
in `SINGLE_CHUNK` is an input a chunk chain hashes with no tree above it, the
empty input and the exact chunk boundary included. `vectors.py` sets out why.

What no vector reaches is the chunk counter, which is 0 in all of them, so a
chunk that dropped it would pass the entire suite. That one is held to
`reference.chunk_output`.

The lowering assertion is not decoration. A gather or a call here still produces
the right digest and only splits the kernel, so values alone cannot catch it.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.blake3.blake3 import (
    BLOCK_LEN,
    CHUNK_LEN,
    chunk_output,
    digest,
    root_words,
)
from hash_frx.blake3.compress import ROOT
from hash_frx.blake3.testing import reference as ref
from hash_frx.blake3.testing.vectors import SINGLE_CHUNK, official_input
from hash_frx.testing.fusion_ready import assert_fusion_ready

_U32 = np.uint32


def _rows(*messages: bytes) -> frx.Array:
    """Equal-length messages as the uint8 `[B, L]` batch `digest` takes."""
    return fnp.asarray(np.array([list(m) for m in messages], dtype=np.uint8))


def _words(data: bytes) -> frx.Array:
    """One chunk as the uint32 `[1, nblocks, 16]` words `chunk_output` takes.

    Cut and packed by the reference, so a test of the chunk chain does not depend
    on the module's own blocking or packing being right.
    """
    blocks = ref.blocks_of(data)
    return fnp.asarray(np.array([[ref.words_of(b) for b in blocks]], dtype=_U32))


def _counter(value: int) -> frx.Array:
    """A chunk index as the (low, high) uint32 pair the seam takes."""
    return fnp.asarray(
        np.array([[value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF]], dtype=_U32)
    )


class DigestAgainstPublishedVectorsTest(parameterized.TestCase):
    @parameterized.parameters(*SINGLE_CHUNK)
    def test_official_vector(self, length: int, expected: str) -> None:
        got = np.asarray(digest(_rows(official_input(length))))
        self.assertEqual(bytes(got[0]).hex(), expected)


class BatchingTest(absltest.TestCase):
    def test_batched_equals_per_message(self) -> None:
        # One call over B messages is B independent hashes: the rows must not
        # interact, which anything reducing across the batch axis would break.
        rng = np.random.default_rng(3)
        messages = [
            bytes(rng.integers(0, 256, size=200, dtype=np.uint8)) for _ in range(4)
        ]
        got = np.asarray(digest(_rows(*messages)))
        for i, message in enumerate(messages):
            with self.subTest(row=i):
                self.assertEqual(bytes(got[i]), ref.hash_single_chunk(message))

    def test_jit_matches_eager(self) -> None:
        # The padding is a constant built from the static length rather than
        # written into the message, so a traced message hashes identically. Two
        # blocks is the smallest input that pads and still chains, and the body
        # is compiled here, so the length is kept to what the property needs.
        rows = _rows(official_input(65), official_input(65))
        np.testing.assert_array_equal(
            np.asarray(frx.jit(digest)(rows)), np.asarray(digest(rows))
        )


class ChunkCounterTest(absltest.TestCase):
    def test_the_counter_reaches_every_block(self) -> None:
        # Chunk 0 is the only counter any published vector hashes, and it is
        # what separates one chunk from the next under a tree.
        data = official_input(200)
        words = _words(data)
        at_zero = np.asarray(root_words(chunk_output(words, len(data), _counter(0))))
        at_seven = np.asarray(root_words(chunk_output(words, len(data), _counter(7))))

        self.assertNotEqual(list(at_zero[0]), list(at_seven[0]))
        self.assertEqual([int(w) for w in at_seven[0]], ref.chunk_output(data, 7, ROOT))


class LoweringTest(absltest.TestCase):
    def test_the_body_is_fusion_ready(self) -> None:
        # The shared whitelist rather than a local blacklist: it also catches a
        # call, a scatter, or a dot, and it reads the lowered module. Three
        # blocks is the smallest input carrying a chained block, a trailing
        # partial one, and both flag placements.
        assert_fusion_ready(digest, _rows(official_input(129)))

    def test_the_digest_is_32_bytes(self) -> None:
        out = digest(_rows(official_input(64), official_input(64)))
        self.assertEqual(out.dtype, fnp.uint8)
        self.assertEqual(out.shape, (2, 32))


class ValidationTest(absltest.TestCase):
    def test_rejects_more_than_one_chunk(self) -> None:
        # Refusing beats hashing a prefix, which would return a perfectly
        # ordinary-looking 32 bytes for the wrong message.
        with self.assertRaises(ValueError):
            digest(_rows(official_input(CHUNK_LEN + 1)))

    def test_rejects_an_unbatched_message(self) -> None:
        with self.assertRaises(ValueError):
            digest(fnp.zeros(BLOCK_LEN, dtype=fnp.uint8))

    def test_rejects_words_that_disagree_with_the_length(self) -> None:
        # The trailing block is padded, so a chunk's byte count cannot be read
        # back off its words — a disagreement is silent unless it is checked.
        words = _words(official_input(200))  # four blocks
        for name, chunk_len in (
            ("too few bytes for the blocks", BLOCK_LEN),
            ("too many", CHUNK_LEN),
            ("not a chunk length at all", CHUNK_LEN + 1),
        ):
            with self.subTest(case=name), self.assertRaises(ValueError):
                chunk_output(words, chunk_len, _counter(0))

    def test_rejects_wrong_word_shapes(self) -> None:
        for name, shape in (
            ("word width", (1, 4, 15)),
            ("rank", (4, 16)),
        ):
            with self.subTest(case=name), self.assertRaises(ValueError):
                chunk_output(fnp.zeros(shape, dtype=fnp.uint32), 200, _counter(0))

    def test_rejects_words_that_are_not_uint32(self) -> None:
        # Wrong-dtype words promote rather than error, so the chain would run to
        # a well-formed digest of the wrong message.
        with self.assertRaises(TypeError):
            chunk_output(fnp.zeros((1, 4, 16), dtype=fnp.int32), 200, _counter(0))


if __name__ == "__main__":
    absltest.main()

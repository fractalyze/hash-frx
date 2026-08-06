# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""BLAKE3 end to end — values, tree shape, batching, and lowering.

The published vectors are the anchor rather than a second implementation, and
this module is the one place they pin a whole `digest` end to end — every length
the standard publishes, from the empty input to a hundred chunks. `vectors.py`
sets out why they reach the layers below a whole hash at all.

Two things they do not isolate get their own cases. A single parent node never
appears alone in a vector, so its four fixed operands are pinned directly; and
the tree *shape* is only exercised at the chunk counts the table happens to
carry, so the gaps are filled against an oracle that builds the shape by the
spec's recursion rather than by pairing levels.

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
    chaining_value,
    chunk_output,
    digest,
    parent_output,
    root_words,
    tree_output,
)
from hash_frx.blake3.compress import ROOT
from hash_frx.blake3.testing import reference as ref
from hash_frx.blake3.testing.vectors import ALL_LENGTHS, official_input
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
    @parameterized.parameters(*ALL_LENGTHS)
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
                self.assertEqual(bytes(got[i]), ref.hash_tree(message))

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


class TreeShapeTest(parameterized.TestCase):
    @parameterized.parameters(2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17)
    def test_the_tree_matches_the_specs_recursion(self, nchunks: int) -> None:
        # The published lengths cover chunk counts 2-9, 16, 31 and 100, which
        # leaves gaps this closes — and it holds the frx tree to an oracle that
        # builds the shape the *other* way, by the spec's largest-power-of-two
        # split rather than by pairing levels. Agreement between two spellings
        # of the shape is the thing worth checking; agreement between two
        # copies of one spelling is not.
        data = official_input(CHUNK_LEN * (nchunks - 1) + 1)
        got = np.asarray(digest(_rows(data)))
        self.assertEqual(bytes(got[0]), ref.hash_tree(data))

    def test_a_halving_tree_would_disagree(self) -> None:
        # The split is the largest power of two strictly below the count, not
        # the midpoint. Three chunks is the smallest input where the two differ:
        # the standard groups (0,1) then 2, a halving would group 0 then (1,2).
        data = official_input(CHUNK_LEN * 2 + 1)
        chunks = [data[i : i + CHUNK_LEN] for i in range(0, len(data), CHUNK_LEN)]
        cvs = [ref.chunk_output(c, i)[:8] for i, c in enumerate(chunks)]

        def parent(left: list[int], right: list[int], flags: int) -> list[int]:
            return ref.compress(list(ref.IV), left + right, 0, BLOCK_LEN, flags)

        standard = parent(
            parent(cvs[0], cvs[1], ref.PARENT)[:8], cvs[2], ref.PARENT | ref.ROOT
        )
        halving = parent(
            cvs[0], parent(cvs[1], cvs[2], ref.PARENT)[:8], ref.PARENT | ref.ROOT
        )
        self.assertNotEqual(standard[:8], halving[:8])
        got = np.asarray(digest(_rows(data)))[0]
        self.assertEqual(
            bytes(got), b"".join(w.to_bytes(4, "little") for w in standard[:8])
        )


class ParentNodeTest(absltest.TestCase):
    def test_a_parent_is_the_spec_parent(self) -> None:
        # No published vector isolates one parent, so its four fixed operands —
        # key words in, counter zero, a full block, PARENT — are pinned here.
        rng = np.random.default_rng(21)
        left = rng.integers(0, 2**32, size=(1, 8), dtype=_U32)
        right = rng.integers(0, 2**32, size=(1, 8), dtype=_U32)
        got = np.asarray(
            chaining_value(parent_output(fnp.asarray(left), fnp.asarray(right)))
        )
        want = ref.compress(
            list(ref.IV),
            [int(w) for w in left[0]] + [int(w) for w in right[0]],
            0,
            BLOCK_LEN,
            ref.PARENT,
        )[:8]
        self.assertEqual([int(w) for w in got[0]], want)

    def test_the_children_are_ordered(self) -> None:
        # Swapping them still produces a well-formed chaining value, so nothing
        # about the shape catches an inverted pair.
        rng = np.random.default_rng(22)
        a = fnp.asarray(rng.integers(0, 2**32, size=(1, 8), dtype=_U32))
        b = fnp.asarray(rng.integers(0, 2**32, size=(1, 8), dtype=_U32))
        self.assertNotEqual(
            list(np.asarray(chaining_value(parent_output(a, b)))[0]),
            list(np.asarray(chaining_value(parent_output(b, a)))[0]),
        )

    def test_root_and_chaining_value_differ_only_by_root(self) -> None:
        # The pending-Output design rests on this: the same node finished two
        # ways is the same compression bar one flag bit.
        data = official_input(200)
        output = chunk_output(_words(data), len(data), _counter(0))
        self.assertNotEqual(
            list(np.asarray(chaining_value(output))[0]),
            list(np.asarray(root_words(output))[0, :8]),
        )
        self.assertEqual(
            [int(w) for w in np.asarray(chaining_value(output))[0]],
            ref.chunk_output(data, 0, 0)[:8],
        )


class TreeBatchingTest(absltest.TestCase):
    def test_rows_do_not_interact_across_a_tree(self) -> None:
        # The tree reshapes between a [B, nodes, 8] view and the flat batch the
        # compression takes, twice per level. A transposed reshape would mix one
        # message's chunks into another's and still return well-formed digests.
        rng = np.random.default_rng(31)
        messages = [
            bytes(rng.integers(0, 256, size=CHUNK_LEN * 3 + 7, dtype=np.uint8))
            for _ in range(3)
        ]
        got = np.asarray(digest(_rows(*messages)))
        for i, message in enumerate(messages):
            with self.subTest(row=i):
                self.assertEqual(bytes(got[i]), ref.hash_tree(message))

    def test_the_chunk_counter_increments_across_the_tree(self) -> None:
        # Every chunk carries its own index, so two messages differing only by a
        # swap of two whole chunks must not hash alike — which they would if the
        # counter were constant.
        rng = np.random.default_rng(32)
        a = bytes(rng.integers(0, 256, size=CHUNK_LEN, dtype=np.uint8))
        b = bytes(rng.integers(0, 256, size=CHUNK_LEN, dtype=np.uint8))
        got = np.asarray(digest(_rows(a + b, b + a)))
        self.assertNotEqual(bytes(got[0]), bytes(got[1]))
        self.assertEqual(bytes(got[0]), ref.hash_tree(a + b))


class PendingOutputTest(absltest.TestCase):
    def test_the_root_node_is_returned_unrun(self) -> None:
        # `tree_output` is the seam the XOF and a keyed root will finish
        # differently, so what it hands back has to be the *unrun* compression:
        # a chunk when the message is one chunk, a parent when it is more.
        one = tree_output(_rows(official_input(CHUNK_LEN)))
        many = tree_output(_rows(official_input(CHUNK_LEN + 1)))
        self.assertEqual(int(np.asarray(one.flags)[0]), ref.CHUNK_END)
        self.assertEqual(int(np.asarray(many.flags)[0]), ref.PARENT)

        # Neither carries ROOT: that belongs to the finishing call alone, which
        # is what lets the same node also be squeezed for extendable output.
        for name, output in (("one chunk", one), ("many", many)):
            with self.subTest(case=name):
                self.assertEqual(int(np.asarray(output.flags)[0]) & ref.ROOT, 0)

    def test_finishing_the_root_reproduces_the_digest(self) -> None:
        data = official_input(CHUNK_LEN * 3 + 5)
        words = root_words(tree_output(_rows(data)))
        got = b"".join(int(w).to_bytes(4, "little") for w in np.asarray(words)[0, :8])
        self.assertEqual(got, ref.hash_tree(data))


class LoweringTest(absltest.TestCase):
    def test_the_tree_body_is_fusion_ready(self) -> None:
        # The level reduction slices with a stride and reshapes between the
        # node view and the flat batch. Both are static, so both must stay
        # `slice`/`reshape` — a dynamic index would lower to a gather, compute
        # the right digest, and only split the kernel.
        assert_fusion_ready(digest, _rows(official_input(CHUNK_LEN * 3 + 1)))

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

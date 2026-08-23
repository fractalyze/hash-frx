# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The tree schedule, independently of the family that runs it.

`extension/tree.py` emits no array operation of its own — every one reaches it
through the caller — so its rules are checked here over plain Python values
rather than against BLAKE3. That is the point: a case written over BLAKE3 passes
with the schedule wrong and BLAKE3's own chunk spelling right, and this is the
layer where the rule actually lives now.

**`levels` is what this file exists for.** The claim it makes is strong — that
pairing adjacent nodes from the bottom and riding an odd trailing node up
produces the *spec's* tree for every chunk count, where the spec instead splits
each node by giving its left subtree the largest power of two strictly below the
count. Those two descriptions have no syntactic resemblance, so the equivalence
is checked against an independent recursive model of the spec's own rule rather
than restated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from absl.testing import absltest

from hash_frx.extension.tree import chain, levels, stream_blocks, units


@dataclass
class _Trace:
    """A node whose value is a string of the compressions that built it, so the
    schedule's call order is readable directly off the result."""

    blocks: list[int] = field(default_factory=list)

    def compress_block(self, node: str, i: int) -> str:
        self.blocks.append(i)
        return f"c{i}({node})"


def _spec_tree(chunks: int) -> tuple:
    """The spec's own recursion (section 2.1), as a nested pair tree over chunk
    indices: a node's left subtree takes the largest power of two STRICTLY below
    its chunk count, and the right takes the remainder.

    Written from the spec's wording rather than from `levels`, which is the whole
    point — it is the independent model the schedule is held to.
    """
    if chunks == 1:
        return (0,)
    split = 1
    while split * 2 < chunks:
        split *= 2
    left = _spec_tree(split)
    right = _tree_shift(_spec_tree(chunks - split), split)
    return (left, right)


def _tree_shift(node: tuple, by: int) -> tuple:
    """Renumber a subtree's chunk indices, so the right subtree counts from where
    the left one stopped."""
    if len(node) == 1:
        return (node[0] + by,)
    return (_tree_shift(node[0], by), _tree_shift(node[1], by))


def _levels_tree(chunks: int) -> tuple:
    """The same tree as `levels` builds it: pair adjacent nodes from the bottom,
    carry an odd trailing node up untouched, stop at two and pair them."""
    nodes: list[tuple] = [(i,) for i in range(chunks)]
    if chunks == 1:
        return nodes[0]
    for pairs, odd in levels(chunks):
        merged: list[tuple] = [(nodes[2 * i], nodes[2 * i + 1]) for i in range(pairs)]
        nodes = merged + ([nodes[-1]] if odd else [])
    assert len(nodes) == 2, nodes
    return (nodes[0], nodes[1])


class UnitsTest(absltest.TestCase):
    def test_empty_still_occupies_one_unit(self) -> None:
        # The floor is the whole subtlety: an empty message is one empty chunk
        # holding one empty block, and the empty digest is a published vector.
        self.assertEqual(units(0, 64), 1)

    def test_exact_and_partial_units(self) -> None:
        self.assertEqual(units(1, 64), 1)
        self.assertEqual(units(64, 64), 1)
        self.assertEqual(units(65, 64), 2)
        self.assertEqual(units(1024, 1024), 1)
        self.assertEqual(units(1025, 1024), 2)


class StreamBlocksTest(absltest.TestCase):
    def test_rounds_up_and_never_returns_zero(self) -> None:
        self.assertEqual(stream_blocks(1, 64), 1)
        self.assertEqual(stream_blocks(64, 64), 1)
        self.assertEqual(stream_blocks(65, 64), 2)
        self.assertEqual(stream_blocks(131, 64), 3)

    def test_differs_from_units_at_zero(self) -> None:
        # `units` floors at one because an empty message still has a block to
        # compress; a zero-byte output request has nothing to read and the
        # caller rejects it before reaching here. Keeping them separate is what
        # stops one question's floor being borrowed for the other's.
        self.assertEqual(stream_blocks(0, 64), 0)
        self.assertEqual(units(0, 64), 1)


class ChainTest(absltest.TestCase):
    def test_every_block_feeds_the_next_in_order(self) -> None:
        t = _Trace()
        out = chain("S", count=3, compress_block=t.compress_block)
        self.assertEqual(t.blocks, [0, 1, 2])
        self.assertEqual(out, "c2(c1(c0(S)))")

    def test_a_single_block_chunk_is_one_compression(self) -> None:
        t = _Trace()
        self.assertEqual(chain("S", count=1, compress_block=t.compress_block), "c0(S)")
        self.assertEqual(t.blocks, [0])

    def test_no_blocks_is_the_node_untouched(self) -> None:
        # A chunk run from past a shared chain has nothing left to fold, which
        # is the `first == last` case BLAKE3 reaches on a one-block chunk.
        t = _Trace()
        self.assertEqual(chain("S", count=0, compress_block=t.compress_block), "S")
        self.assertEqual(t.blocks, [])


class LevelsTest(absltest.TestCase):
    def test_a_single_node_has_no_reduction_to_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two nodes"):
            list(levels(1))

    def test_two_nodes_are_already_the_root_pair(self) -> None:
        self.assertEqual(list(levels(2)), [])

    def test_an_odd_node_rides_up_rather_than_pairing_early(self) -> None:
        # Three nodes: one pair on the bottom level and a trailing node that has
        # no sibling there. Compressing it early is a different tree.
        self.assertEqual(list(levels(3)), [(1, True)])

    def test_a_power_of_two_never_leaves_an_odd_node(self) -> None:
        self.assertEqual(list(levels(8)), [(4, False), (2, False)])

    def test_level_count_is_logarithmic(self) -> None:
        for chunks in (2, 3, 5, 16, 17, 1000):
            with self.subTest(chunks=chunks):
                self.assertEqual(
                    len(list(levels(chunks))), max(0, (chunks - 1).bit_length() - 1)
                )

    def test_matches_the_spec_recursion_at_every_count(self) -> None:
        # The equivalence the module claims, against an independent model of the
        # spec's own largest-power-of-two split.
        for chunks in range(1, 130):
            with self.subTest(chunks=chunks):
                self.assertEqual(_levels_tree(chunks), _spec_tree(chunks))

    def test_matches_the_spec_recursion_past_a_power_of_two(self) -> None:
        # The counts where a naive halving diverges from the spec's split: one
        # past a power of two, where the left subtree must stay perfect.
        for chunks in (513, 1023, 1024, 1025, 2049):
            with self.subTest(chunks=chunks):
                self.assertEqual(_levels_tree(chunks), _spec_tree(chunks))


if __name__ == "__main__":
    absltest.main()

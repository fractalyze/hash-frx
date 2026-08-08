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
    Mode,
    chaining_value,
    chunk_output,
    derive_key,
    digest,
    hash_mode,
    keyed_digest,
    keyed_mode,
    keyed_xof,
    parent_output,
    root_words,
    tree_output,
    xof,
)
from hash_frx.blake3.compress import KEYED_HASH, ROOT
from hash_frx.blake3.testing import reference as ref
from hash_frx.blake3.testing.vectors import (
    ALL_LENGTHS,
    CONTEXT,
    DERIVE_KEY,
    EXTENDED,
    EXTENDED_DERIVE_KEY,
    EXTENDED_KEYED,
    EXTENDED_MULTI_BLOCK,
    EXTENDED_MULTI_CHUNK,
    EXTENDED_SINGLE_BLOCK,
    KEY,
    KEYED,
    PER_LAYER_LENGTHS,
    official_input,
    rows,
)
from hash_frx.testing.fusion_ready import assert_fusion_ready
from hash_frx.word import split

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


def _compressions(length: int, out_len: int = 32) -> int:
    """How many `compress` calls reading `length` bytes out to `out_len` emits.

    Read off the lowered module rather than by patching, so it counts what the
    program actually carries: every call site is a copy of the compression body,
    and a body is exactly 60 xors.
    """
    text = frx.jit(lambda m: xof(m, out_len)).lower(_rows(official_input(length)))
    xors = text.as_text().count("stablehlo.xor")
    assert xors % 60 == 0, f"{xors} xors is not a whole number of compressions"
    return xors // 60


def _counter(value: int) -> frx.Array:
    """A chunk index as the (low, high) uint32 pair the seam takes."""
    return fnp.asarray(np.array([split(value)], dtype=_U32))


class DigestAgainstPublishedVectorsTest(parameterized.TestCase):
    @parameterized.parameters(*ALL_LENGTHS)
    def test_official_vector(self, length: int, expected: str) -> None:
        got = np.asarray(digest(_rows(official_input(length))))
        self.assertEqual(bytes(got[0]).hex(), expected)


class BatchingTest(parameterized.TestCase):
    @parameterized.named_parameters(("one chunk", 200), ("a tree", CHUNK_LEN * 3 + 7))
    def test_batched_equals_per_message(self, size: int) -> None:
        # One call over B messages is B independent hashes: the rows must not
        # interact, which anything reducing across the batch axis would break.
        # Under a tree there is a second way to break it — the reshapes between
        # the [B, nodes, 8] view and the flat batch the compression takes would
        # land one message's chunks on another's row and still return
        # well-formed digests.
        rng = np.random.default_rng(3)
        messages = [
            bytes(rng.integers(0, 256, size=size, dtype=np.uint8)) for _ in range(3)
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
        at_zero = np.asarray(
            root_words(chunk_output(words, len(data), _counter(0), hash_mode()))
        )
        at_seven = np.asarray(
            root_words(chunk_output(words, len(data), _counter(7), hash_mode()))
        )

        self.assertNotEqual(list(at_zero[0]), list(at_seven[0]))
        self.assertEqual([int(w) for w in at_seven[0]], ref.chunk_output(data, 7, ROOT))

    def test_the_counter_increments_across_the_tree(self) -> None:
        # Every chunk carries its own index, so two messages differing only by a
        # swap of two whole chunks must not hash alike — which they would if the
        # counter were constant across the batched chunk call.
        rng = np.random.default_rng(32)
        a = bytes(rng.integers(0, 256, size=CHUNK_LEN, dtype=np.uint8))
        b = bytes(rng.integers(0, 256, size=CHUNK_LEN, dtype=np.uint8))
        got = np.asarray(digest(_rows(a + b, b + a)))
        self.assertNotEqual(bytes(got[0]), bytes(got[1]))
        self.assertEqual(bytes(got[0]), ref.hash_tree(a + b))


class TreeShapeTest(parameterized.TestCase):
    @parameterized.parameters(12, 15, 17)
    def test_the_tree_matches_the_specs_recursion(self, nchunks: int) -> None:
        # Only the counts the published table does not reach. It covers 2-9, 16,
        # 31 and 100, and `official_input(CHUNK_LEN * (n - 1) + 1)` for n in 2-9
        # *is* a published length byte for byte, so those would re-run
        # `DigestAgainstPublishedVectorsTest` against a weaker oracle.
        #
        # What these add is the shape gap, checked against an oracle that builds
        # the tree the *other* way — by the spec's largest-power-of-two split
        # rather than by pairing levels. Agreement between two spellings of the
        # shape is worth something; between two copies of one spelling, nothing.
        data = official_input(CHUNK_LEN * (nchunks - 1) + 1)
        got = np.asarray(digest(_rows(data)))
        self.assertEqual(bytes(got[0]), ref.hash_tree(data))

    def test_a_halving_tree_would_disagree(self) -> None:
        # The split is the largest power of two strictly below the count, not
        # the midpoint. Three chunks is the smallest input where the two differ:
        # the standard groups (0,1) then 2, a halving would group 0 then (1,2).
        data = official_input(CHUNK_LEN * 2 + 1)
        chunks = ref.blocks_of(data, CHUNK_LEN)
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


class TailChunkTest(parameterized.TestCase):
    # Where the last chunk leaves the shared chain, which is its block count.
    # The published tails are only ever 1 byte or a whole chunk, so every value
    # between them is unreached by the vector suite — and it is exactly what
    # decides how far the short chunk rides along. One length per distinct block
    # count, plus both sides of each boundary `_units` rounds at.
    @parameterized.parameters(1, 64, 65, 129, 512, 960, 961, 1023)
    def test_a_short_last_chunk_hashes_the_same_as_the_oracle(self, tail: int) -> None:
        rng = np.random.default_rng(tail)
        data = bytes(rng.integers(0, 256, size=CHUNK_LEN * 2 + tail, dtype=np.uint8))
        got = np.asarray(digest(_rows(data)))
        self.assertEqual(bytes(got[0]), ref.hash_tree(data))


class ParentNodeTest(absltest.TestCase):
    def test_a_parent_is_the_spec_parent(self) -> None:
        # No published vector isolates one parent, so its four fixed operands —
        # key words in, counter zero, a full block, PARENT — are pinned here.
        rng = np.random.default_rng(21)
        left = rng.integers(0, 2**32, size=(1, 8), dtype=_U32)
        right = rng.integers(0, 2**32, size=(1, 8), dtype=_U32)
        got = np.asarray(
            chaining_value(
                parent_output(fnp.asarray(left), fnp.asarray(right), hash_mode())
            )
        )
        want = ref.compress(
            list(ref.IV),
            [int(w) for w in left[0]] + [int(w) for w in right[0]],
            0,
            BLOCK_LEN,
            ref.PARENT,
        )[:8]
        self.assertEqual([int(w) for w in got[0]], want)

    def test_rejects_children_it_cannot_pair(self) -> None:
        # The one entry a caller hands raw arrays to. Without the guard a wrong
        # child shape is reported against `block` — an operand the caller never
        # passed — and a wrong dtype promotes into well-formed wrong words.
        good = fnp.zeros((2, 8), dtype=fnp.uint32)
        for name, left, right, err in (
            ("width", fnp.zeros((2, 4), dtype=fnp.uint32), good, ValueError),
            ("rank", fnp.zeros(8, dtype=fnp.uint32), good, ValueError),
            (
                "batch disagreement",
                fnp.zeros((3, 8), dtype=fnp.uint32),
                good,
                ValueError,
            ),
            ("dtype", fnp.zeros((2, 8), dtype=fnp.int32), good, TypeError),
            ("dtype on the right", good, fnp.zeros((2, 8), dtype=fnp.int32), TypeError),
        ):
            with self.subTest(case=name), self.assertRaises(err):
                parent_output(left, right, hash_mode())

    def test_the_children_are_ordered(self) -> None:
        # Swapping them still produces a well-formed chaining value, so nothing
        # about the shape catches an inverted pair.
        rng = np.random.default_rng(22)
        a = fnp.asarray(rng.integers(0, 2**32, size=(1, 8), dtype=_U32))
        b = fnp.asarray(rng.integers(0, 2**32, size=(1, 8), dtype=_U32))
        self.assertNotEqual(
            list(np.asarray(chaining_value(parent_output(a, b, hash_mode())))[0]),
            list(np.asarray(chaining_value(parent_output(b, a, hash_mode())))[0]),
        )

    def test_root_and_chaining_value_differ_only_by_root(self) -> None:
        # The pending-Output design rests on this: the same node finished two
        # ways is the same compression bar one flag bit.
        data = official_input(200)
        output = chunk_output(_words(data), len(data), _counter(0), hash_mode())
        cv = [int(w) for w in np.asarray(chaining_value(output))[0]]
        self.assertNotEqual(cv, [int(w) for w in np.asarray(root_words(output))[0, :8]])
        self.assertEqual(cv, ref.chunk_output(data, 0, 0)[:8])

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
        one = tree_output(_rows(official_input(CHUNK_LEN)), hash_mode())
        many = tree_output(_rows(official_input(CHUNK_LEN + 1)), hash_mode())
        # Exact equality, so it also says neither carries ROOT — that belongs
        # to the finishing call alone, which is what lets the same node be
        # squeezed for extendable output instead.
        self.assertEqual(int(np.asarray(one.flags)[0]), ref.CHUNK_END)
        self.assertEqual(int(np.asarray(many.flags)[0]), ref.PARENT)


class ExtendableOutputTest(parameterized.TestCase):
    @parameterized.parameters(*EXTENDED)
    def test_official_extended_output(self, length: int, expected: str) -> None:
        # The published `hash` field is 131 bytes, of which the digest is the
        # first 32. So the same table anchors the stream, at every input length,
        # with no second source — and 131 bytes is three output blocks and a
        # 3-byte remainder, so the counter running 0/1/2 and the partial tail
        # are both on the path here.
        got = np.asarray(xof(_rows(official_input(length)), 131))
        self.assertEqual(bytes(got[0]).hex(), expected)

    @parameterized.parameters(
        EXTENDED_SINGLE_BLOCK[-1], EXTENDED_MULTI_BLOCK[0], EXTENDED_MULTI_CHUNK[-1]
    )
    def test_the_digest_is_the_head_of_the_stream(
        self, length: int, expected: str
    ) -> None:
        # That the two published columns agree is structural in `vectors.py` —
        # the digest one is sliced off this one — so what is left to check is
        # that `digest` and `xof` agree with it *through the implementation*,
        # once per layer. One row each: a chunk with no tree, a chunk that
        # chains, and a parent-node tree.
        message = _rows(official_input(length))
        self.assertEqual(bytes(np.asarray(digest(message))[0]).hex(), expected[:64])
        self.assertEqual(bytes(np.asarray(xof(message, 131))[0]).hex(), expected)

    @parameterized.parameters(1, 63, 64, 65, 128, 200)
    def test_every_output_length_is_a_prefix(self, out_len: int) -> None:
        # A partial trailing output block is a slice of a whole one, so a length
        # that is not a multiple of 64 must not disturb the bytes before it.
        # One length per shape: short of a block, one short, exact, one over,
        # an exact multiple, and a partial tail several blocks in.
        #
        # Read at a multi-block single chunk. That is the root kind the counter
        # subtlety bites — its leading blocks already ran at the chunk's index
        # and must not be re-run at the output counter — and it is the case
        # `reference.py` names for the same reason. A larger input would cost
        # a tree per parameter and pin nothing this does not.
        data = official_input(129)
        got = np.asarray(xof(_rows(data), out_len))
        self.assertEqual(out_len, got.shape[1])
        self.assertEqual(bytes(got[0]), ref.hash_xof(data, out_len))

    def test_rejects_an_empty_output(self) -> None:
        with self.assertRaises(ValueError):
            xof(_rows(official_input(64)), 0)

    def test_rows_do_not_interact_across_the_stream(self) -> None:
        # Every operand is spread one row per output block before the batched
        # call, so the rows of a message sit next to each other rather than
        # interleaved with another message's. At a batch of one those two
        # layouts are the same array, which is why this needs three distinct
        # messages: a transposed spread hands row 0's node to row 1's counter
        # and still returns well-formed bytes of the right length.
        rng = np.random.default_rng(41)
        messages = [
            bytes(rng.integers(0, 256, size=CHUNK_LEN + 7, dtype=np.uint8))
            for _ in range(3)
        ]
        got = np.asarray(xof(_rows(*messages), 131))
        for i, message in enumerate(messages):
            with self.subTest(row=i):
                self.assertEqual(bytes(got[i]), ref.hash_xof(message, 131))

    def test_the_stream_is_not_the_digest_repeated(self) -> None:
        # The counter is what separates one output block from the next, so a
        # version that dropped it would return the same 64 bytes over and over
        # — well-formed, the right length, and wrong.
        got = np.asarray(xof(_rows(official_input(64)), 128))[0]
        self.assertNotEqual(bytes(got[:64]), bytes(got[64:]))


class KeyedHashTest(parameterized.TestCase):
    @parameterized.parameters(*KEYED)
    def test_official_keyed_vector(self, length: int, expected: str) -> None:
        # The whole published table under the vectors' own key. The layers it
        # names are what makes it worth running entire here rather than at a
        # length or two: below 65 bytes there is no parent to open from the key
        # at all, so only the multi-chunk rows can catch a tree that kept the IV
        # while its chunks took the key.
        got = np.asarray(keyed_digest(KEY, _rows(official_input(length))))
        self.assertEqual(bytes(got[0]).hex(), expected)

    @parameterized.parameters(*rows(EXTENDED_KEYED, PER_LAYER_LENGTHS))
    def test_official_keyed_extended_output(self, length: int, expected: str) -> None:
        # One row per layer rather than the table again. Reading further is
        # section 2.6 repeated on a node the mode has already reached, and
        # `ExtendableOutputTest` pins that at all 35 lengths; what is left is
        # that a keyed node survives being read as a stream rather than a digest.
        got = np.asarray(keyed_xof(KEY, _rows(official_input(length)), 131))
        self.assertEqual(bytes(got[0]).hex(), expected)

    def test_keyed_output_differs_from_unkeyed(self) -> None:
        # Read at an UNpublished length, which is the only way this says
        # anything: at a published one both sides are already pinned to two
        # different hex strings above, so the inequality would follow from
        # assertions already made rather than from running anything.
        message = _rows(official_input(100))
        self.assertNotEqual(
            bytes(np.asarray(keyed_digest(KEY, message))[0]),
            bytes(np.asarray(digest(message))[0]),
        )

    def test_a_different_key_is_a_different_hash(self) -> None:
        # The key reaches every compression, so nothing about the message can
        # make two keys agree.
        message = _rows(official_input(200))
        self.assertNotEqual(
            bytes(np.asarray(keyed_digest(KEY, message))[0]),
            bytes(np.asarray(keyed_digest(bytes(len(KEY)), message))[0]),
        )

    def test_the_mode_flag_rides_on_the_unrun_node(self) -> None:
        # `root_words` only ever ORs ROOT, so a mode that reached the key words
        # and not the flags would be absent from every compression and visible
        # nowhere but here. Exact equality, so it also says ROOT is still the
        # finishing call's alone.
        # 128 bytes rather than a whole chunk: two blocks is the shortest
        # input whose last block is not also its first, so CHUNK_START is
        # absent and the value read is the same. A full chunk runs fifteen
        # compressions whose output this discards.
        node = tree_output(_rows(official_input(2 * BLOCK_LEN)), keyed_mode(KEY))
        self.assertEqual(int(np.asarray(node.flags)[0]), ref.CHUNK_END | ref.KEYED_HASH)

    def test_rows_do_not_interact_under_a_key(self) -> None:
        # One key is broadcast across the rows rather than tiled along them, and
        # at a batch of one those two layouts are the same array — so this needs
        # three distinct messages, checked per row against the oracle.
        rng = np.random.default_rng(39)
        messages = [
            bytes(rng.integers(0, 256, size=CHUNK_LEN + 7, dtype=np.uint8))
            for _ in range(3)
        ]
        got = np.asarray(keyed_xof(KEY, _rows(*messages), 131))
        for i, message in enumerate(messages):
            with self.subTest(row=i):
                self.assertEqual(bytes(got[i]), ref.keyed_xof(KEY, message, 131))

    def test_the_key_may_be_a_tracer(self) -> None:
        # The key is an operand, not a host constant, so one compiled program
        # serves every key — which is what lets a consumer re-key without
        # re-tracing and keeps secret material out of the constant pool. A key
        # that had been baked in fails this as a tracer error rather than as
        # wrong bytes.
        message = _rows(official_input(65))
        key = fnp.asarray(np.frombuffer(KEY, dtype=np.uint8))
        np.testing.assert_array_equal(
            np.asarray(frx.jit(keyed_digest)(key, message)),
            np.asarray(keyed_digest(KEY, message)),
        )

    @parameterized.named_parameters(("short", 31), ("long", 33), ("empty", 0))
    def test_rejects_a_key_that_is_not_32_bytes(self, size: int) -> None:
        # A key of the wrong width has no silent failure to fall into — it would
        # reach `pack_le` and change the chaining-value shape — but the error it
        # raises there names an operand the caller never passed.
        with self.assertRaises(ValueError):
            keyed_digest(bytes(size), _rows(official_input(64)))


class DeriveKeyTest(parameterized.TestCase):
    @parameterized.parameters(*DERIVE_KEY)
    def test_official_derive_key_vector(self, length: int, expected: str) -> None:
        # The published context string, at every published length of material.
        got = np.asarray(derive_key(CONTEXT, _rows(official_input(length)), 32))
        self.assertEqual(bytes(got[0]).hex(), expected)

    @parameterized.parameters(*rows(EXTENDED_DERIVE_KEY, PER_LAYER_LENGTHS))
    def test_official_derive_key_extended_output(
        self, length: int, expected: str
    ) -> None:
        # One row per layer, for the reason the keyed case gives.
        got = np.asarray(derive_key(CONTEXT, _rows(official_input(length)), 131))
        self.assertEqual(bytes(got[0]).hex(), expected)

    def test_the_context_separates(self) -> None:
        # What the mode exists for. Two contexts over one key material derive
        # unrelated keys, which is the property a consumer relies on to reuse
        # one secret across purposes.
        material = _rows(official_input(200))
        self.assertNotEqual(
            bytes(np.asarray(derive_key("one", material, 32))[0]),
            bytes(np.asarray(derive_key("two", material, 32))[0]),
        )

    def test_the_context_and_the_material_are_not_interchangeable(self) -> None:
        # The context is the domain and the message is the secret, which is the
        # inverse of the reading the argument order invites. Swapped, this
        # derives well-formed bytes of the wrong thing and nothing about the
        # shapes objects.
        one, two = b"context one", b"context two"
        self.assertNotEqual(
            bytes(np.asarray(derive_key(one, _rows(two), 32))[0]),
            bytes(np.asarray(derive_key(two, _rows(one), 32))[0]),
        )

    @parameterized.named_parameters(
        ("empty", 0), ("one block", BLOCK_LEN), ("a tree", CHUNK_LEN * 2 + 1)
    )
    def test_the_context_pass_is_a_whole_hash(self, size: int) -> None:
        # Every published row shares one 47-byte context, which is a single
        # block of a single chunk — so the table says nothing about a context
        # that chains or that stands a tree above itself. It is the same
        # `tree_output` the material pass runs, and these are the lengths that
        # say so rather than assume it.
        context = bytes(official_input(size))
        material = official_input(200)
        got = np.asarray(derive_key(context, _rows(material), 32))
        self.assertEqual(bytes(got[0]), ref.derive_key(context, material, 32))

    def test_a_str_context_is_its_utf8_bytes(self) -> None:
        # The standard names the context UTF-8, so the two spellings are one
        # context rather than two.
        material = _rows(official_input(200))
        np.testing.assert_array_equal(
            np.asarray(derive_key(CONTEXT, material, 32)),
            np.asarray(derive_key(CONTEXT.encode(), material, 32)),
        )

    def test_the_material_may_be_a_tracer(self) -> None:
        # The context is a host constant by the standard's own instruction, but
        # the material is a secret a consumer hashes inside its own jit.
        material = _rows(official_input(65))
        np.testing.assert_array_equal(
            np.asarray(frx.jit(lambda m: derive_key(CONTEXT, m, 32))(material)),
            np.asarray(derive_key(CONTEXT, material, 32)),
        )


class LoweringTest(absltest.TestCase):
    def test_output_length_costs_no_compressions(self) -> None:
        # The output blocks are the rows of ONE batched call — each reads the
        # same root node and differs only in its counter — so the program does
        # not grow with the output at all. Sixteen blocks emit what one does.
        #
        # The arithmetic does grow, of course: those are sixteen rows of that
        # call. What is pinned here is that they are rows and not call sites,
        # which is the difference between this and a sponge's squeeze, and the
        # thing a sequential rewrite would silently give up.
        #
        # Read at a one-block input, where the whole program is that single
        # compression: the claim is then 1 site against the 16 a sequential
        # squeeze would emit, rather than 17 against 32 — the tree underneath
        # is not what this is about, and paying for it here buys nothing.
        for out_len in (32, 131, 1024):
            with self.subTest(out_len=out_len):
                self.assertEqual(_compressions(BLOCK_LEN, out_len), 1)

    def test_the_extendable_body_is_fusion_ready(self) -> None:
        # The spread that widens one node to a row per output block is the new
        # shape here, and it has to stay `broadcast_in_dim`/`reshape` — a
        # per-row gather would return the same bytes and split the kernel. The
        # tree beneath is covered by the two cases below, so this reads at the
        # smallest input with more than one output block to spread.
        assert_fusion_ready(lambda m: xof(m, 131), _rows(official_input(129)))

    def test_a_short_last_chunk_costs_no_compressions(self) -> None:
        # A short last chunk shares the batched chain to its own last block and
        # rejoins the full chunks to finish, so it emits nothing extra at all.
        # Handled apart it is `16 + m` call sites for a chunk of `m` blocks —
        # at a full-length tail nearly twice the program, for a message one byte
        # *shorter*. No digest catches that, and unaligned is the ordinary case.
        #
        # Counted rather than bounded: a ratio wide enough not to be brittle is
        # also wide enough to admit several extra compressions. One `compress`
        # emits exactly 60 xors — 7 rounds of 8 in `G`, plus 4 feeding forward.
        aligned = _compressions(CHUNK_LEN * 2)
        self.assertEqual(aligned, 17)
        for tail in (1, 64, 512, 1023):
            with self.subTest(tail=tail):
                self.assertEqual(_compressions(CHUNK_LEN + tail), aligned)

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

    def test_the_keyed_body_is_fusion_ready(self) -> None:
        # The key is the one operand hash mode does not carry, and it is
        # broadcast across the rows and packed from bytes. Both are the shapes
        # `conventions.md` warns lower to a gather when written as an indexed
        # read — right bytes, split kernel.
        key = fnp.asarray(np.frombuffer(KEY, dtype=np.uint8))
        assert_fusion_ready(keyed_digest, key, _rows(official_input(129)))

    def test_the_derive_key_body_is_fusion_ready(self) -> None:
        # Two whole hashes in one body, the first over a host constant. It is
        # the one place a mode's key is *computed* rather than passed, so it is
        # the one place the key could arrive as something other than a broadcast.
        assert_fusion_ready(
            lambda m: derive_key(CONTEXT, m, 32), _rows(official_input(129))
        )

    def test_the_digest_is_32_bytes(self) -> None:
        out = digest(_rows(official_input(64), official_input(64)))
        self.assertEqual(out.dtype, fnp.uint8)
        self.assertEqual(out.shape, (2, 32))


class ValidationTest(absltest.TestCase):
    def test_rejects_an_unbatched_message(self) -> None:
        with self.assertRaises(ValueError):
            digest(fnp.zeros(BLOCK_LEN, dtype=fnp.uint8))

    def test_rejects_key_words_it_cannot_open_a_node_from(self) -> None:
        # `Mode` is the second record a caller hands raw arrays to, so it names
        # the mistake for the reason `parent_output` does. A scheme holding its
        # own key words builds one directly and never passes `keyed_mode`, so
        # the width and dtype are unchecked anywhere else — and a `[1, 8]` or
        # int32 key reaches `compress` to be reported against `chaining_value`,
        # an operand the caller never passed.
        for name, key, err in (
            ("batched", fnp.zeros((1, 8), dtype=fnp.uint32), ValueError),
            ("width", fnp.zeros(4, dtype=fnp.uint32), ValueError),
            ("dtype", fnp.zeros(8, dtype=fnp.int32), TypeError),
        ):
            with self.subTest(case=name), self.assertRaises(err):
                Mode(key, KEYED_HASH)

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
                chunk_output(words, chunk_len, _counter(0), hash_mode())

    def test_rejects_wrong_word_shapes(self) -> None:
        for name, shape in (
            ("word width", (1, 4, 15)),
            ("rank", (4, 16)),
        ):
            with self.subTest(case=name), self.assertRaises(ValueError):
                chunk_output(
                    fnp.zeros(shape, dtype=fnp.uint32), 200, _counter(0), hash_mode()
                )

    def test_rejects_words_that_are_not_uint32(self) -> None:
        # Wrong-dtype words promote rather than error, so the chain would run to
        # a well-formed digest of the wrong message.
        with self.assertRaises(TypeError):
            chunk_output(
                fnp.zeros((1, 4, 16), dtype=fnp.int32), 200, _counter(0), hash_mode()
            )


if __name__ == "__main__":
    absltest.main()

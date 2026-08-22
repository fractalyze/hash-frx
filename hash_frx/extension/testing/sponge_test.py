# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The sponge schedules, independently of any family that runs them.

`absorb_squeeze` emits no array operation of its own — every one reaches it
through the caller's `absorb` / `permute` / `read` — so its rules are checked
here against a recording fake over plain Python values rather than against a
hash. That is the point of the fake: a case written over Keccak or Ascon would
pass with the schedule wrong and the family's own peeling or guarding right, and
this is the layer where the rule actually lives now.

**The squeeze rule is what this file exists for.** The absorb's final
permutation has already run when the squeeze starts, so the first output block
is available immediately and no permutation follows the last read. Both
mis-spellings are pinned as failing: permuting before the first read shifts
every output block by one permutation, and permuting after the last one is work
nothing reads. The first is a wrong digest on every message, which no
byte-exactness suite catches if the oracle is transcribed from the same mistake.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hash_frx.extension.sponge import (
    absorb_squeeze,
    field_absorb,
    merge_into_rate,
    squeeze_blocks,
)


@dataclass
class _Trace:
    """A sponge whose state is a string of the operations that built it, so the
    schedule's call order is readable directly off the result."""

    permutes: int = 0
    absorbs: list[int] = field(default_factory=list)

    def absorb(self, state: str, i: int) -> str:
        self.absorbs.append(i)
        return f"{state}+a{i}"

    def permute(self, state: str) -> str:
        self.permutes += 1
        return f"p({state})"

    def read(self, state: str) -> str:
        return state


class AbsorbSqueezeScheduleTest(absltest.TestCase):
    def test_every_absorbed_block_is_followed_by_a_permutation(self) -> None:
        t = _Trace()
        out = absorb_squeeze(
            "S", blocks=3, squeezes=1, absorb=t.absorb, permute=t.permute, read=t.read
        )
        self.assertEqual(t.absorbs, [0, 1, 2])
        self.assertEqual(out, ["p(p(p(S+a0)+a1)+a2)"])

    def test_the_first_read_needs_no_further_permutation(self) -> None:
        # The absorb's final permutation has already run, so the first output
        # block is available before any further one. A schedule that permuted
        # first would shift every block by one permutation.
        t = _Trace()
        out = absorb_squeeze(
            "S", blocks=1, squeezes=3, absorb=t.absorb, permute=t.permute, read=t.read
        )
        self.assertEqual(out[0], "p(S+a0)")
        self.assertEqual(out, ["p(S+a0)", "p(p(S+a0))", "p(p(p(S+a0)))"])

    def test_no_permutation_follows_the_last_read(self) -> None:
        # `blocks` permutations for the absorb, `squeezes - 1` between the
        # reads, and none after — so the count is exact rather than bounded.
        for blocks in (0, 1, 5):
            for squeezes in (1, 2, 4):
                with self.subTest(blocks=blocks, squeezes=squeezes):
                    t = _Trace()
                    absorb_squeeze(
                        "S",
                        blocks=blocks,
                        squeezes=squeezes,
                        absorb=t.absorb,
                        permute=t.permute,
                        read=t.read,
                    )
                    self.assertEqual(t.permutes, blocks + squeezes - 1)

    def test_a_single_squeeze_permutes_only_for_the_absorb(self) -> None:
        t = _Trace()
        absorb_squeeze(
            "S", blocks=2, squeezes=1, absorb=t.absorb, permute=t.permute, read=t.read
        )
        self.assertEqual(t.permutes, 2)

    def test_the_reads_come_back_in_order(self) -> None:
        out = absorb_squeeze(
            0,
            blocks=0,
            squeezes=4,
            absorb=lambda s, i: s,
            permute=lambda s: s + 1,
            read=lambda s: s,
        )
        self.assertEqual(out, [0, 1, 2, 3])

    def test_the_schedule_never_looks_inside_a_read(self) -> None:
        # `read` may hand back any shape the caller assembles — Keccak reads one
        # lane slice, Ascon a (lo, hi) pair — and the schedule only collects it.
        out = absorb_squeeze(
            "S",
            blocks=0,
            squeezes=2,
            absorb=lambda s, i: s,
            permute=lambda s: s,
            read=lambda s: (s, s),
        )
        self.assertEqual(out, [("S", "S"), ("S", "S")])

    def test_an_empty_absorb_still_squeezes(self) -> None:
        # A caller whose padding produced no block would still owe its output;
        # nothing in the package does, but the schedule must not require one.
        t = _Trace()
        out = absorb_squeeze(
            "S", blocks=0, squeezes=2, absorb=t.absorb, permute=t.permute, read=t.read
        )
        self.assertEqual(t.absorbs, [])
        self.assertEqual(out, ["S", "p(S)"])


class SqueezeBlocksTest(absltest.TestCase):
    def test_rounds_up(self) -> None:
        # SHA3-256 reads its 32 bytes out of one 136-byte rate; SHAKE256 at 64
        # bytes still fits one; a 200-byte request spans two.
        self.assertEqual(squeeze_blocks(32, 136), 1)
        self.assertEqual(squeeze_blocks(64, 136), 1)
        self.assertEqual(squeeze_blocks(136, 136), 1)
        self.assertEqual(squeeze_blocks(137, 136), 2)
        self.assertEqual(squeeze_blocks(200, 136), 2)
        # Ascon-Hash256: 32 bytes at an 8-byte rate is the four reads its
        # schedule used to spell as one peeled read plus a loop of three.
        self.assertEqual(squeeze_blocks(32, 8), 4)


class MergeIntoRateTest(absltest.TestCase):
    def test_only_the_rate_prefix_changes(self) -> None:
        state = fnp.asarray(np.arange(10, dtype=np.uint32))
        block = fnp.asarray(np.array([0xFF, 0xFF, 0xFF], dtype=np.uint32))
        got = np.asarray(merge_into_rate(state, block, operator.xor))
        want = np.arange(10, dtype=np.uint32)
        want[:3] ^= np.uint32(0xFF)
        np.testing.assert_array_equal(got, want)

    def test_one_spelling_serves_both_state_ranks(self) -> None:
        # The batched sponge and the incremental one carry the same state at
        # different ranks; the trailing-axis indexing is what lets one helper
        # serve both, and a rank-0-indexed one would silently merge along the
        # batch axis instead.
        row = np.arange(10, dtype=np.uint32)
        block = np.array([1, 2, 3], dtype=np.uint32)
        unbatched = np.asarray(
            merge_into_rate(fnp.asarray(row), fnp.asarray(block), operator.xor)
        )
        batched = np.asarray(
            merge_into_rate(
                fnp.asarray(np.stack([row, row + 100])),
                fnp.asarray(np.stack([block, block])),
                operator.xor,
            )
        )
        np.testing.assert_array_equal(batched[0], unbatched)

    def test_the_operation_is_the_caller_s(self) -> None:
        # XOR for the byte sponges; the parameter exists so a construction that
        # merges differently is not a second copy of the concatenate.
        state = fnp.asarray(np.arange(5, dtype=np.uint32))
        block = fnp.asarray(np.array([10, 10], dtype=np.uint32))
        np.testing.assert_array_equal(
            np.asarray(merge_into_rate(state, block, operator.add))[:2],
            np.array([10, 11], dtype=np.uint32),
        )

    def test_a_full_width_block_leaves_no_capacity(self) -> None:
        state = fnp.asarray(np.zeros(4, dtype=np.uint32))
        block = fnp.asarray(np.arange(4, dtype=np.uint32))
        np.testing.assert_array_equal(
            np.asarray(merge_into_rate(state, block, operator.xor)),
            np.arange(4, dtype=np.uint32),
        )


class FieldAbsorbTest(absltest.TestCase):
    def test_runs_the_block_count(self) -> None:
        state = fnp.asarray(np.zeros(4, dtype=np.int32))
        got = np.asarray(
            field_absorb(
                state,
                blocks=3,
                absorb=lambda s, i: s + fnp.int32(1),
                permute=lambda s: s * fnp.int32(2),
            )
        )
        # ((0+1)*2 + 1)*2 + 1)*2 = 14
        np.testing.assert_array_equal(got, np.full(4, 14, dtype=np.int32))

    def test_the_index_reaches_the_absorb_as_a_tracer(self) -> None:
        # The whole difference from `absorb_squeeze`: the block index is runtime
        # data here, which is what lets the bound be read at runtime.
        state = fnp.asarray(np.zeros(3, dtype=np.int32))
        got = np.asarray(
            field_absorb(
                state,
                blocks=4,
                absorb=lambda s, i: s + i.astype(fnp.int32),
                permute=lambda s: s,
            )
        )
        np.testing.assert_array_equal(got, np.full(3, 0 + 1 + 2 + 3, dtype=np.int32))

    def test_a_zero_block_count_leaves_the_state(self) -> None:
        state = fnp.asarray(np.arange(3, dtype=np.int32))
        np.testing.assert_array_equal(
            np.asarray(
                field_absorb(
                    state,
                    blocks=0,
                    absorb=lambda s, i: s + fnp.int32(1),
                    permute=lambda s: s,
                )
            ),
            np.arange(3, dtype=np.int32),
        )

    def test_it_survives_tracing(self) -> None:
        # The loop is a `while_loop` precisely so it lowers under `jit`; an
        # eager-only case would not see a control-flow mistake.
        def run(x: fnp.ndarray) -> fnp.ndarray:
            return field_absorb(
                x,
                blocks=2,
                absorb=lambda s, i: s + fnp.int32(1),
                permute=lambda s: s * fnp.int32(3),
            )

        x = fnp.asarray(np.zeros(2, dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(frx.jit(run)(x)), np.asarray(run(x)))


if __name__ == "__main__":
    absltest.main()

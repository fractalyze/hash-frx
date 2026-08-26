# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`masked_chain`'s two walks, against a fake compression rather than a family.

The families pin their own digests against published vectors, and those cases
do exercise this walk — but only end to end, where a mask that were off by one
block is indistinguishable from a padding rule that were off by one block. What
belongs here is the claim the walk makes on its own: that `live` selects a
prefix, and that selecting all of it is the same walk as not masking at all.

That second half is load-bearing rather than decorative. `chain` — the static
walk every blocks-in family runs — routes through this function at `live=None`,
so "the masked walk at full length IS the static walk" is what lets one function
serve both. A fake compression is what makes it checkable: `s + b` composes to a
prefix sum, so the expected midstate is arithmetic rather than a second
transcription of the thing under test (the `sponge_test` arrangement).
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.extension.md import masked_chain

_WIDTH = 4
_NBLOCKS = 5


def _blocks(batch: int) -> np.ndarray:
    """Distinct per (row, block, lane), so a walk that dropped a block or took
    one twice cannot land on the right sum by symmetry."""
    return (1 + np.arange(batch * _NBLOCKS * _WIDTH, dtype=np.uint32)).reshape(
        batch, _NBLOCKS, _WIDTH
    )


def _fold(state: frx.Array, block: frx.Array) -> frx.Array:
    """A stand-in compression: associative, order-insensitive, and cheap to
    predict — folding a prefix is that prefix's sum."""
    return state + block


def _expected(h0: np.ndarray, blocks: np.ndarray, live: int) -> np.ndarray:
    return h0 + blocks[:, :live].sum(axis=1, dtype=np.uint32)


class MaskedChainTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("none_live", 0),
        ("one", 1),
        ("all_but_one", _NBLOCKS - 1),
        ("all", _NBLOCKS),
    )
    def test_live_folds_exactly_that_many_blocks(self, live: int) -> None:
        """The mask selects a prefix — and it is read as runtime data, under a
        `jit` whose block count is static while `live` is not."""
        h0 = np.arange(_WIDTH, dtype=np.uint32)
        blocks = _blocks(batch=3)

        walk = frx.jit(
            lambda h, b, ln: masked_chain(h, b, compress_block=_fold, live=ln)
        )
        got = walk(fnp.asarray(h0), fnp.asarray(blocks), np.int32(live))

        np.testing.assert_array_equal(np.asarray(got), _expected(h0, blocks, live))

    def test_masking_at_full_length_is_the_unmasked_walk(self) -> None:
        """`live=None` and `live=nblocks` are the same walk, which is what lets
        `chain` route its static walk through this function."""
        h0 = np.arange(_WIDTH, dtype=np.uint32)
        blocks = _blocks(batch=2)

        static = masked_chain(
            fnp.asarray(h0), fnp.asarray(blocks), compress_block=_fold
        )
        masked = masked_chain(
            fnp.asarray(h0),
            fnp.asarray(blocks),
            compress_block=_fold,
            live=np.int32(_NBLOCKS),
        )

        np.testing.assert_array_equal(np.asarray(static), np.asarray(masked))

    def test_a_live_count_past_the_block_axis_folds_every_block(self) -> None:
        """The loop is bounded by the compiled block count, not by `live`, so a
        `live` past the end saturates rather than reading off the end. The
        capacity ladder rounds a buffer UP, so this is the state a caller
        reaches by asking for more than the region holds."""
        h0 = np.zeros(_WIDTH, dtype=np.uint32)
        blocks = _blocks(batch=1)

        got = masked_chain(
            fnp.asarray(h0),
            fnp.asarray(blocks),
            compress_block=_fold,
            live=np.int32(_NBLOCKS + 3),
        )

        np.testing.assert_array_equal(np.asarray(got), _expected(h0, blocks, _NBLOCKS))

    def test_the_unbatched_midstate_broadcasts_across_the_batch(self) -> None:
        """`h0` arrives unbatched — one shared initial or resumed midstate — and
        every row starts from it rather than from a materialized copy."""
        h0 = np.array([7, 8, 9, 10], dtype=np.uint32)
        blocks = _blocks(batch=4)

        got = masked_chain(fnp.asarray(h0), fnp.asarray(blocks), compress_block=_fold)

        self.assertEqual(np.asarray(got).shape, (4, _WIDTH))
        np.testing.assert_array_equal(np.asarray(got), _expected(h0, blocks, _NBLOCKS))


if __name__ == "__main__":
    absltest.main()

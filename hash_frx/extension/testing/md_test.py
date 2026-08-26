# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`masked_chain`'s two walks, against a fake compression rather than a family.

The families pin their own digests against published vectors, and those cases
do exercise this walk — but only end to end, where a mask that were off by one
block is indistinguishable from a padding rule that were off by one block. What
belongs here is the claim the walk makes on its own: that `live` selects a
prefix, and that the unmasked walk is the same fold with no predicate.

A fake compression is what makes it checkable: `s + b` composes to a prefix sum,
so the expected midstate is arithmetic rather than a second transcription of the
thing under test (the `sponge_test` arrangement).
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.extension.md import masked_chain

_WIDTH = 4
_NBLOCKS = 5
_BATCH = 3

# Distinct per (row, block, lane), so a walk that dropped a block or took one
# twice cannot land on the right sum by symmetry.
_STATE0 = np.tile(np.arange(_WIDTH, dtype=np.uint32), (_BATCH, 1))
_BLOCKS = (1 + np.arange(_BATCH * _NBLOCKS * _WIDTH, dtype=np.uint32)).reshape(
    _BATCH, _NBLOCKS, _WIDTH
)


def _fold(state: frx.Array, i: int) -> frx.Array:
    """A stand-in compression: the caller indexes, as every real one does."""
    return state + fnp.asarray(_BLOCKS)[:, i]


def _expected(live: int) -> np.ndarray:
    return _STATE0 + _BLOCKS[:, :live].sum(axis=1, dtype=np.uint32)


# One jit zone, so the four masked cases share a trace instead of compiling the
# same walk once each — they differ only in `live`, which is an operand.
_WALK = frx.jit(
    lambda s, ln: masked_chain(s, count=_NBLOCKS, compress_block=_fold, live=ln)
)


class MaskedChainTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("zero", 0),
        ("one", 1),
        ("all_but_one", _NBLOCKS - 1),
        ("all", _NBLOCKS),
    )
    def test_live_folds_exactly_that_many_blocks(self, live: int) -> None:
        """The mask selects a prefix — and it is read as runtime data, under a
        `jit` whose block count is static while `live` is not."""
        got = _WALK(fnp.asarray(_STATE0), np.int32(live))

        np.testing.assert_array_equal(np.asarray(got), _expected(live))

    def test_a_live_count_past_the_block_axis_folds_every_block(self) -> None:
        """The loop is bounded by the compiled block count, not by `live`, so a
        `live` past the end saturates rather than reading off the end. The
        capacity ladder rounds a buffer UP, so this is the state a caller
        reaches by asking for more than the region holds."""
        got = _WALK(fnp.asarray(_STATE0), np.int32(_NBLOCKS + 3))

        np.testing.assert_array_equal(np.asarray(got), _expected(_NBLOCKS))

    def test_the_unmasked_walk_folds_every_block(self) -> None:
        """`live=None` — the walk `chain` routes through, which emits no
        predicate at all and so must still reach the same midstate the masked
        walk reaches at full length."""
        got = masked_chain(fnp.asarray(_STATE0), count=_NBLOCKS, compress_block=_fold)

        np.testing.assert_array_equal(np.asarray(got), _expected(_NBLOCKS))


if __name__ == "__main__":
    absltest.main()

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Poseidon's normal-form linear layers: fusion-ready, and each reading its
chained input a bounded number of times.

The rule and the 22-round permutation that produced it are in
`hash_frx.linear`; the limits below are its per-layer instances.
"""

import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import babybear_mont as F

from hash_frx.poseidon.linear import apply_sparse_partial
from hash_frx.testing.fusion_ready import (
    assert_fusion_ready,
    assert_input_uses,
    primitive_count,
)
from hash_frx.testing.random_field import rand_field

_WIDTHS = (8, 16)


def _sparse_args(w: int) -> tuple:
    """`(dot_row, col_vec, active, tail)` for a width-`w` partial round."""
    return (
        rand_field(1, (w,), F),
        rand_field(2, (w - 1,), F),
        rand_field(3, (), F),
        rand_field(4, (w - 1,), F),
    )


class SparsePartialTest(absltest.TestCase):
    def test_equals_the_documented_normal_form(self) -> None:
        for w in _WIDTHS:
            dot_row, col_vec, active, tail = _sparse_args(w)
            got = apply_sparse_partial(dot_row, col_vec, active, tail)
            # Independent in form, not a transcription of the unrolled fold: a
            # test reference carries no fusion constraint, so it may reduce.
            want0 = dot_row[0] * active + fnp.sum(dot_row[1:] * tail)
            want = fnp.concatenate([want0[None], tail + col_vec * active])
            self.assertTrue(bool(fnp.array_equal(got, want)), f"width {w}")

    def test_reads_the_shared_lane_value_twice(self) -> None:
        # `active` is the post-S-box lane-0 value every lane consumes: once in
        # the lane-0 dot, once in the rank-1 update. A per-lane form reads it
        # w-1 times, and each read re-derives the whole preceding round.
        for w in _WIDTHS:
            assert_input_uses(apply_sparse_partial, *_sparse_args(w), arg=2, limit=2)

    def test_multiplies_do_not_scale_with_width(self) -> None:
        # The same rule seen through the op count: 2w-1 multiplies per-lane
        # versus a constant few in array form. Sharper than the read count here,
        # and it fails naming the width that drifted.
        counts = {
            w: primitive_count(apply_sparse_partial, *_sparse_args(w), name="mul")
            for w in (8, 12, 16)
        }
        self.assertEqual(
            len(set(counts.values())),
            1,
            msg=(
                f"multiplies must not scale with width, got {counts}. A per-lane "
                f"scalar form gives 2w-1; see apply_sparse_partial."
            ),
        )

    def test_is_fusion_ready(self) -> None:
        assert_fusion_ready(apply_sparse_partial, *_sparse_args(16))


if __name__ == "__main__":
    absltest.main()

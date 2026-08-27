# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The normal-form linear helpers compute the right thing, and stay fusible.

Two obligations, and the second is the one that rots silently: `unrolled_sum` and
`apply_matrix` must agree with the reductions they deliberately avoid, *and* must
not lower to one. A rewrite to `fnp.sum` would keep every value test green while
splitting the round body's kernel at the `kInput` fusion boundary — so the
lowering is asserted, not just the arithmetic.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from hash_frx.linear import apply_matrix, unrolled_sum
from hash_frx.testing.fusion_ready import assert_fusion_ready
from hash_frx.testing.random_field import rand_field


class UnrolledSumTest(absltest.TestCase):
    def test_matches_the_reduction_it_avoids(self) -> None:
        terms = [rand_field(i, (4,), F) for i in range(5)]
        want = terms[0]
        for t in terms[1:]:
            want = want + t
        self.assertTrue(bool(fnp.array_equal(unrolled_sum(terms), want)))

    def test_single_term_is_the_term(self) -> None:
        x = rand_field(1, (4,), F)
        self.assertTrue(bool(fnp.array_equal(unrolled_sum([x]), x)))

    def test_pairs_the_terms_rather_than_folding_them_left(self) -> None:
        # Both spellings cost the same adds; what pairing buys is the critical
        # path — one add per doubling where a left fold makes each add wait for
        # the last. A round sums the whole state once, so that depth is what it
        # waits on, and the win shows up wherever too few warps are resident to
        # hide the latency.
        n = 16
        terms = [rand_field(i, (4,), F) for i in range(n)]
        jaxpr = frx.make_jaxpr(lambda *t: unrolled_sum(list(t)))(*terms).jaxpr
        depth = {v: 0 for v in jaxpr.invars}
        longest = 0
        for eqn in jaxpr.eqns:
            d = 1 + max((depth.get(v, 0) for v in eqn.invars), default=0)
            for out in eqn.outvars:
                depth[out] = d
            longest = max(longest, d)
        self.assertEqual(longest, 4, "16 terms pair down in 4 rounds, not 15")
        adds = sum(1 for e in jaxpr.eqns if e.primitive.name == "add")
        self.assertEqual(adds, n - 1, "pairing must not cost an extra add")

    def test_lowers_without_a_boundary_op(self) -> None:
        terms = [rand_field(i, (4,), F) for i in range(4)]
        assert_fusion_ready(lambda *t: unrolled_sum(list(t)), *terms, reduces=0)


class ApplyMatrixTest(absltest.TestCase):
    def test_matches_the_matmul_it_avoids(self) -> None:
        w = 4
        m = rand_field(11, (w, w), F)
        s = rand_field(12, (w,), F)
        want = unrolled_sum([m[:, j] * s[j] for j in range(w)])
        self.assertTrue(bool(fnp.array_equal(apply_matrix(m, s), want)))

    def test_identity_matrix_is_the_state(self) -> None:
        w = 4
        eye = fnp.array([[int(i == j) for j in range(w)] for i in range(w)], dtype=F)
        s = rand_field(13, (w,), F)
        self.assertTrue(bool(fnp.array_equal(apply_matrix(eye, s), s)))

    def test_lowers_without_a_boundary_op(self) -> None:
        w = 4
        m = rand_field(14, (w, w), F)
        s = rand_field(15, (w,), F)
        assert_fusion_ready(apply_matrix, m, s, reduces=0)

    def test_rejects_a_non_1d_state(self) -> None:
        m = rand_field(16, (4, 4), F)
        with self.assertRaisesRegex(ValueError, "state must be 1-D"):
            apply_matrix(m, rand_field(17, (4, 1), F))

    def test_rejects_a_shape_mismatched_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "square matrix"):
            apply_matrix(rand_field(18, (3, 4), F), rand_field(19, (4,), F))


if __name__ == "__main__":
    absltest.main()

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The affine layer equals the oracle polynomial and stays fusion-ready."""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array
from zk_dtypes import binary_field_t5 as F

from hash_frx.testing.fusion_ready import assert_fusion_ready, assert_input_uses
from hash_frx.vision.linear import apply_linearized_affine, apply_matrix
from hash_frx.vision.params import vision_mark32_params
from hash_frx.vision.testing.decode import ints
from hash_frx.vision.testing.reference import evaluate_affine


def _rand(seed: int, shape: tuple[int, ...]) -> Array:
    # Full-width GF(2^32) draws: `random_field.rand_field` caps at 2**30 (a
    # prime-field bound), which would never exercise the top tower level.
    ints = np.random.default_rng(seed).integers(0, 1 << 32, size=shape)
    return fnp.array(ints, dtype=F)


_P = vision_mark32_params(F)


class AffineLayerTest(absltest.TestCase):
    def test_matches_the_oracle_polynomial(self) -> None:
        # Both shipped polynomials: the 4-entry B exercises the short chain,
        # the 33-entry B_inv every squaring up to x**(2**31).
        s = _rand(1, (24,))
        lanes = ints(s)
        for name, coeffs in (("b", _P.b), ("b_inv", _P.b_inv)):
            with self.subTest(poly=name):
                got = apply_linearized_affine(coeffs, s)
                want = [evaluate_affine(ints(coeffs), x) for x in lanes]
                self.assertEqual(ints(got), want)

    def test_rejects_bad_coefficients(self) -> None:
        s = _rand(2, (24,))
        with self.assertRaises(ValueError):
            apply_linearized_affine(fnp.array([1], dtype=F), s)  # constant only
        with self.assertRaises(ValueError):
            apply_linearized_affine(fnp.ones((2, 2), dtype=F), s)  # 2-D

    def test_normal_form_is_fusion_ready(self) -> None:
        s = _rand(3, (24,))
        # Element-wise only — no reduce/dot/gather boundary (whitelist gate).
        assert_fusion_ready(lambda v: apply_linearized_affine(_P.b, v), s, reduces=0)
        assert_fusion_ready(
            lambda v: apply_linearized_affine(_P.b_inv, v), s, reduces=0
        )
        assert_fusion_ready(lambda v: apply_matrix(_P.mds, v), s, reduces=0)
        # The matrix form reduces, so the gate must bite — else the check is
        # vacuous.
        with self.assertRaises(AssertionError):
            assert_fusion_ready(lambda v: _P.mds @ v, s, reduces=0)

    def test_state_reads_stay_constant(self) -> None:
        # The chained-input rule in `hash_frx.linear`: the state feeds only the
        # first coefficient term and the first squaring (whose two factors are
        # the state itself), so THREE reads no matter the polynomial length or
        # width — the layers chain 16 times per permute, so a width-scaled read
        # count would compound per step rather than add. b_inv is the sharp
        # case: 33 coefficients, still 3.
        s = _rand(4, (24,))
        assert_input_uses(lambda v: apply_linearized_affine(_P.b_inv, v), s, limit=3)
        assert_input_uses(lambda v: apply_matrix(_P.mds, v), s, limit=24)

        # A per-lane spelling re-reads the state once per lane; the counter
        # must bite on it or the limit above proves nothing.
        def per_lane(v: Array) -> Array:
            return fnp.stack([apply_linearized_affine(_P.b, v[i]) for i in range(24)])

        with self.assertRaises(AssertionError):
            assert_input_uses(per_lane, s, limit=3)


if __name__ == "__main__":
    absltest.main()

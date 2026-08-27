# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""VisionParams: fully-free surface + fail-loud validation."""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import binary_field_t5 as F

from hash_frx.vision.params import VisionParams, vision_mark32_params


def _good(**over: Any) -> VisionParams:
    w, n_rounds = 4, 2
    base = dict(
        width=w,
        dtype=F,
        rounds=n_rounds,
        b=fnp.array([1, 2, 3, 4], dtype=F),
        b_inv=fnp.array([5, 6, 7, 8, 9], dtype=F),
        mds=fnp.ones((w, w), dtype=F),
        round_keys=fnp.zeros((2 * n_rounds + 1, w), dtype=F),
    )
    base.update(over)
    return VisionParams(**base)


class VisionParamsTest(absltest.TestCase):
    def test_bad_round_key_shape_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            _good(round_keys=fnp.zeros((4, 4), dtype=F))  # 2N+1 = 5 rows
        self.assertIn("round_keys", str(cm.exception))

    def test_bad_mds_shape_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            _good(mds=fnp.ones((4, 5), dtype=F))
        self.assertIn("mds", str(cm.exception))

    def test_a_bare_constant_polynomial_raises(self) -> None:
        # Length 1 is the affine constant with no linear part — not a
        # permutation, so not a B.
        with self.assertRaises(ValueError) as cm:
            _good(b=fnp.array([1], dtype=F))
        self.assertIn("b", str(cm.exception))

    def test_a_2d_polynomial_raises(self) -> None:
        with self.assertRaises(ValueError):
            _good(b_inv=fnp.ones((2, 2), dtype=F))

    def test_nonpositive_rounds_raises(self) -> None:
        with self.assertRaises(ValueError):
            _good(rounds=0, round_keys=fnp.zeros((1, 4), dtype=F))

    def test_mismatched_table_dtype_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            _good(mds=fnp.ones((4, 4), dtype=fnp.uint32))
        self.assertIn("dtype", str(cm.exception))


class VisionParamsValueEqualityTest(absltest.TestCase):
    """Params compare by value: the permutation rides pytree aux (meta_fields),
    so independently built equal params must be == and hash-equal — identity
    equality re-traces every jit zone that carries one as pytree aux."""

    def test_equal_by_value_across_instances(self) -> None:
        a, b = _good(), _good()
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_differs_on_scalar_field(self) -> None:
        deeper = _good(rounds=3, round_keys=fnp.zeros((7, 4), dtype=F))
        self.assertNotEqual(_good(), deeper)

    def test_differs_on_constant_arrays(self) -> None:
        self.assertNotEqual(_good(), _good(b=fnp.array([9, 2, 3, 4], dtype=F)))


class VisionMark32FactoryTest(absltest.TestCase):
    def test_the_shipped_instance_shape(self) -> None:
        # The Vision Mark-32 coordinates (https://eprint.iacr.org/2024/633,
        # Section 3): m = 24 over GF(2^32), N = 8; B of degree 4 (4 entries),
        # B^{-1} dense (n + 1 = 33 entries).
        p = vision_mark32_params(F)
        self.assertEqual(p.width, 24)
        self.assertEqual(p.rounds, 8)
        self.assertEqual(p.dtype, F)
        self.assertEqual(tuple(p.b.shape), (4,))
        self.assertEqual(tuple(p.b_inv.shape), (33,))
        self.assertEqual(tuple(p.mds.shape), (24, 24))
        self.assertEqual(tuple(p.round_keys.shape), (17, 24))

    def test_two_factory_calls_compare_equal(self) -> None:
        self.assertEqual(vision_mark32_params(F), vision_mark32_params(F))


if __name__ == "__main__":
    absltest.main()

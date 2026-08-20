# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""RescueParams: fully-free surface + fail-loud validation + the alpha pins."""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from zk_dtypes import goldilocks_mont as F

from hash_frx.rescue.params import RescueParams, rescue_rpo128_params


def _good(**over: Any) -> RescueParams:
    w, n_rounds = 4, 2
    base = dict(
        width=w,
        dtype=F,
        rounds=n_rounds,
        alpha=3,
        inv_alpha=5,
        mds=fnp.ones((w, w), dtype=F),
        round_constants=fnp.zeros((2 * n_rounds, w), dtype=F),
    )
    base.update(over)
    return RescueParams(**base)


class RescueParamsTest(absltest.TestCase):
    def test_bad_round_constant_shape_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            _good(round_constants=fnp.zeros((3, 4), dtype=F))  # 2N = 4 rows
        self.assertIn("round_constants", str(cm.exception))

    def test_bad_mds_shape_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            _good(mds=fnp.ones((4, 5), dtype=F))
        self.assertIn("mds", str(cm.exception))

    def test_nonpositive_rounds_raises(self) -> None:
        with self.assertRaises(ValueError):
            _good(rounds=0, round_constants=fnp.zeros((0, 4), dtype=F))

    def test_nonpositive_alpha_raises(self) -> None:
        with self.assertRaises(ValueError):
            _good(alpha=0)

    def test_nonpositive_inv_alpha_raises(self) -> None:
        with self.assertRaises(ValueError):
            _good(inv_alpha=0)

    def test_mismatched_table_dtype_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            _good(mds=fnp.ones((4, 4), dtype=fnp.uint64))
        self.assertIn("dtype", str(cm.exception))


class RescueParamsValueEqualityTest(absltest.TestCase):
    """Params compare by value: the permutation rides pytree aux (meta_fields),
    so independently built equal params must be == and hash-equal — identity
    equality re-traces every jit zone that carries one as pytree aux."""

    def test_equal_by_value_across_instances(self) -> None:
        a, b = _good(), _good()
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_differs_on_scalar_field(self) -> None:
        # inv_alpha is the sharp scalar: stored rather than derived, so only
        # its seat in the value key keeps two instances differing in nothing
        # else from sharing a trace.
        self.assertNotEqual(_good(), _good(inv_alpha=7))

    def test_differs_on_constant_arrays(self) -> None:
        other = _good(round_constants=fnp.ones((4, 4), dtype=F))
        self.assertNotEqual(_good(), other)


class RescueRpo128FactoryTest(absltest.TestCase):
    def test_the_shipped_instance_shape(self) -> None:
        # The RPO-128 coordinates (https://eprint.iacr.org/2022/1577,
        # Section 2.1, Table 1): m = 12 over Goldilocks, N = 7, alpha = 7;
        # constants one row per half-round.
        p = rescue_rpo128_params(F)
        self.assertEqual(p.width, 12)
        self.assertEqual(p.rounds, 7)
        self.assertEqual(p.alpha, 7)
        self.assertEqual(p.dtype, F)
        self.assertEqual(tuple(p.mds.shape), (12, 12))
        self.assertEqual(tuple(p.round_constants.shape), (14, 12))

    def test_inv_alpha_is_the_inverse_of_alpha(self) -> None:
        # inv_alpha is DATA on the parameter surface (the core cannot read p);
        # that it really is alpha^-1 mod (p - 1) — so the two power maps
        # compose to the identity — is only pinned here, where the field size
        # may be read off the dtype.
        p = rescue_rpo128_params(F)
        modulus = zk_dtypes.pfinfo(F).modulus
        self.assertEqual(p.inv_alpha, pow(p.alpha, -1, modulus - 1))

    def test_two_factory_calls_compare_equal(self) -> None:
        self.assertEqual(rescue_rpo128_params(F), rescue_rpo128_params(F))


if __name__ == "__main__":
    absltest.main()

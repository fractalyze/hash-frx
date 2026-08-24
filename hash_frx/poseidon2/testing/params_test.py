"""Poseidon2Params: fully-free surface + fail-loud validation."""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from hash_frx.poseidon2.params import (
    Poseidon2Params,
    default_external_matrix,
)


def _good(**over: Any) -> Poseidon2Params:
    w, er, ir = 16, 4, 20
    base = dict(
        width=w,
        dtype=F,
        alpha=3,
        external_rounds=er,
        internal_rounds=ir,
        external_constants_initial=fnp.zeros((er, w), dtype=F),
        external_constants_terminal=fnp.zeros((er, w), dtype=F),
        internal_constants=fnp.zeros((ir,), dtype=F),
        internal_diag=fnp.ones((w,), dtype=F),
    )
    base.update(over)
    return Poseidon2Params(**base)


class Poseidon2ParamsTest(absltest.TestCase):
    def test_external_matrix_defaults_to_canonical(self) -> None:
        p = _good()
        ext = p.external_matrix
        if ext is None:
            raise AssertionError("external_matrix should default to canonical")
        self.assertEqual(ext.shape, (16, 16))
        self.assertEqual(ext.dtype, F)

    def test_bad_rc_shape_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            _good(internal_constants=fnp.zeros((19,), dtype=F))  # wrong round count
        self.assertIn("internal_constants", str(cm.exception))

    def test_internal_constants_are_one_per_round(self) -> None:
        """One constant per partial round, not a width-wide row.

        The partial round S-boxes lane 0 alone, so only lane 0's constant is
        ever read. Carrying `(internal_rounds, width)` made every caller build
        a zero matrix and fill column 0 for `poseidon2.py` to slice back out —
        a round trip with no information in it, and a zero-lane validation to
        police the padding it introduced.
        """
        self.assertEqual(_good().internal_constants.shape, (20,))

    def test_width_wide_internal_constants_raise(self) -> None:
        """The old `(internal_rounds, width)` spelling is refused rather than
        silently sliced: a caller still passing the padded matrix gets told,
        instead of having lanes 1..w-1 quietly discarded."""
        with self.assertRaises(ValueError) as cm:
            _good(internal_constants=fnp.zeros((20, 16), dtype=F))
        self.assertIn("internal_constants", str(cm.exception))


class Poseidon2ParamsValueEqualityTest(absltest.TestCase):
    """Params compare by value: the permutation rides pytree aux (meta_fields),
    so independently built equal params must be == and hash-equal — identity
    equality re-traces every jit zone that carries one as pytree aux."""

    def test_equal_by_value_across_instances(self) -> None:
        a, b = _good(), _good()
        self.assertIsNot(a, b)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_differs_on_scalar_field(self) -> None:
        self.assertNotEqual(_good(), _good(alpha=5))

    def test_differs_on_constant_arrays(self) -> None:
        ones = fnp.ones((4, 16), dtype=F)
        self.assertNotEqual(_good(), _good(external_constants_initial=ones))


class DtypeSpellingTest(absltest.TestCase):
    """`F` and `np.dtype(F)` name one dtype but hash differently (#215).

    They compare equal, so equality alone never caught this: as jit cache keys
    they are two entries, and a consumer that spelled the dtype the other way
    silently re-traced every call. `dtype` is normalized to the scalar type,
    which is also the spelling that stays callable for the zero comparisons in
    `__post_init__` — `np.dtype(F)(0)` raises.
    """

    def test_both_spellings_normalize_to_one_key(self) -> None:
        scalar = _good()
        wrapped = _good(dtype=np.dtype(F))
        self.assertIs(wrapped.dtype, F)
        self.assertEqual(scalar, wrapped)
        self.assertEqual(hash(scalar), hash(wrapped))


class DefaultExternalMatrixWidthTest(absltest.TestCase):
    """The width-4 default is refused, not served (#215): the formula
    degenerates to `2 * M4` there, where canonical Poseidon2 is plain `M4`, so
    a default would hand back a different permutation than the references."""

    def test_width_four_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "width >= 8"):
            default_external_matrix(4, F)

    def test_width_eight_still_builds(self) -> None:
        self.assertEqual(default_external_matrix(8, F).shape, (8, 8))


if __name__ == "__main__":
    absltest.main()

"""Normal-form linear layers equal the matrix form and stay fusion-ready."""

import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from hash_frx.poseidon2.linear import (
    apply_external_m4,
    apply_internal,
    apply_matrix,
)
from hash_frx.poseidon2.params import default_external_matrix
from hash_frx.testing.fusion_ready import (
    assert_fusion_ready,
    assert_input_uses,
    primitive_count,
)
from hash_frx.testing.random_field import rand_field

# Plonky3's M4 = circ(2,3,1,1) — the base M4 `default_external_matrix` builds from.
_STD_M4 = ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))


class LinearLayerTest(absltest.TestCase):
    def test_apply_matrix_equals_matmul(self) -> None:
        w = 16
        m = rand_field(1, (w, w), F)
        s = rand_field(2, (w,), F)
        self.assertTrue(bool(fnp.array_equal(apply_matrix(m, s), m @ s)))

    def test_apply_internal_equals_jdiag(self) -> None:
        w = 16
        d = rand_field(3, (w,), F)
        s = rand_field(4, (w,), F)
        m_int = fnp.ones((w, w), dtype=F) + fnp.diag(d)
        self.assertTrue(bool(fnp.array_equal(apply_internal(d, s), m_int @ s)))

    def test_apply_external_m4_equals_default_matrix(self) -> None:
        # The literal-coefficient external layer must byte-match the array form
        # apply_matrix(default_external_matrix(w)) it stands in for.
        for w in (8, 16):
            m = default_external_matrix(w, F)
            s = rand_field(7, (w,), F)
            self.assertTrue(
                bool(
                    fnp.array_equal(apply_external_m4(s, _STD_M4), apply_matrix(m, s))
                ),
                f"width {w}",
            )

    def test_external_multiplies_scale_with_width_not_its_square(self) -> None:
        # The block structure is the whole reason the layer is affordable: the
        # dense column sum spends a multiply per (output, lane) pair, so w=16
        # costs 256 against the 4-per-lane the M4 images cost. Read as a ratio,
        # because literal 1 coefficients fold and the absolute count is the
        # base M4's to decide.
        counts = {
            w: primitive_count(
                lambda v: apply_external_m4(v, _STD_M4),
                rand_field(9, (w,), F),
                name="mul",
            )
            for w in (8, 16)
        }
        self.assertLessEqual(
            counts[16],
            2 * counts[8] + 1,
            msg=(
                f"external multiplies must grow with width, not its square, got "
                f"{counts}. A dense column sum gives w**2; see apply_external_m4."
            ),
        )

    def test_apply_external_m4_rejects_bad_width(self) -> None:
        with self.assertRaises(ValueError):
            apply_external_m4(rand_field(2, (6,), F), _STD_M4)  # 6 % 4 != 0
        with self.assertRaises(ValueError):
            apply_external_m4(
                rand_field(2, (16, 16), F), _STD_M4
            )  # 2-D, not a lane vector

    def test_apply_matrix_rejects_mismatched_state(self) -> None:
        w = 16
        m = rand_field(1, (w, w), F)
        with self.assertRaises(ValueError):
            apply_matrix(m, rand_field(2, (w, w), F))  # 2-D, not a lane vector
        with self.assertRaises(ValueError):
            apply_matrix(m, rand_field(2, (w + 1,), F))  # length mismatches matrix

    def test_apply_internal_rejects_mismatched_state(self) -> None:
        w = 16
        d = rand_field(3, (w,), F)
        with self.assertRaises(ValueError):
            apply_internal(d, rand_field(4, (w + 1,), F))  # length mismatches diag

    def test_normal_form_is_fusion_ready(self) -> None:
        w = 16
        m = rand_field(1, (w, w), F)
        d = rand_field(3, (w,), F)
        s = rand_field(2, (w,), F)
        # Element-wise only — no reduce/dot/gather boundary (whitelist gate).
        assert_fusion_ready(lambda v: apply_matrix(m, v), s, reduces=0)
        assert_fusion_ready(lambda v: apply_internal(d, v), s, reduces=0)
        assert_fusion_ready(lambda v: apply_external_m4(v, _STD_M4), s, reduces=0)
        # The matrix form reduces, so the gate must bite — else the check is vacuous.
        with self.assertRaises(AssertionError):
            assert_fusion_ready(lambda v: m @ v, s, reduces=0)

    def test_state_reads_stay_linear_in_width(self) -> None:
        # The chained-input rule in `hash_frx.linear`: reading `state` inside
        # both loops of a w*w layer costs w**2 reads, one hoist costs w.
        for w in (8, 16):
            m = rand_field(1, (w, w), F)
            d = rand_field(3, (w,), F)
            s = rand_field(2, (w,), F)
            assert_input_uses(lambda v: apply_matrix(m, v), s, limit=w)
            assert_input_uses(lambda v: apply_external_m4(v, _STD_M4), s, limit=w)
            # The internal layer also sums every lane, hence the one extra.
            assert_input_uses(lambda v: apply_internal(d, v), s, limit=w + 1)


if __name__ == "__main__":
    absltest.main()

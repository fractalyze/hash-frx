"""Poseidon2 implements Permutation and preserves shape/dtype."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import frx
import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from hash_frx.permutation import Permutation
from hash_frx.poseidon2.params import Poseidon2Params
from hash_frx.poseidon2.poseidon2 import Poseidon2


def _params() -> Poseidon2Params:
    w, er, ir = 16, 4, 20
    return Poseidon2Params(
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


def _marker_body_size(fn: Callable[..., Any], *args: Any) -> tuple[int, int]:
    """The permute marker's body as (equation count, scan count).

    The body is a `composite`, so counting primitives in the traced graph itself
    sees only the marker and reads zero whatever the body holds — which is how a
    per-round-copy check passes without checking anything.
    """
    jaxpr = frx.make_jaxpr(fn)(*args).jaxpr
    composites = [e for e in jaxpr.eqns if e.primitive.name == "composite"]
    assert len(composites) == 1, f"expected one marker, got {len(composites)}"
    body = composites[0].params["jaxpr"]
    body = body.jaxpr if hasattr(body, "jaxpr") else body
    scans = sum(1 for e in body.eqns if e.primitive.name == "scan")
    return len(body.eqns), scans


class Poseidon2InternalJScaleTest(absltest.TestCase):
    """`internal_j_scale` generalizes the internal matrix to
    c*J + Diag(internal_diag); the default must stay byte-identical to the
    historical J + Diag form."""

    def test_explicit_one_equals_default(self) -> None:
        base = _params()
        scaled = dataclasses.replace(base, internal_j_scale=fnp.ones((), dtype=F))
        x = fnp.arange(16, dtype=fnp.uint32).view(F)
        self.assertTrue(
            bool(fnp.all(Poseidon2(base).permute(x) == Poseidon2(scaled).permute(x)))
        )

    def test_non_unit_scale_changes_output(self) -> None:
        base = _params()
        scaled = dataclasses.replace(base, internal_j_scale=fnp.full((), 2, dtype=F))
        x = fnp.arange(16, dtype=fnp.uint32).view(F)
        self.assertFalse(
            bool(fnp.all(Poseidon2(base).permute(x) == Poseidon2(scaled).permute(x)))
        )


class Poseidon2PermuteShapeTest(absltest.TestCase):
    def test_is_a_permutation(self) -> None:
        p = Poseidon2(_params())
        self.assertIsInstance(p, Permutation)
        self.assertEqual(p.width, 16)
        self.assertEqual(p.dtype, F)

    def test_permute_shape_and_vmap(self) -> None:
        p = Poseidon2(_params())
        x = fnp.arange(16, dtype=F)
        out = p.permute(x)
        self.assertEqual(out.shape, (16,))
        self.assertEqual(out.dtype, F)
        batch = fnp.stack([x, x + F(1)])
        bout = frx.vmap(p.permute)(batch)  # thread-per-hash
        self.assertEqual(bout.shape, (2, 16))
        self.assertEqual(bout.dtype, F)
        self.assertTrue(bool(fnp.array_equal(bout[0], out)))

    def test_custom_external_matrix_is_applied(self) -> None:
        base = _params()
        ext = base.external_matrix
        if ext is None:
            raise AssertionError("external_matrix should default to canonical")
        custom = ext.at[0, 0].add(F(1))  # a different valid MDS-shaped matrix
        over = Poseidon2(Poseidon2Params(**{**vars(base), "external_matrix": custom}))
        x = fnp.arange(16, dtype=F)
        # external_matrix is an operand (external_matrix @ state), so a different
        # matrix produces a different permutation — the override is genuinely used.
        self.assertFalse(
            bool(fnp.array_equal(over.permute(x), Poseidon2(base).permute(x)))
        )

    def test_rounds_are_a_loop_not_a_copy_per_round(self) -> None:
        # The marker body must hold one round of each kind rather than one copy
        # per round: that is what lets it inline to a `while` the generic
        # static_while path lowers in a single kernel. Doubling both round
        # counts is the test — an unrolled body grows with them, a scanned one
        # does not move at all.
        base = _params()
        wide = dataclasses.replace(
            base,
            external_constants_initial=fnp.concatenate(
                [base.external_constants_initial] * 2
            ),
            external_constants_terminal=fnp.concatenate(
                [base.external_constants_terminal] * 2
            ),
            internal_constants=fnp.concatenate([base.internal_constants] * 2),
            external_rounds=base.external_rounds * 2,
            internal_rounds=base.internal_rounds * 2,
        )
        state = fnp.zeros((base.width,), dtype=F)
        sizes = {
            name: _marker_body_size(Poseidon2(p).permute, state)
            for name, p in (("base", base), ("double", wide))
        }
        self.assertEqual(
            sizes["base"],
            sizes["double"],
            msg=(
                f"doubling the round counts must not grow the marker body, got "
                f"{sizes} — the rounds are unrolled, not scanned."
            ),
        )

    def test_permute_rejects_wrong_shape(self) -> None:
        p = Poseidon2(_params())
        with self.assertRaises(ValueError):
            p.permute(fnp.zeros((15,), dtype=F))  # width != 16
        with self.assertRaises(ValueError):
            p.permute(fnp.zeros((2, 16), dtype=F))  # batched, not a 1-D state


if __name__ == "__main__":
    absltest.main()

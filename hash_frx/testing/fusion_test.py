# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""fused_region runs its decomposition and emits one zorch.fused_region composite,
and `fused_region_over` does the same for a whole computation over a permutation."""

from collections.abc import Callable
from typing import Any

import frx
import frx.numpy as fnp
from absl.testing import absltest
from frx import Array
from zk_dtypes import koalabear_mont as F

from hash_frx.fusion import FUSED_REGION_MARKER, fused_region, fused_region_over
from hash_frx.testing.random_field import rand_field

# The stub permutation's single ABI constant — one value, so two permutes inside
# a region still share one operand.
_SHIFT = fnp.ones(8, dtype=F)


class FusedRegionTest(absltest.TestCase):
    def test_runs_the_decomposition(self) -> None:
        s0 = rand_field(1, (8,), F)
        decomp = lambda s: s + s + s  # straight-line
        self.assertTrue(bool(fnp.array_equal(fused_region(decomp, s0), decomp(s0))))

    def test_runs_under_jit(self) -> None:
        # composite *lowering* under @jit must produce the decomposition's result.
        s0 = rand_field(2, (8,), F)
        decomp = lambda s: s + s + s
        out = frx.jit(lambda v: fused_region(decomp, v))(s0)
        self.assertTrue(bool(fnp.array_equal(out, decomp(s0))))

    def test_emits_one_fused_region_composite(self) -> None:
        s0 = rand_field(1, (8,), F)
        decomp = lambda s: s + s
        txt = frx.jit(lambda v: fused_region(decomp, v)).lower(s0).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        self.assertIn(FUSED_REGION_MARKER, txt)

    def test_marker_name_matches_the_xla_recognizer(self) -> None:
        # The recognizer matches by name, so the string is an ABI, not a label,
        # and an unrecognized name loses fusion silently rather than erroring.
        self.assertEqual(FUSED_REGION_MARKER, "zorch.fused_region")


class _StubPerm:
    """A permutation whose fused-region ABI needs one constant operand — enough
    to show the threading, without a real primitive's round schedule."""

    width = 8
    dtype = F
    has_dedicated_fusion = True
    fused_region_marker = ("hash_frx.stub", 1)

    def permute(self, state: Array) -> Array:
        return state + _SHIFT

    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        return (
            (leading, _SHIFT),
            (lambda state, shift: state + shift),
            {"permutation": "stub", "width": 8},
        )


class FusedRegionOverTest(absltest.TestCase):
    """The shared assembly both sponges wrap their whole hash in."""

    def test_the_body_runs_over_the_rebuilt_permute(self) -> None:
        s0 = rand_field(1, (8,), F)
        out = frx.jit(
            lambda v: fused_region_over(
                _StubPerm(),
                v,
                lambda x, permute: permute(permute(x)),
                name="hash_frx.stub_chain",
                version=1,
            )
        )(s0)
        self.assertTrue(bool(fnp.array_equal(out, s0 + _SHIFT + _SHIFT)))

    def test_the_permutation_constants_are_operands_not_captures(self) -> None:
        # The reason this helper exists rather than each construction closing
        # over a permute: an array the body materialises on the host is lifted
        # into an unnamed operand ahead of the declared ones, once per site, and
        # a permute appears once per absorbed block.
        s0 = rand_field(1, (8,), F)
        eqns = [
            e
            for e in frx.make_jaxpr(
                lambda v: fused_region_over(
                    _StubPerm(),
                    v,
                    lambda x, permute: permute(permute(x)),
                    name="hash_frx.stub_chain",
                    version=1,
                )
            )(s0).jaxpr.eqns
            if e.primitive.name == "composite"
        ]
        self.assertLen(eqns, 1)
        # Two permutes inside, still one copy of the constant they share.
        self.assertLen(eqns[0].invars, 2)

    def test_the_construction_attrs_win_over_the_permutations(self) -> None:
        # The construction owns the marker name, so on a collision it owns the
        # attribute too.
        s0 = rand_field(1, (8,), F)
        eqns = [
            e
            for e in frx.make_jaxpr(
                lambda v: fused_region_over(
                    _StubPerm(),
                    v,
                    lambda x, permute: permute(x),
                    name="hash_frx.stub_chain",
                    version=1,
                    width=99,
                    rate=4,
                )
            )(s0).jaxpr.eqns
            if e.primitive.name == "composite"
        ]
        attrs = {key: leaves[0] for key, leaves, _ in eqns[0].params["attributes"]}
        self.assertEqual(attrs, {"permutation": "stub", "width": 99, "rate": 4})


if __name__ == "__main__":
    absltest.main()

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""fused_region runs its decomposition and emits one zorch.fused_region composite."""

import frx
import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from hash_frx.fusion import FUSED_REGION_MARKER, fused_region
from hash_frx.testing.random_field import rand_field


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
        # The recognizer matches by name, so the string is an ABI, not a label.
        # It stays `zorch.` because this marker is generic and `zorch` emits it
        # too — unlike the hash-specific markers, which move to `hash_frx.`.
        self.assertEqual(FUSED_REGION_MARKER, "zorch.fused_region")


if __name__ == "__main__":
    absltest.main()

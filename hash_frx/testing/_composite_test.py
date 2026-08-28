# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""composite emits one named composite marker carrying its attrs."""

import frx.numpy as fnp
from absl.testing import absltest

from hash_frx._composite import composite
from hash_frx.testing.composite_eqn import (
    composite_attrs,
    composite_eqns,
)


class CompositeTest(absltest.TestCase):
    def test_emits_one_named_composite_carrying_attrs(self) -> None:
        # The plural plus an explicit count, where every other caller takes
        # `composite_eqn` and lets the helper assert it: "emits exactly one" is
        # the property under test HERE, so asserting it through the helper that
        # asserts it would test nothing.
        eqns = composite_eqns(
            lambda x: composite(lambda a, **_: a + a, x, name="hash_frx.t", k=3),
            fnp.arange(4),
        )
        self.assertLen(eqns, 1)
        self.assertEqual(eqns[0].params["name"], "hash_frx.t")
        attrs = composite_attrs(eqns[0])
        self.assertEqual(attrs["k"], 3)


if __name__ == "__main__":
    absltest.main()

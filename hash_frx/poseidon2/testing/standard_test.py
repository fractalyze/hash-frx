# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The shipped sets are built on first access, not at import.

`fusion.routing` requires its answer to be read per construction and never at
module scope, because a `Poseidon2` snapshots the marker its backend implies.
A module-level `KoalaBear16 = Poseidon2(...)` would take that snapshot during
`import`, before a consumer has had any chance to select a backend — and would
charge every reader of the module the array construction and a live device.

`package_test` pins the same property one level up, for the package init, but
cannot reach here: it deliberately carries no frx in its runfiles.
"""

from __future__ import annotations

from absl.testing import absltest

from hash_frx.poseidon2 import standard


def _built(name: str) -> bool:
    """Whether `name` has been constructed yet. `__getattr__` caches into
    `globals()`, so presence in the module dict is exactly that statement.
    Returns a bool rather than asserting on `vars()` directly, which would dump
    the whole module dict into the failure message."""
    return name in vars(standard)


class LazyConstructionTest(absltest.TestCase):
    def test_members_are_built_on_first_access_not_at_import(self) -> None:
        """The whole lifecycle in one test, deliberately.

        The "not built yet" half is only true before anything in the process
        has asked for the name, and the cache lives on the module rather than
        the test. So this must be the single test that touches these members —
        splitting it in two would make the halves order-dependent, which is how
        the first version of this test failed.
        """
        for name in (
            "KOALABEAR16_PARAMS",
            "KoalaBear16",
            "BABYBEAR16_PARAMS",
            "BabyBear16",
        ):
            self.assertFalse(_built(name), f"{name} built at import")

        # Reading a citation is a plain module-level string; it must not drag
        # the constant tables or a backend in behind it. Both are checked,
        # because each set answers for its own `__getattr__` arm.
        self.assertEqual(
            standard.KOALABEAR16_PLONKY3_COMMIT,
            "4318eba062fd1cbca3dbe98904ad18ad950f3b49",
        )
        self.assertEqual(
            standard.BABYBEAR16_PLONKY3_COMMIT,
            "90008383a99bdcbf725c91c91efbdf6775da7054",
        )
        self.assertFalse(_built("KoalaBear16"), "reading the citation built it")
        self.assertFalse(_built("BabyBear16"), "reading the citation built it")

        perm = standard.KoalaBear16
        self.assertTrue(_built("KoalaBear16"))
        self.assertIs(standard.KoalaBear16, perm, "not cached — rebuilt per access")
        # The permutation is built over the object the export hands out, not a
        # second equal-but-distinct bundle.
        self.assertTrue(_built("KOALABEAR16_PARAMS"))
        self.assertIs(standard.KOALABEAR16_PARAMS, standard.KOALABEAR16_PARAMS)

        # Building one set must not build the other: the two arms are separate,
        # and a shared `_params()` would make a KoalaBear consumer pay for
        # BabyBear's tables and its `zk_dtypes` field.
        self.assertFalse(_built("BabyBear16"), "KoalaBear16 built its sibling")
        self.assertFalse(_built("BABYBEAR16_PARAMS"), "KoalaBear16 built its sibling")

        bb = standard.BabyBear16
        self.assertTrue(_built("BabyBear16"))
        self.assertIs(standard.BabyBear16, bb, "not cached — rebuilt per access")
        self.assertTrue(_built("BABYBEAR16_PARAMS"))
        self.assertIs(standard.BABYBEAR16_PARAMS, standard.BABYBEAR16_PARAMS)
        # And the two sets are genuinely different parameterizations, not one
        # object handed out twice under two names.
        self.assertIsNot(standard.BABYBEAR16_PARAMS, standard.KOALABEAR16_PARAMS)
        self.assertNotEqual(
            standard.BABYBEAR16_PARAMS.alpha, standard.KOALABEAR16_PARAMS.alpha
        )

    def test_unknown_attribute_still_raises(self) -> None:
        """`__getattr__` must decline names it does not own rather than invent
        them — otherwise a typo resolves to a permutation."""
        with self.assertRaises(AttributeError):
            _ = standard.KoalaBear32


if __name__ == "__main__":
    absltest.main()

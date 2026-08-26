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

        It sweeps `standard._SETS` rather than naming the sets, so a set that
        ships without being swept is not a thing that can happen: the table it
        joins to be exported is the table this reads.
        """
        for name in standard._SETS:
            self.assertFalse(_built(name), f"{name} built at import")

        # Reading a citation is a plain module-level string; it must not drag
        # the constant tables or a backend in behind it.
        self.assertEqual(
            standard.KOALABEAR16_PLONKY3_COMMIT,
            "4318eba062fd1cbca3dbe98904ad18ad950f3b49",
        )
        self.assertEqual(
            standard.BABYBEAR16_PLONKY3_COMMIT,
            "90008383a99bdcbf725c91c91efbdf6775da7054",
        )
        for name in standard._SETS:
            self.assertFalse(_built(name), f"reading a citation built {name}")

        # Build one set's names and assert the OTHER set stayed unbuilt. That is
        # the property that matters — not "the arms are separate", which is a
        # mechanism, but "building one set must not build another", which
        # survives any mechanism and is what a shared builder would break.
        koalabear = standard.KoalaBear16
        self.assertTrue(_built("KoalaBear16"))
        self.assertIs(standard.KoalaBear16, koalabear, "rebuilt per access")
        self.assertFalse(_built("BabyBear16"), "KoalaBear16 built its sibling")
        self.assertFalse(_built("BABYBEAR16_PARAMS"), "KoalaBear16 built its sibling")

        babybear = standard.BabyBear16
        self.assertTrue(_built("BabyBear16"))
        self.assertIs(standard.BabyBear16, babybear, "rebuilt per access")

        # Every name in the table caches, swept rather than listed so a set
        # added without an assertion cannot slip through.
        for name in standard._SETS:
            with self.subTest(name=name):
                self.assertTrue(_built(name), f"{name} never built")
                self.assertIs(getattr(standard, name), getattr(standard, name))

        # And the sets are genuinely different parameterizations rather than one
        # object handed out under several names.
        self.assertIsNot(standard.BABYBEAR16_PARAMS, standard.KOALABEAR16_PARAMS)
        self.assertNotEqual(
            standard.BABYBEAR16_PARAMS.alpha, standard.KOALABEAR16_PARAMS.alpha
        )
        # A set's two names are ONE bundle, though: a permutation must be built
        # over the object the export hands out, not over a second equal one.
        # Two equal-but-distinct bundles are two jit cache keys for one
        # parameterization, which `params.py` states the cost of.
        self.assertIs(standard.KoalaBear16._p, standard.KOALABEAR16_PARAMS)
        self.assertIs(standard.BabyBear16._p, standard.BABYBEAR16_PARAMS)

    def test_dir_lists_sets_that_have_not_been_built(self) -> None:
        """`__dir__` exists so tab-completion sees a set before anything asks
        for it — a module with a `__getattr__` and no `__dir__` hides its own
        exports."""
        listed = dir(standard)
        for name in standard._SETS:
            self.assertIn(name, listed)

    def test_unknown_attribute_still_raises(self) -> None:
        """`__getattr__` must decline names it does not own rather than invent
        them — otherwise a typo resolves to a permutation."""
        with self.assertRaises(AttributeError):
            _ = standard.KoalaBear32


if __name__ == "__main__":
    absltest.main()

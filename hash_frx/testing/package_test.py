# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Package-integrity guards for accidents tooling keeps reintroducing.

Deliberately light: this target depends on the bare package init and `absl`
only, and
every assertion below holds WITHOUT importing a hash. That is not incidental —
`test_importing_the_package_pulls_no_hash` is the pin on the lazy re-export
arrangement, and it can only be true in a process where nothing else has already
imported a family. frx is not in these runfiles at all, which makes the property
enforced rather than merely asserted: an eager re-export cannot quietly boot a
backend here, it fails to import. The counterparts that need the package
resolved — every export, and the submodule-collision rule — live in
`public_api_test`, which already carries the whole layout.
"""

import ast
import pathlib
import sys

from absl.testing import absltest

import hash_frx


def _type_checking_names() -> set[str]:
    """The names bound inside `__init__.py`'s `if TYPE_CHECKING:` block, read
    from the source rather than executed — the block never runs, so this is the
    only way to see it. Parsed here rather than in the module so the module owes
    nothing to its own guard."""
    tree = ast.parse(pathlib.Path(hash_frx.__file__).read_text(encoding="utf-8"))
    # The block is top-level, so `tree.body` is the whole search space — walking
    # the tree would only add ways to match something that is not it.
    return {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
        for stmt in node.body
        if isinstance(stmt, ast.ImportFrom)
        for alias in stmt.names
    }


class PackageTest(absltest.TestCase):
    def test_version_attr_survives(self) -> None:
        # dev-release stamps hash_frx.__version__ at release time, so an emptied
        # hash_frx/__init__.py breaks the wheel build and nothing else — weeks
        # later. Fail here instead.
        self.assertTrue(getattr(hash_frx, "__version__", ""))

    def test_version_is_a_final_release_version(self) -> None:
        # release.yml refuses a tag that disagrees with this, so a pre-release
        # or dev suffix reaching main makes every tag unreleasable — and makes
        # dev-release stamp `X.Y.ZrcN.devTS`. The suffixes belong on the build.
        self.assertRegex(hash_frx.__version__, r"^\d+(\.\d+)*$")


class PublicApiSurfaceTest(absltest.TestCase):
    def test_all_is_exports_plus_version(self) -> None:
        self.assertEqual(set(hash_frx.__all__), {"__version__", *hash_frx._EXPORTS})

    def test_dir_offers_the_exports_before_they_are_touched(self) -> None:
        # `__getattr__` binds lazily, so an untouched export is absent from
        # `globals()`; without `__dir__` it would also be absent from
        # `dir(hash_frx)` and from tab-completion, which is how a consumer
        # discovers the surface.
        self.assertContainsSubset(hash_frx.__all__, dir(hash_frx))

    def test_type_checking_block_matches_exports(self) -> None:
        # mypy reads the `TYPE_CHECKING` block; the interpreter reads
        # `_EXPORTS`. A name added to one and not the other type-checks and
        # imports differently, which is the drift this catches.
        self.assertEqual(_type_checking_names(), set(hash_frx._EXPORTS))

    def test_unknown_attribute_still_raises(self) -> None:
        # `__getattr__` must decline rather than invent, or `from hash_frx
        # import <submodule>` stops falling through to the import system.
        with self.assertRaises(AttributeError):
            _ = hash_frx.NotAHash

    def test_importing_the_package_pulls_no_hash(self) -> None:
        # The whole reason the re-exports are lazy. Importing a hash puts its
        # constant tables on the default backend, which initializes that
        # backend; binding the names eagerly here would move that onto `import
        # hash_frx` and defeat `markers.py`'s "readable free of every hash's
        # dependencies" property. Measured at the commit that added this:
        # `import hash_frx` 0.1 ms, `import hash_frx.sha256.sha256` 54 ms + a live
        # backend.
        self.assertNotIn("frx", sys.modules)
        self.assertEqual([m for m in sys.modules if m.startswith("hash_frx.")], [])


if __name__ == "__main__":
    absltest.main()

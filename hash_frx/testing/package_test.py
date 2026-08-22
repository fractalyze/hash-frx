# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Package-integrity guards for accidents tooling keeps reintroducing.

Deliberately light: this target depends on `//hash_frx` and `absl` only, and
every assertion below holds WITHOUT importing a hash. That is not incidental —
`test_importing_the_package_pulls_no_hash` is the pin on the lazy re-export
arrangement, and it can only be true in a process where nothing else has already
imported a family. The counterpart that actually resolves every export lives in
`public_api_test`, in its own process for exactly that reason.
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
    source = pathlib.Path(hash_frx.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.ImportFrom):
                names.update(alias.asname or alias.name for alias in stmt.names)
    return names


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

    def test_no_export_collides_with_a_submodule(self) -> None:
        # `from hash_frx import X` prefers the attribute, but importing
        # `hash_frx.X` anywhere binds the MODULE onto the package — so a name
        # that is both resolves differently depending on what else has been
        # imported. `pbkdf2` is the live example: the function cannot be
        # re-exported while `hash_frx/pbkdf2.py` holds the name.
        package_dir = pathlib.Path(hash_frx.__file__).parent
        submodules = {p.stem for p in package_dir.glob("*.py")} | {
            p.name for p in package_dir.iterdir() if (p / "__init__.py").exists()
        }
        self.assertEqual(set(hash_frx._EXPORTS) & submodules, set())

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
        # `import hash_frx` 0.1 ms, `import hash_frx.sha256` 54 ms + a live
        # backend.
        self.assertNotIn("frx", sys.modules)
        self.assertEqual([m for m in sys.modules if m.startswith("hash_frx.")], [])


if __name__ == "__main__":
    absltest.main()

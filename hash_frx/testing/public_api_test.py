# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every public export resolves, and resolves to the module's own object.

The heavy half of the public-API guard: touching an export imports the module
behind it, which for most rows puts constant tables on the default backend. That
is why this is a separate target from `package_test`, whose central assertion is
that importing the package does NOT do any of this — the two cannot share a
process.

What this pins is that `_EXPORTS` names a module that really defines the name,
and that the re-export is the *same object* rather than a copy. A wrong module
path in `_EXPORTS` is otherwise invisible until a consumer hits that one name.
"""

import importlib

from absl.testing import absltest, parameterized

import hash_frx


class PublicApiTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (name, name, module) for name, module in sorted(hash_frx._EXPORTS.items())
    )
    def test_export_is_the_defining_module_s_object(
        self, name: str, module: str
    ) -> None:
        self.assertIs(
            getattr(hash_frx, name), getattr(importlib.import_module(module), name)
        )

    def test_repeated_access_is_cached(self) -> None:
        # `__getattr__` writes the binding into `globals()`, so the second
        # access must not re-enter it. Same object either way; what this pins is
        # that the name is actually present in the module dict afterwards, which
        # is what makes the second access a plain lookup.
        _ = hash_frx.Sha256
        self.assertIn("Sha256", vars(hash_frx))

    def test_submodule_import_still_resolves(self) -> None:
        # `__getattr__` declines unknown names, which is what leaves
        # `from hash_frx import <submodule>` to the import system. `pbkdf2` is
        # the name that has to keep working this way — it is a module and a
        # function, and the module wins.
        from hash_frx import pbkdf2

        self.assertTrue(callable(pbkdf2.pbkdf2))


if __name__ == "__main__":
    absltest.main()

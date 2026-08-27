# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Walk the package's shipped modules — the one place that knows how.

Three tests need "every `.py` this package ships, excluding the test trees":
the marker registry's completeness guard, the row registry's, and the seam-pin
guard that came after. Each had re-derived the same four lines, and the second
copy dropped the comment explaining the subtle part, which is how the third
would have got it wrong.

Three entry points: `shipped_sources()` for the walk itself, which the row
registry's guard needs because it imports each module to reach live classes;
`declared()` for the walk plus a regex, for the two guards that read
declarations the import cannot see; and `declared_anywhere()` for the flat set,
which is what those two want most of the time.

**The subtlety is `relative_to(root)`.** Under Bazel the absolute path runs
through the calling test's own `…/<target>.runfiles/_main/hash_frx/…` prefix, so
a `"testing" in path.parts` check matches every file and the sweep silently
yields nothing — a completeness guard that passes because it looked at nothing.
Comparing package-relative parts is what makes it look at the package.

What reaches the walk is still bounded by the caller's BUILD deps: a module
whose target is not in runfiles is not on disk to be found. That is a property
of the caller, not of this helper, so a target relying on a complete sweep says
so in its own deps.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import hash_frx


def shipped_sources() -> Iterator[tuple[str, Path]]:
    """Yield `(dotted_module_name, path)` for every shipped module, test trees
    excluded. Ordered, so a failure names the same module every run."""
    root = Path(next(iter(hash_frx.__path__)))
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "testing" in relative.parts:
            continue
        name = ".".join(("hash_frx", *relative.with_suffix("").parts))
        yield name.removesuffix(".__init__"), path


def declared(pattern: re.Pattern[str]) -> dict[str, set[str]]:
    """Every capture of `pattern` in each shipped module, keyed by module name.

    The completeness guards reduce to the same question — what does the package
    declare, and does the registry agree — and they read it from the SOURCE
    because the declarations do not survive to runtime: a marker name is a
    string literal, and a seam-conformance pin lives inside a `TYPE_CHECKING`
    block that never executes.

    Keyed by module rather than flattened, because some of those rules are about
    *where* a declaration sits — a seam pin belongs in the module defining the
    row, so a pin written in the wrong file must not satisfy the check. Callers
    that do not need the keying take `declared_anywhere()`.
    """
    return {
        name: set(pattern.findall(path.read_text())) for name, path in shipped_sources()
    }


def declared_anywhere(pattern: re.Pattern[str]) -> set[str]:
    """Every capture of `pattern` across the shipped modules, as one set.

    A named function rather than a `union` each caller writes, because the empty
    case is the one that matters: a sweep that reached no files answers "nothing
    is declared", and against a set-difference check that reads as a pass. The
    guard for it belongs here, once, rather than in whichever caller remembered.
    """
    per_module = declared(pattern)
    if not per_module:
        raise AssertionError(
            "the package sweep reached no modules, so any check over it would "
            "pass by looking at nothing — see this module's docstring on "
            "`relative_to(root)` and on the caller's BUILD deps"
        )
    return set().union(*per_module.values())

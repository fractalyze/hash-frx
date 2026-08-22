# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Walk the package's shipped modules — the one place that knows how.

Three tests need "every `.py` this package ships, excluding the test trees":
the marker registry's completeness guard, the row registry's, and anything that
comes after. Each had re-derived the same four lines, and the second copy
dropped the comment explaining the subtle part, which is how the third would
have got it wrong.

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

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The re-layering gate: a refactor that is supposed to move code and nothing
else must lower to the same StableHLO, byte for byte.

The byte-exactness suites cannot see the failure this catches. A marker name
that changed, an operand that moved, an attribute that was dropped, a composite
that stopped being emitted — every one of those still computes the right digest,
because an unrecognized or absent marker inlines its decomposition and produces
identical bytes on a slower path. The digest tests pass and the kernel is gone.
So the wire surface needs a gate of its own, and lowered text is that gate.

**What has to be normalized: one line.** `.lower(...).as_text()` was measured
byte-deterministic across separate processes — same SSA numbering, same constant
ordering, same everything — and it carries no `loc(...)` attribution to strip.
The only thing that moves without the program moving is the module name, which
frx derives from the traced callable's `__name__`: the identical computation is
`module @jit_digest` through `sha256.digest` and `module @jit__lambda` through a
lambda wrapping it. Normalizing that one line is the whole of it, and keeping
the normalization this small is deliberate — every additional substitution is a
class of real change the gate stops seeing.

Usage is a capture on the base revision and a comparison on the branch:

    # on the base revision
    golden = lowering_text(sha256.digest, msg)
    # on the branch, in a test
    assert_lowering_unchanged(self, sha256.digest, msg, golden=golden)

`lowering_text` is also what a per-row golden should be built from when a
re-layering step wants one checked in.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable
from typing import Any

import frx

# `module @jit_<name>` — frx builds it from the traced callable's __name__, so
# it changes when a function is renamed or wrapped, neither of which changes the
# program. Anchored at line start and applied once: the name appears in the
# module header, and nowhere the program's meaning depends on it.
_MODULE_NAME = re.compile(r"^module @[\w.$]+", re.MULTILINE)


def normalize(text: str) -> str:
    """Erase the module name — the only variance measured between lowerings of
    the same computation. Everything else in `as_text()` is load-bearing and
    deliberately left alone."""
    return _MODULE_NAME.sub("module @<jit>", text, count=1)


def lowering_text(fn: Callable[..., Any], *args: Any) -> str:
    """The normalized StableHLO `fn` lowers to for `args`. Capture this on the
    revision being preserved; compare against it afterwards."""
    return normalize(frx.jit(fn).lower(*args).as_text())


def assert_lowering_unchanged(
    test: Any, fn: Callable[..., Any], *args: Any, golden: str
) -> None:
    """Assert `fn(*args)` still lowers to `golden`.

    On failure the message is a unified diff rather than two 4000-line blobs —
    the useful information is which operations moved, and a raw inequality
    buries it.
    """
    actual = lowering_text(fn, *args)
    expected = normalize(golden)
    if actual == expected:
        return
    diff = "\n".join(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="golden",
            tofile="actual",
            lineterm="",
            n=2,
        )
    )
    test.fail(f"lowered StableHLO changed:\n{diff}")

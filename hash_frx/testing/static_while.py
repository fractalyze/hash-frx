# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only assertion that a call lowers to ONE static_while fusion on the
generic path.

Fractalyze XLA's `WhileToForConverter` compiles a bounded loop over a batched
state to one kernel, on CPU and GPU alike, and that is the path a family moves
to once its dedicated emitter retires. Whether a family is ready for it is a
property of its decomposition, which a marker hides either way: a recognized
marker swaps the body for its emitter, and an unrecognized one inlines it with
the right bytes whatever form it is in. So the call is lowered with every
`lax.composite` inlined — the module that retirement would leave — and the
compiled text is read for the fusions whose config names the path.

The form the converter accepts is in docs/reference/conventions.md. When a
loop declines, `TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=while_to_for_converter=2`
logs why.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

import frx
from frx import lax

# The config rides in the fusion's backend config, printed once per fusion.
_STATIC_WHILE = re.compile(r'"(static_while:[^"]*)"')
_WHILE = re.compile(r"\bwhile\(")


def _unmarked(
    decomposition: Callable[..., Any], *, name: str, version: int = 0
) -> Callable[..., Any]:
    """`lax.composite`'s signature, handing back the body with no marker."""
    del name, version
    return decomposition


@contextlib.contextmanager
def markers_inlined() -> Iterator[None]:
    """Every marker traced inside this block is its bare decomposition.

    `hash_frx._composite` is the one caller of `lax.composite`, so the patch
    covers every marker kind. Caches are cleared on the way in and out: the
    module-level `jit(inline=True)` zone each family hoists its marked call
    into would otherwise serve a trace that still carries the marker — or,
    after the block, one that has lost it.
    """
    frx.clear_caches()
    try:
        with mock.patch.object(lax, "composite", _unmarked):
            yield
    finally:
        frx.clear_caches()


def static_while_configs(compiled: str) -> list[str]:
    """The static_while fusion configs in a compiled module, duplicates kept, so
    the length is the fusion count."""
    return _STATIC_WHILE.findall(compiled)


def assert_one_static_while(test: Any, fn: Callable[..., Any], *args: Any) -> Any:
    """Assert `fn(*args)`, every marker inlined, compiles to exactly one
    static_while fusion with no `while` left beside it, and return what that
    module computes — so the caller's byte check is about this path rather
    than the routed one.

    The second half is what makes the first mean one kernel: a loop the
    converter declines still computes the right bytes, a kernel per trip, next
    to whatever did convert.
    """
    with markers_inlined():
        compiled = frx.jit(fn).lower(*args).compile()
        text = compiled.as_text()
        configs = static_while_configs(text)
        test.assertLen(configs, 1, f"expected one static_while fusion: {configs}")
        test.assertIsNone(
            _WHILE.search(text), "a loop the converter declined is left in the module"
        )
        return compiled(*args)

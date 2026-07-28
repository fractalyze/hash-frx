# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only fusion-readiness assertion for straight-line bodies.

A body that must lower to one kernel is element-wise field ops plus at most an
inherent reduce — no gather/scatter/dot/while boundary. ``assert_fusion_ready``
lowers ``fn(*args)`` and checks the StableHLO uses only fusion-safe ops plus
exactly ``reduces`` reduce(s). It is a whitelist rather than a gather/dot
blacklist, so ANY boundary op or extra reduce trips it — and any new op in a
fusion-critical body gets a conscious look. A cheap proxy for the XLA rewriter,
which is the authoritative gate.

Applies to the linear layers, not to a whole permutation: a permutation fuses
via its composite marker and normal-form linear layers, which is a different
fusion shape with no dot for XLA to optimize.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import frx

# Element-wise field + structural ops that stay inside one kernel.
_FUSION_SAFE = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "negate",
        "constant",
        "convert",
        "broadcast_in_dim",
        "reshape",
        "slice",
        "concatenate",
        "transpose",
    }
)


def assert_fusion_ready(fn: Callable[..., Any], *args: Any, reduces: int = 0) -> None:
    """Assert ``fn``'s lowered body is straight-line element-wise plus exactly
    ``reduces`` reduce(s); raise ``AssertionError`` naming offenders otherwise."""
    hlo = frx.jit(fn).lower(*args).as_text()
    ops = re.findall(r"stablehlo\.([a-z_]+)", hlo)
    n = ops.count("reduce")
    if n != reduces:
        raise AssertionError(
            f"expected {reduces} reduce(s), got {n} (ops: {sorted(set(ops))})"
        )
    offenders = sorted({o for o in ops if o != "reduce" and o not in _FUSION_SAFE})
    if offenders:
        raise AssertionError(f"non-fusion-safe ops in body: {offenders}")

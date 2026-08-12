# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only fusion-readiness assertion for straight-line bodies.

A body that must lower to one kernel is element-wise ops plus at most an
inherent reduce — no gather/scatter/dot/while boundary, and no call.
``assert_fusion_ready`` lowers ``fn(*args)`` and checks both: that no call
survives, and that the StableHLO uses only fusion-safe ops plus exactly
``reduces`` reduce(s). It is a whitelist rather than a gather/dot blacklist, so
ANY boundary op or extra reduce trips it — and any new op in a fusion-critical
body gets a conscious look. A cheap proxy for the XLA rewriter, which is the
authoritative gate.

The call check is separate because it has to be: a call is not a `stablehlo.`
op, so the op scan is blind to it, and a body can have an entirely fusion-safe
vocabulary while still carrying the boundary that rejects it.

``assert_input_uses`` is the second half of what a layer owes, and it is a
different question from fusion-readiness: a body can be perfectly fusion-safe and
still read its chained input once per lane, which is the quadratic fan-out
`hash_frx.linear` forbids. Nothing about the layer's output or its op vocabulary
shows it — only counting reads in the traced graph does.

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
        # An index the kernel counts rather than a value it loads: `iota` reads
        # no memory, so it fuses into its consumer wherever `constant` does.
        # Admitted on that property alone — a body reaches for it under the
        # operand-ABI rule in `docs/reference/conventions.md`, but that is not
        # what earns an op a place here.
        "iota",
        # Bit-oriented hashes compute in machine words rather than a field, so
        # their element-wise vocabulary is boolean and shift rather than
        # add/multiply. `select` belongs here for the same reason `add` does —
        # it is element-wise — and is how a compile-time mask is applied without
        # `fnp.where`, whose internal jit would put a call in the body.
        "and",
        "or",
        "xor",
        "not",
        "shift_left",
        "shift_right_logical",
        "select",
    }
)


def assert_fusion_ready(fn: Callable[..., Any], *args: Any, reduces: int = 0) -> None:
    """Assert ``fn``'s lowered body is straight-line element-wise plus exactly
    ``reduces`` reduce(s); raise ``AssertionError`` naming offenders otherwise."""
    hlo = frx.jit(fn).lower(*args).as_text()

    # A call is a boundary the rewriter rejects, and the op scan below cannot
    # see one: `func.call` is not a `stablehlo.` op, so a body whose whole
    # vocabulary is fusion-safe still fails the contract if an `fnp` wrapper
    # carrying an internal jit put a call in it. `fnp.where` is the usual
    # source — it lowers to `call @_where` while emitting a clean
    # `stablehlo.select` inside the callee.
    callees = sorted(set(re.findall(r"call @([A-Za-z_][\w.]*)", hlo)))
    if callees:
        raise AssertionError(
            f"call boundaries in body: {callees} — use the `lax` primitive "
            "rather than the `fnp` wrapper"
        )

    ops = re.findall(r"stablehlo\.([a-z_]+)", hlo)
    n = ops.count("reduce")
    if n != reduces:
        raise AssertionError(
            f"expected {reduces} reduce(s), got {n} (ops: {sorted(set(ops))})"
        )
    offenders = sorted({o for o in ops if o != "reduce" and o not in _FUSION_SAFE})
    if offenders:
        raise AssertionError(f"non-fusion-safe ops in body: {offenders}")


def assert_input_uses(
    fn: Callable[..., Any], *args: Any, arg: int = 0, limit: int
) -> None:
    """Assert ``fn`` reads its ``arg``-th traced input at most ``limit`` times.

    A layer must not read a chained input more times than its output shape
    requires. Each extra read is a separate expression the compiler re-derives the
    value for, so where the value is loop-carried — a permutation's rounds — the
    cost compounds per round rather than adding. `hash_frx.linear` states the rule
    and what it cost to learn it.

    ``limit`` is where the scaling law goes: pass ``limit=width`` for a layer whose
    lanes each legitimately touch the state once, or the measured constant for a
    value every lane shares. Written that way one call already pins the scaling; a
    second width only catches a superlinearity whose constant is small enough to
    hide at the first.
    """
    jaxpr = frx.make_jaxpr(fn)(*args).jaxpr
    var = jaxpr.invars[arg]
    n = sum(eqn.invars.count(var) for eqn in jaxpr.eqns)
    if n > limit:
        raise AssertionError(
            f"input {arg} read {n} times, over the limit of {limit}. Read it once "
            f"as an array, or hoist its lanes once and index the hoisted list."
        )


def primitive_count(fn: Callable[..., Any], *args: Any, name: str) -> int:
    """How many ``name`` primitives ``fn`` traces to — the same traced-graph
    accounting as ``assert_input_uses``, counting ops instead of reads. Sharper
    than a read count where a per-lane form multiplies as well as re-reads, and it
    fails naming the width that drifted."""
    jaxpr = frx.make_jaxpr(fn)(*args).jaxpr
    return sum(1 for eqn in jaxpr.eqns if eqn.primitive.name == name)

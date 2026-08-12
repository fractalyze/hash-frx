# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""fused_region — mark a straight-line region as one fused kernel.

Wraps a decomposition in a `frx.lax.composite` named `zorch.fused_region`;
Fractalyze XLA's `ZorchFusedRegionRewriter` turns that marker into a single
custom-fusion kernel — one kernel by construction, not by a per-hash compiler
pattern match. The name is deliberately generic: one marker fuses any
straight-line region (a hash permutation, a fold, a sumcheck round, …), so it is
not named after any single use.

Marker names are a wire ABI: Fractalyze XLA's recognizers match by name, and an
unrecognized name does not error — the composite inlines and fusion is silently
lost. A contract change therefore stages through `composite.version` rather than
a rename, and a rename needs the recognizer to accept both spellings first, so
no build ever requires a toolchain newer than itself.

The hash markers carry the `hash_frx.` prefix, naming the repo that owns the
primitive. This one keeps `zorch.` because it is generic rather than owned:
`zorch`'s sumcheck, constraint-eval and jagged regions emit it alongside the
hashes here, so neither prefix describes it.

The operand list is as much a wire ABI as the name, and it is not only what is
passed here: `lax.composite` lifts every array the decomposition materialises on
the host into an operand *ahead* of these, one per call site. So a marked body
threads its constants as operands or counts them on device (`iota`) — the rule,
its two remedies, and what it cost to learn are in
`docs/reference/conventions.md`.

The decomposition must be straight-line element-wise — no loops, reductions, or
gathers — so the region lowers to one kernel: a round sequence is unrolled into
the body (fixed, small counts) and the linear layers use the normal-form helpers
(not `fnp.dot`/`reduce`/`gather`). Use `lax` primitives over `fnp` wrappers
(`lax.select`, not `fnp.where`): a `fnp` wrapper's internal `jit` lowers to a
call inside the body, which the single-kernel rewriter rejects. Name-routed
markers with a dedicated emitter are exempt — their emitters tolerate reductions
and calls.

The `lax.composite` marker is emitted via `hash_frx._composite.composite`, the
one place every marker in this package routes through.
"""

from __future__ import annotations

from collections.abc import Callable

from frx import Array

from hash_frx._composite import _Region, composite

FUSED_REGION_MARKER = "zorch.fused_region"


def fused_region(
    decomposition: Callable[..., _Region],
    *operands: Array,
    name: str = FUSED_REGION_MARKER,
    version: int = 0,
    **attrs: object,
) -> _Region:
    """Mark a region (`decomposition`) as one fused kernel — or, under a
    name-routed marker, a boundary a vendor expands into a kernel chain.

    Under the default marker the region must be straight-line element-wise — no
    loops, reductions, or gathers — so it lowers to a single kernel. It is called
    with `operands`, which become the composite's operands in order.

    A non-default `name` routes the region to a dedicated XLA emitter instead of
    the generic one — e.g. a Poseidon2 region goes to `Poseidon2Fusion` rather
    than the generic `LoopFusion` (unusable for a full hash permutation). The
    `operands` must then follow that emitter's ABI. Such a region need not be
    single-kernel: a vendor may expand it into a kernel chain, so `decomposition`
    may return a pytree, not just an Array.

    `version` and `attrs` ride through to the composite (`composite.version` and
    `composite.attributes`) — the structural metadata a recognizer parses. Both
    default to absent (`version=0`, no attrs), so a plain straight-line region is
    unchanged.
    """
    return composite(decomposition, *operands, name=name, version=version, **attrs)

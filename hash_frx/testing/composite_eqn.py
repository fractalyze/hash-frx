# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only readers for a marker at the JAXPR depth — the eqn, its attributes,
its operand shapes.

`marker_recognized.py` reads the lowered module, because that is where "emitted"
and "recognized" separate. These stop at the jaxpr instead, which is what a test
wants when it needs the eqn ITSELF — its operands, its output aval, the
attributes it carries — rather than a name on the wire. Every marker family
asked that question and every one answered it with its own copy; this is the one
home (#193).

**Top-level trace eqns only.** These walk `jaxpr.eqns` and do not descend into
sub-jaxprs, so a composite emitted behind a non-inlined `frx.jit` or inside a
`lax.scan` body is INVISIBLE here — the count comes back 0, not 1, where
`emitted_composites` on the same function returns the name. That is not a
limitation to work around: the emission zones in this package are
`@partial(frx.jit, inline=True)` precisely so their bodies splice into the
enclosing trace, and a marker that has gone opaque is a fact worth failing on.
But it does mean the two readers are NOT interchangeable, and a scan-driven
path (`keccak/streaming.py`, `extension/sponge.py`) wants the lowered one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import frx


def composite_eqns(fn: Callable[..., Any], *args: Any) -> list[Any]:
    """Every composite eqn in `fn`'s jaxpr — emission order, duplicates kept, so
    the length is the count (`emitted_composites`' rule, read at the jaxpr)."""
    return [
        e
        for e in frx.make_jaxpr(fn)(*args).jaxpr.eqns
        if e.primitive.name == "composite"
    ]


def composite_eqn(fn: Callable[..., Any], *args: Any) -> Any:
    """The ONE composite eqn in `fn`'s jaxpr, for a region that must emit
    exactly one.

    Raises `AssertionError` rather than taking a `TestCase`, which is this
    package's split for a `(fn, *args)` reader — `fusion_ready.assert_fusion_ready`
    and `lowering_golden.assert_lowering_unchanged` do the same, and the latter
    writes down why: `TestCase.failureException` IS `AssertionError`, so it reads
    as a test failure without the helper having to be handed a `TestCase`.
    """
    eqns = composite_eqns(fn, *args)
    if len(eqns) != 1:
        raise AssertionError(f"expected one composite, got {len(eqns)}")
    return eqns[0]


def composite_attrs(eqn: Any) -> dict[str, Any]:
    """A composite eqn's `attributes` param as a plain dict.

    The param is a sequence of `(key, leaves, treedef)` triples because it rides
    the pytree encoding. One LEAF per key rather than one scalar — an array
    attribute (poseidon2's `external_m4`, poseidon's `mds`) rides as a
    `HashableArray`, which flattens to a single leaf too, so this serves those
    as well.
    """
    return {key: leaves[0] for key, leaves, _ in eqn.params["attributes"]}


def composite_shapes(eqn: Any) -> list[tuple[int, ...]]:
    """A composite eqn's operand shapes, in the recognizer's positional ABI
    order — the assertion that pins operand COUNT and width together."""
    return [tuple(v.aval.shape) for v in eqn.invars]

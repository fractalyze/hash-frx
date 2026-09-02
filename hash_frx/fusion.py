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
no build ever requires a toolchain newer than itself. A third staging exists
where two operand forms are disjoint in element type AND rank: they ride ONE
name and the recognizer tells them apart by operands, which is what
`sha256.SHA256_BYTES_MARKER` does — that requires a recognizer written for it,
so it is a per-ABI question rather than a per-name one.

The hash markers carry the `hash_frx.` prefix, naming the repo that owns the
primitive. This one keeps `zorch.` because it is generic rather than owned:
`zorch`'s sumcheck, constraint-eval and jagged regions emit it alongside the
hashes here, so neither prefix describes it.

The operand list is as much a wire ABI as the name: a name-routed emitter reads
its constants positionally, so a marked body threads them as operands or counts
them on device (`iota`) — a constant left inside the decomposition is one the
emitter cannot find. The rule, its two remedies, and the stronger reason it used
to carry (a `lax.composite` operand lift that frx#218 retired) are in
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

import enum
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import frx
from frx import Array

from hash_frx._composite import _Region, composite
from hash_frx.markers import dedicated_permute_marker

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation

FUSED_REGION_MARKER = "zorch.fused_region"


class FusionPath(enum.Enum):
    """How a hash's one-call unit lowers on the backend it was built for.

    Every row is a device row — it takes a tracer and returns an ``Array`` — so
    the only question left is routing:

    - ``DEDICATED``: the pinned plugin *and* the backend carry the dedicated
      emitter, so the marked call lowers to one device unit. Only this state
      lets a consumer route a whole-region composite (a sponge hash, a Merkle
      commit) through the hash's marker.
    - ``GENERIC``: the backend does not route the marker — either the generic
      region marker, or a dedicated name that inlines unrecognized. Right bytes,
      no dedicated kernel.

    Derived per (hash, backend) at construction — the emitter switch is a
    property of the pin and the backend, so it is never a class constant
    (`keccak.permutation._routes_to_dedicated_emitter` is the pattern).
    """

    DEDICATED = "dedicated"
    GENERIC = "generic"

    @classmethod
    def from_marker(cls, name: str) -> "FusionPath":
        """The device path a marker choice implies: a hash-named marker is one
        kernel, the generic region marker is not. Deriving the path
        from the marker actually chosen is what keeps the two from drifting
        when a routing gate grows another case.

        For a family whose permute marker and whose primitive-driving capability
        are the same question, which is most of them. Where a backend drives the
        primitive but routes no standalone permute kernel, the marker is the
        generic one while the path is still `DEDICATED`; such a family builds the
        path with `from_routing` off its own capability instead
        (`poseidon2._drives_the_primitive` is the case)."""
        return cls.from_routing(name != FUSED_REGION_MARKER)

    @classmethod
    def from_routing(cls, routed: bool) -> "FusionPath":
        """The device path an emitter switch implies, for a hash whose marker
        name never varies (`routed` = the pin *and* the backend carry the
        dedicated emitter)."""
        return cls.DEDICATED if routed else cls.GENERIC

    @property
    def is_one_kernel(self) -> bool:
        """Whether one call lowers to one device unit — the gate for wrapping a
        whole region over this hash in an expandable composite. The question
        `has_dedicated_fusion` used to answer."""
        return self is FusionPath.DEDICATED


def routing(available: bool, backends: tuple[str, ...]) -> bool:
    """Whether a dedicated emitter is reachable: the pinned plugin ships it AND
    this backend is one of the arms it was written for.

    **Call it per construction, never at import.** Reading it at module scope
    freezes the routing to whichever backend happened to be default when the
    module loaded, which is wrong the moment a process builds a hash after
    switching backends. `frx.default_backend()` is memoized, so asking late is
    cheap.

    The two arguments stay per-family constants because they are per-family
    facts: `available` flips with the `frx>=` floor in `pyproject.toml` when
    that emitter lands, and `backends` names the arms it was written for. Only
    the question is shared. What a `False` answer means for the marker is
    `FusionPath.from_routing`'s to say.
    """
    return available and frx.default_backend() in backends


def permute_marker(name: str, version: int) -> tuple[str, int]:
    """The `(name, version)` a permutation puts on the wire for its region.

    Takes whichever name the family's routing gate selected and answers both
    halves of the choice in one place: a generic region carries no version (the
    recognizer reads only the name there, so a version would claim a contract
    the marker does not have), and a routed one gets whichever permute spelling
    `markers` is currently emitting.

    It lives here rather than in `markers` because it needs `FUSED_REGION_MARKER`,
    and `markers` is deliberately dependency-free — so the dependency can only run
    fusion → markers, never back.
    """
    if name == FUSED_REGION_MARKER:
        return FUSED_REGION_MARKER, 0
    return dedicated_permute_marker(name, version)


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


def inert_region_spec(
    perm: "Permutation", leading: Array
) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
    """What `fused_region_spec` returns on the generic path: the leading
    operand alone, a permute that ignores the (absent) ABI constants, and no
    attrs. Every non-dedicated permutation answers identically — consumers
    gate on `fusion_path.is_one_kernel` and never route a whole-region
    composite through it, so the spec only has to be shaped right, never
    emitter-readable — which is why the stub lives here rather than once per
    implementation."""
    return (leading,), (lambda state, *_ops: perm.permute(state)), {}


def fused_region_over(
    perm: "Permutation",
    leading: Array,
    body: Callable[[Array, Callable[[Array], Array]], _Region],
    *,
    name: str,
    version: int,
    **attrs: object,
) -> _Region:
    """Mark a whole computation over `perm.permute` as ONE region, in the
    permutation's own fused-region ABI.

    `body(leading, permute)` is the computation — a sponge's absorb and squeeze,
    a compression tree — written against a plain `permute` callable, so it names
    no operand layout. This rebuilds that callable from the ABI operands
    `perm.fused_region_spec` hands out and threads them through as the region's
    own operands, which is what keeps the constants a permutation needs out of
    the body: an array the body materialises on the host is lifted into an
    unnamed operand ahead of the declared ones, one per site, and a permute
    appears once per absorbed block.

    The permutation's identifying attrs ride alongside `attrs`, so the marker
    says which primitive runs inside it as well as what the enclosing
    construction is. `attrs` wins on a collision, the construction being the one
    that owns the marker name.

    Callers gate on `fusion_path.is_one_kernel`: a permutation on the generic
    marker hands out an inert spec, which names no layout for an emitter to
    read.
    """
    operands, permute_from_operands, perm_attrs = perm.fused_region_spec(leading)

    def region(lead: Array, *constants: Array, **_attrs: object) -> _Region:
        # `_attrs` is marker metadata passed through; the body does not read it.
        return body(lead, lambda state: permute_from_operands(state, *constants))

    merged: dict[str, Any] = {**perm_attrs, **attrs}
    return fused_region(region, *operands, name=name, version=version, **merged)

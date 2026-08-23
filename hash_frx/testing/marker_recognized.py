# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only assertion that a fusion marker is RECOGNIZED, not merely emitted.

Emitting a marker and having it routed to a dedicated emitter are different
properties, and the difference does not show in the output: an unrecognized name
is not an error, it inlines back to the decomposition and computes identical
bytes. So every value-level test passes either way, and a lowered-module test
proves only that this repo wrote the string.

The compiled module is where the two separate. A recognized marker has become a
`kind=kCustom` fusion whose instruction takes the emitter's routing key as its
name; an unrecognized one has left no such fusion behind.

Marker names are a wire ABI shared with Fractalyze XLA, so this is what catches a
rename here that the pinned toolchain does not accept, and a toolchain bump that
drops a name this repo still emits.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import frx

_COMPOSITE_NAME = re.compile(r'stablehlo\.composite "([^"]+)"')


def emitted_composites(fn: Callable[..., Any], *args: Any) -> list[str]:
    """The composite names `fn` lowers to, exactly.

    A marker name must be matched whole. This package's names nest by
    construction — `hash_frx.digest.sha256` is a prefix of
    `…sha256_bytes`, which is a prefix of `…sha256_bytes_len`, and the same
    holds for `poseidon`/`poseidon_sparse`/`poseidon2` and
    `blake3`/`blake3_parent` — so `assertIn(SOME_MARKER, lowered_text)` is
    satisfied by any LONGER sibling and quietly stops testing what it says.
    That is not hypothetical: a routing change that moves the wire from one
    marker to a longer sibling leaves every such assertion green.

    `assert_marker_recognized` already matches its routing key whole for this
    reason; this is the same rule for the emission side, where the question is
    which name the wire carries rather than which emitter claimed it.
    """
    return sorted(set(_COMPOSITE_NAME.findall(frx.jit(fn).lower(*args).as_text())))


def _custom_fusion_lines(fn: Callable[..., Any], *args: Any) -> list[str]:
    """The compiled module's custom-fusion instructions."""
    compiled = frx.jit(fn).lower(*args).compile().as_text()
    return [ln for ln in compiled.splitlines() if "kind=kCustom" in ln]


def assert_marker_recognized(
    test: Any, routing_key: str, fn: Callable[..., Any], *args: Any
) -> None:
    """Assert `fn` compiles to a custom fusion named `routing_key`.

    The instruction name is matched as `%<routing_key> =` rather than by
    substring: `poseidon` is a prefix of `poseidon2`, so a substring match would
    let either emitter satisfy the other's assertion.
    """
    lines = _custom_fusion_lines(fn, *args)
    test.assertTrue(lines, f"no custom fusion: {routing_key} was not recognized")
    test.assertTrue(
        any(f"%{routing_key} =" in ln for ln in lines),
        f"recognized, but not as {routing_key}: {lines}",
    )

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only assertion that `Permutation.fused_region_marker` is what `permute`
actually emits.

The attribute exists so a consumer can RE-MARK a permute inside its own composite
decomposition. That consumer is downstream and does not read this repo's
constants, so the attribute has to be true rather than merely present: a marker
that disagrees with the emission re-marks the fallback with a name no emitter
routes, which does not error — the region inlines and computes identical bytes.

What this catches depends on how an implementation emits, and both cases are
covered on purpose:

- An implementation that marks with a constant of its own — `KeccakF1600`, and
  the seam fakes — declares the attribute separately from what it emits, so the
  two can disagree and this is what notices.
- An implementation that emits *through* the attribute — `Poseidon`, `Poseidon2`,
  `SparsePoseidon` — cannot disagree by construction, which is the stronger
  arrangement and the reason to prefer it. Here this assertion degenerates to
  "exactly one composite, and `has_dedicated_fusion` agrees"; the emission is
  pinned to the module constants by that instance's own marker-emission test, so
  the constant-to-attribute chain stays covered end to end.

`has_dedicated_fusion` is checked in the same breath because it is defined as
"the marker is not the generic one" — two ways of saying one thing, and this is
where they are held to it. That half is independent of how the marker is emitted,
so it holds for every implementation.
"""

from __future__ import annotations

from typing import Any

import frx
from frx import Array

from hash_frx.fusion import FUSED_REGION_MARKER
from hash_frx.permutation import Permutation


def _composite_lines(perm: Permutation, state: Array) -> list[str]:
    txt = frx.jit(perm.permute).lower(state).as_text()
    return [ln for ln in txt.splitlines() if "stablehlo.composite" in ln]


def assert_marker_matches_emission(
    test: Any, perm: Permutation, state: Array
) -> None:
    """Assert `perm.fused_region_marker` names the composite `permute` emits, and
    that `has_dedicated_fusion` agrees with it."""
    name, version = perm.fused_region_marker
    test.assertEqual(
        perm.has_dedicated_fusion,
        name != FUSED_REGION_MARKER,
        f"has_dedicated_fusion disagrees with fused_region_marker {name!r}",
    )

    lines = _composite_lines(perm, state)
    test.assertLen(lines, 1, f"permute must emit exactly one composite: {lines}")
    line = lines[0]
    test.assertIn(f'"{name}"', line)
    # StableHLO prints `version` only when non-zero, so a generic region is
    # asserted by the absence of the attribute rather than by `version = 0`.
    if version:
        test.assertIn(f"version = {version}", line)
    else:
        test.assertNotIn("version = ", line)

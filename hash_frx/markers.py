# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The registry of every composite marker this package owns, and its axis.

Every hash here is a fixed-width primitive composed under a domain-extension
construction, and the markers split along that line:

- ``PERM`` — a primitive's unit: one permutation call (`hash_frx.poseidon2`,
  `hash_frx.keccak_f`) or one compression-level hop of a tree hash
  (`hash_frx.blake3_parent`, `hash_frx.blake3_compress`).
- ``DIGEST`` — a whole hash: an entire absorb/squeeze chain or tree collapsed
  behind one digest-shaped call (`hash_frx.sha256`, `hash_frx.keccak_sponge`,
  `hash_frx.blake3`, and the field-level `hash_frx.sponge_hash`).

**Namespace rule.** A *new* primitive marker is born ``hash_frx.perm.<name>``
and a *new* whole-hash marker ``hash_frx.digest.<name>`` (the field sponge's
future spelling is ``hash_frx.field_sponge`` — field-level, so neither byte
namespace claims it). The rows below keep their pre-namespace spellings
because a marker name is a wire ABI (`hash_frx.fusion`): renaming one requires
the Fractalyze XLA recognizer to accept both spellings first, so the renames
stage cross-repo rather than riding a hash-frx change alone.

The registry restates each emitting module's constants as literals instead of
importing them, so reading it stays free of every hash's dependencies (frx, the
`blake3` binding); `markers_test` holds every row equal to its module constant
and the enumeration complete, which is where drift is caught. The inversion —
emitting modules importing their names from here — would de-duplicate too, but
a marker's name and version belong beside its emitter: the version constants
carry their contract-change notes at the emission site, which a central file
would strand.

`zorch.fused_region` and the other `zorch.*` markers are deliberately absent:
they are generic regions the `zorch` repos own, not hashes this package does.
"""

from __future__ import annotations

import enum
from typing import NamedTuple

# The namespaces new markers are born into (see the module docstring). Existing
# rows migrate only behind dual-spelling recognition in Fractalyze XLA.
PERM_NAMESPACE = "hash_frx.perm."
DIGEST_NAMESPACE = "hash_frx.digest."


class MarkerAxis(enum.Enum):
    """Which side of the primitive ∘ construction split a marker sits on."""

    PERM = "perm"  # one primitive unit: a permute, or a tree hash's node hop
    DIGEST = "digest"  # a whole hash collapsed behind one digest-shaped call


class Marker(NamedTuple):
    name: str  # the `composite.name` on the wire
    version: int  # the `composite.version` the emitting module currently rides
    axis: MarkerAxis


MARKERS: tuple[Marker, ...] = (
    # Primitives — one marked region per permute / node hop.
    Marker("hash_frx.poseidon", 1, MarkerAxis.PERM),
    Marker("hash_frx.sparse_poseidon", 2, MarkerAxis.PERM),
    Marker("hash_frx.poseidon2", 2, MarkerAxis.PERM),
    Marker("hash_frx.keccak_f", 1, MarkerAxis.PERM),
    Marker("hash_frx.blake3_parent", 1, MarkerAxis.PERM),
    Marker("hash_frx.blake3_compress", 1, MarkerAxis.PERM),
    # Whole hashes — the construction's entire chain behind one call.
    Marker("hash_frx.sha256", 1, MarkerAxis.DIGEST),
    Marker("hash_frx.keccak_sponge", 1, MarkerAxis.DIGEST),
    Marker("hash_frx.blake3", 1, MarkerAxis.DIGEST),
    Marker("hash_frx.sponge_hash", 1, MarkerAxis.DIGEST),
)

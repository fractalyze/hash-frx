# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The registry of every composite marker this package owns, and its kind.

A marker names the KERNEL that lowers — the fusible unit — not the taxonomy:
an axis-B construction (Merkle–Damgård, Tree, Sponge, Duplex) is a schedule
over a unit, so it gets no namespace of its own. A construction lives baked
into a ``DIGEST`` name (`hash_frx.sha256` IS Merkle–Damgård, the sponge
markers ARE Sponge) and/or as a composite attribute (the sponge mode rides as
``construction=<mode>``); Duplex is stateful, so it has no whole-hash marker
at all — each step is a ``PERM``. Three kinds cover every kernel:

- ``PERM`` — a permutation (n→n): one `permute` call is the marked region.
- ``COMPRESS`` — a compression function (n→m): BLAKE3's parent-node and
  streaming compressions. The `Compression` *class* — a truncated permutation
  — has no marker of its own: its unit IS the permute, so it rides ``PERM``.
- ``DIGEST`` — a whole hash: arbitrary input to a digest, construction baked
  in. `hash_frx.sponge_hash` (the field sponge) sits here too — field-level
  rather than byte-level, but a whole hash all the same.

**Namespace rule.** Every marker is spelled ``hash_frx.<kind>.<name>``:
``hash_frx.perm.*`` / ``hash_frx.compress.*`` / ``hash_frx.digest.*``
(``compress`` rather than ``comp``, which would read as "composite"). The rows
below carry the namespaced spellings; the pre-namespace spellings stayed on
the wire until the pinned Fractalyze XLA recognizers accepted both — a marker
name is a wire ABI (`hash_frx.fusion`), so the rename staged behind
dual-spelling recognition (#165, which also tracks retiring the old
spellings once every producer has flipped).

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
COMPRESS_NAMESPACE = "hash_frx.compress."
DIGEST_NAMESPACE = "hash_frx.digest."


class MarkerKind(enum.Enum):
    """The kind of fusible unit a marker names (the module docstring's split)."""

    PERM = "perm"  # permutation (n→n)
    COMPRESS = "compress"  # compression function (n→m)
    DIGEST = "digest"  # whole hash, construction baked in


class Marker(NamedTuple):
    name: str  # the `composite.name` on the wire
    version: int  # the `composite.version` the emitting module currently rides
    kind: MarkerKind


MARKERS: tuple[Marker, ...] = (
    # Permutations — one marked region per permute.
    Marker("hash_frx.perm.poseidon", 1, MarkerKind.PERM),
    Marker("hash_frx.perm.poseidon_sparse", 2, MarkerKind.PERM),
    Marker("hash_frx.perm.poseidon2", 2, MarkerKind.PERM),
    Marker("hash_frx.perm.keccak_f", 1, MarkerKind.PERM),
    Marker("hash_frx.perm.vision", 1, MarkerKind.PERM),
    # Compression functions — BLAKE3's node-level units (the streaming one is
    # the bare family name; the namespace already says "compression").
    Marker("hash_frx.compress.blake3_parent", 1, MarkerKind.COMPRESS),
    Marker("hash_frx.compress.blake3", 1, MarkerKind.COMPRESS),
    # Whole hashes — the construction's entire chain behind one call.
    Marker("hash_frx.digest.sha256", 1, MarkerKind.DIGEST),
    # The raw-bytes whole-message form: same digest, different wire ABI (the
    # message operand is unpadded bytes; padding lives inside the region). A
    # new name rather than a version bump — the pinned sha256 recognizer
    # hard-fails on an operand-ABI mismatch instead of declining.
    Marker("hash_frx.digest.sha256_bytes", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.keccak_sponge", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.blake3", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.field_sponge", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.grostl256", 1, MarkerKind.DIGEST),
)

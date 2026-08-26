# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The registry of every composite marker this package owns, and its kind.

A marker names the KERNEL that lowers — the fusible unit — not the taxonomy:
an axis-B construction (Merkle–Damgård, Tree, Sponge, Duplex) is a schedule
over a unit, so it gets no namespace of its own. A construction lives baked
into a ``DIGEST`` name (`hash_frx.digest.sha256` IS Merkle–Damgård, the sponge
markers ARE Sponge) and/or as a composite attribute (the field sponge's
chaining rule rides as ``construction=<value>``, whose spellings are frozen
wire ABI rather than `SpongeChaining`'s member names); Duplex is stateful, so
it has no whole-hash marker at all — each step is a ``PERM``. Three kinds
cover every kernel:

- ``PERM`` — a permutation (n→n): one `permute` call is the marked region.
- ``COMPRESS`` — a compression function (n→m): BLAKE3's parent-node and
  streaming compressions. The `Compression` *class* — a truncated permutation
  — has no marker of its own: its unit IS the permute, so it rides ``PERM``.
- ``DIGEST`` — a whole hash: arbitrary input to a digest, construction baked
  in. `hash_frx.digest.field_sponge` (the field sponge) sits here too — field-level
  rather than byte-level, but a whole hash all the same.

**Naming rule.** A marker name is a wire ABI (`hash_frx.fusion`), and two
naming disciplines are in use:

- *Primitive-named* — ``hash_frx.<kind>.<name>``: ``hash_frx.perm.*`` /
  ``hash_frx.compress.*`` / ``hash_frx.digest.*`` (``compress`` rather than
  ``comp``, which would read as "composite"). One emitting module owns the
  name, so its constants live beside that emitter.
- *Operation-named* — ``hash_frx.<operation>``, a single segment with no kind
  prefix: the name IS the kind, so there is no primitive for it to nest behind,
  and WHICH primitive runs rides a composite attribute instead of the name.
  Several modules emit one, so none of them owns it.

`Marker.naming` carries the distinction and `markers_test` holds each row to the
spelling its discipline requires, so a new operation name is a value there
rather than another exemption in the test.

Both renames staged behind dual-spelling recognition in Fractalyze XLA, a marker
name being wire ABI: the pre-namespace spellings stayed on the wire until the
pinned recognizers accepted both (#165, which also tracks retiring them once
every producer has flipped), and the per-primitive permute spellings are doing
the same now behind `hash_frx.permute` (fractalyze/xla#616).

The registry restates each primitive-named module's constants as literals
instead of importing them, so reading it stays free of every hash's dependencies
(frx, the `blake3` binding); `markers_test` holds every row equal to its module
constant and the enumeration complete, which is where drift is caught. The
inversion — emitting modules importing their names from here — would de-duplicate
too, but a marker's name and version belong beside its emitter: the version
constants carry their contract-change notes at the emission site, which a central
file would strand. An operation-named marker has no single emitter to sit beside,
which is why its constants are the exception and are defined here.

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

# The operation-named permute marker: one name for every permutation, with the
# primitive carried in the `permutation` composite attribute the dedicated
# markers already emit. Naming the OPERATION rather than the primitive is what
# lets a permutation ship without minting a marker name, and what gives the
# plugin's registry something to look up other than a suffix.
PERMUTE_MARKER = "hash_frx.permute"

# One name, one wire ABI, so one version -- it tracks the permute SCHEMA
# (state, then the primitive's own constants), not any primitive's parameters.
# The per-primitive versions it replaces staged per-primitive contracts; under
# this name a primitive's contract change rides its `permutation` attribute
# instead.
PERMUTE_MARKER_VERSION = 1

# Emitted only where the pinned plugin recognizes it. The dual-spelling half
# landed in fractalyze/xla#616; this stays False until the `frx>=` floor moves
# to a wheel carrying it, because flipping first makes every permute marker
# unrecognized -- it would not fail, it would silently inline and lose fusion.
#
# Same shape and the same reason as the per-family `_DEDICATED_EMITTER_AVAILABLE`
# constants, and it retires with them when the plugin exposes what it can route.
# It needs no backend axis of its own, unlike those constants' `_EMITTER_BACKENDS`
# siblings: this only re-spells a marker the family's own routing gate has ALREADY
# chosen on this backend, and xla#616 recognizes the new name in the same shared
# rewriter that reads the spellings it replaces. So it cannot change WHERE a
# permutation routes, only what the region it already routes to is called.
_OPERATION_NAMED_PERMUTE = False


def dedicated_permute_marker(name: str, version: int) -> tuple[str, int]:
    """The `(name, version)` a DEDICATED permute marker rides today.

    Takes the primitive's own spelling and returns either it or the
    operation-named one, so the choice is made in a single place rather than
    six. Only the DEDICATED spelling is decided here, because answering the
    generic case too would need `FUSED_REGION_MARKER` and this module is
    deliberately dependency-free; `fusion.permute_marker` composes the two and
    is what a permutation actually calls.
    """
    if _OPERATION_NAMED_PERMUTE:
        return PERMUTE_MARKER, PERMUTE_MARKER_VERSION
    return name, version


class MarkerKind(enum.Enum):
    """The kind of fusible unit a marker names (the module docstring's split)."""

    PERM = "perm"  # permutation (n→n)
    COMPRESS = "compress"  # compression function (n→m)
    DIGEST = "digest"  # whole hash, construction baked in


class MarkerNaming(enum.Enum):
    """Which naming discipline a row follows (the module docstring's split).

    Orthogonal to `MarkerKind`, which says what unit lowers: an operation name
    still names a kind, it just spells it without the primitive.
    """

    PRIMITIVE = "primitive"  # `hash_frx.<kind>.<name>`, one emitter owns it
    OPERATION = "operation"  # `hash_frx.<operation>`, the primitive is an attribute


class Marker(NamedTuple):
    name: str  # the `composite.name` on the wire
    version: int  # the `composite.version` the emitting module currently rides
    kind: MarkerKind
    naming: MarkerNaming = MarkerNaming.PRIMITIVE


MARKERS: tuple[Marker, ...] = (
    # The operation-named permute marker; the per-primitive spellings below it
    # are retiring and stay recognized for one pin cycle.
    Marker(
        PERMUTE_MARKER, PERMUTE_MARKER_VERSION, MarkerKind.PERM, MarkerNaming.OPERATION
    ),
    # Permutations — one marked region per permute.
    Marker("hash_frx.perm.poseidon", 1, MarkerKind.PERM),
    Marker("hash_frx.perm.poseidon_sparse", 2, MarkerKind.PERM),
    Marker("hash_frx.perm.poseidon2", 2, MarkerKind.PERM),
    Marker("hash_frx.perm.keccak_f", 1, MarkerKind.PERM),
    Marker("hash_frx.perm.vision", 1, MarkerKind.PERM),
    Marker("hash_frx.perm.ascon_p", 1, MarkerKind.PERM),
    # Compression functions — BLAKE3's node-level units (the streaming one is
    # the bare family name; the namespace already says "compression").
    Marker("hash_frx.compress.blake3_parent", 1, MarkerKind.COMPRESS),
    Marker("hash_frx.compress.blake3", 1, MarkerKind.COMPRESS),
    # Whole hashes — the construction's entire chain behind one call.
    Marker("hash_frx.digest.sha256", 1, MarkerKind.DIGEST),
    # The raw-bytes whole-message form: same digest, different wire ABI (the
    # message operand is unpadded bytes; padding lives inside the region). A new
    # name rather than a version bump on `…sha256`, whose recognizer hard-fails
    # on an operand-ABI mismatch instead of declining. THIS name's recognizer
    # dispatches on operands instead, which is what lets two ABIs — the length
    # as the message's shape, or as a separate operand — ride under it
    # (`sha256.SHA256_BYTES_MARKER` for why that is safe here).
    Marker("hash_frx.digest.sha256_bytes", 1, MarkerKind.DIGEST),
    # SHA-512, the 64-bit SHA-2 sibling: blocks-shaped only — no raw-bytes
    # sibling until a recognizer ships for one (`hash_frx.digest.sha512` notes the
    # deliberate absence).
    Marker("hash_frx.digest.sha512", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.keccak_sponge", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.blake3", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.field_sponge", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.grostl256", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.ascon_hash256", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.ascon_xof128", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.ripemd160", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.blake2b", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.blake2s", 1, MarkerKind.DIGEST),
    Marker("hash_frx.digest.sm3", 1, MarkerKind.DIGEST),
)

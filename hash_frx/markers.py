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

# The operation-named words-in Merkle-Damgard digest marker: one name for every
# family whose message arrives already padded and packed into words, with the
# primitive carried in the `primitive` composite attribute.
#
# Note what the name means, because it is a namespace as well: `DIGEST_NAMESPACE`
# is `hash_frx.digest.` WITH the dot, and this is the bare stem. The trailing dot
# is what keeps the two from colliding — nearly every whole-hash marker here is
# spelled `hash_frx.digest.<something>`, and most of them are NOT this schema
# (the raw-bytes forms pad in-kernel, the sponges are not Merkle-Damgard).
MD_DIGEST_MARKER = "hash_frx.digest"

# One name, one wire ABI, so one version — it tracks the words-in SCHEMA
# (`[h0, constants, blocks] -> digest bytes`), not any family's parameters.
MD_DIGEST_MARKER_VERSION = 1

# ON: the `frx>=` floor now carries the recognizer (fractalyze/xla#625, with the
# SHA-512 and SM3 registry entries it routes), so the operation name reaches a
# real emitter rather than inlining.
#
# It went on in ONE commit with both families' `_DEDICATED_EMITTER_AVAILABLE`
# and the floor itself, which is what the gate existed to buy. The name alone
# would have left `fusion_path` reporting `GENERIC` while the lowering became a
# kernel underneath it — metadata lying about what it describes, which is
# exactly what `fusion_path_test`'s matrix law exists to catch.
#
# **The reason this was gated differs from `_OPERATION_NAMED_PERMUTE`'s, and
# copying that rationale would be wrong.** Flipping the permute name early loses
# fusion, because the per-primitive permute spellings route TODAY. These never
# did: a words-in digest resolved through `IsSha256Marker` or the operation
# name, and `hash_frx.digest.sha512` / `…sm3` matched neither, so there was no
# fusion for an early flip to lose — only the metadata disagreement above.
#
# Only the two families that ride `words_in_digest_marker` move. SHA-256 keeps
# its own spelling: it is the one words-in family whose per-family name the
# plugin still recognizes, so moving it is a separate change with a rollback
# story of its own.
_OPERATION_NAMED_MD_DIGEST = True


# The operation-named Merkle-Damgard STREAM FINALIZE marker: finish a hash from
# a live midstate at a runtime stream position, with the compression carried in
# the `primitive` composite attribute.
#
#   [h, consts..., pending u8[block], counts s32[2], extras u8[..., E]]
#
# A third MD schema and so a third operation name, on the rule
# `MD_DIGEST_MARKER` states: the message here is not a block count but a stream
# POSITION. `counts` is `[pending_len, total_len]` and rides as a runtime
# OPERAND, which is the point — a producer tracing this hop cannot know how many
# blocks the padded tail spans, so it emits both candidates and selects between
# them, and a kernel taking the position as an operand runs one.
#
# **Flat, not `hash_frx.stream.finalize`.** An operation name is a sibling of
# the other operations, not a child of one — `markers_test` holds every one of
# them to a single segment, and a dotted spelling would read as "the finalize of
# a primitive called stream". `stream_absorb` and `stream_squeeze` join it as
# siblings rather than as a namespace.
STREAM_FINALIZE_MARKER = "hash_frx.stream_finalize"

# One name, one wire ABI, so one version — it tracks the RESUME schema above,
# not any family's parameters.
STREAM_FINALIZE_MARKER_VERSION = 1

# The operation-named RAW-BYTES Merkle-Damgard digest marker: one name for every
# family whose message arrives UNPADDED, with the primitive carried in the
# `primitive` composite attribute exactly as `MD_DIGEST_MARKER` carries it.
#
# A separate operation from `MD_DIGEST_MARKER` rather than a version of it,
# because the two are different WIRE ABIs and not different parameterizations of
# one: words-in hands the kernel pre-padded blocks, raw-bytes hands it the
# message and a zero tail and the padding happens inside the region. An emitter
# reading one as the other reads a message where a block count should be.
#
# **Flat, not `hash_frx.digest.bytes`.** The dotted spelling would put a LIVE
# operation name inside `DIGEST_NAMESPACE`, which is where the RETIRING
# per-family spellings live and is slated for deletion as a group -- and it
# would read as "digest of a family called bytes". `hash_frx.permute` and
# `hash_frx.digest` are flat for the same reason: an operation name is a sibling
# of the other operations, not a child of one.
BYTES_DIGEST_MARKER = "hash_frx.digest_bytes"

# One name, one wire ABI, so one version -- it tracks the raw-bytes SCHEMA
# (`[h0, consts..., msg, tail] -> digest bytes`), not any family's parameters.
# `consts...` is zero or more: RIPEMD-160 carries none at all, where the BLAKE2
# pair each carry an IV, so the count is the registry entry's and not the
# schema's.
BYTES_DIGEST_MARKER_VERSION = 1

# Emitted only where the pinned plugin recognizes it. The recognizer landed in
# fractalyze/xla#635 over the #632 envelope, and the registry entries it
# resolves through in #636 (RIPEMD-160), #639 (BLAKE2s) and #642 (BLAKE2b);
# this stays False until the `frx>=` floor moves to a wheel carrying them.
#
# **The reason this is gated differs from BOTH its siblings', and copying either
# rationale would be wrong.** Flipping the permute name early loses fusion the
# old spellings already have. Flipping the words-in name early would have left
# `fusion_path` reporting `GENERIC` over a real kernel. Flipping THIS one early
# does neither -- and does nothing at all: the three families emit
# `hash_frx.digest.ripemd160` / `...blake2s` / `...blake2b`, no recognizer
# matches those, and an old plugin does not match the operation name either, so
# they inline before and after.
#
# So this flag is not what makes the flip safe; the ORDERING is, and the hazard
# lives on the family gates rather than here. Turning a family's
# `_DEDICATED_EMITTER_AVAILABLE` on before the floor carries its entry makes
# `fusion_path` advertise a dedicated kernel over a body that still inlines --
# metadata lying about what it describes, which is what `fusion_path_test`'s
# matrix law exists to catch. This rides with those gates so the rename and the
# routing claim land together: floor first, then this and the gates in one
# commit.
#
# What is already in place is the `primitive` composite attribute, which the
# three families emit under BOTH spellings -- inert while each carries its own
# name, so the flip is a rename and nothing else.
_OPERATION_NAMED_BYTES_DIGEST = False


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


def words_in_digest_marker(name: str, version: int) -> tuple[str, int]:
    """The `(name, version)` a words-in Merkle-Damgard digest rides today.

    `dedicated_permute_marker`'s sibling, at the digest kind: takes the family's
    own spelling and returns either it or the operation-named one, so the choice
    is made in one place rather than once per family.

    Only the words-in schema belongs here. A raw-bytes digest is a DIFFERENT
    wire ABI — its message operand is unpadded and the padding happens inside
    the region — so it does not join this name; `bytes_in_digest_marker` is its
    own sibling behind its own flag, and `hash_frx.digest.sha256_bytes` states
    why that is a new name and not a version bump.
    """
    if _OPERATION_NAMED_MD_DIGEST:
        return MD_DIGEST_MARKER, MD_DIGEST_MARKER_VERSION
    return name, version


def bytes_in_digest_marker(name: str, version: int) -> tuple[str, int]:
    """The `(name, version)` a raw-bytes Merkle-Damgard digest rides today.

    `words_in_digest_marker`'s sibling one schema over: takes the family's own
    spelling and returns either it or the operation-named one, so the choice is
    made in one place rather than once per family.

    Only the STATIC-length raw-bytes schema belongs here — `[h0, consts..., msg,
    tail]`, where the block count is a shape property.

    **The runtime-LENGTH forms are a different schema and keep their own names.**
    They carry a scalar length operand and synthesize their padding from it, so
    their block count is a runtime value. Two families are in that group and
    both are easy to mis-wire here, for opposite reasons:
    `sha256.SHA256_BYTES_MARKER`, because SHA-256 is the family the raw-bytes
    envelope was lifted out of; and `grostl.GROSTL256_MARKER`, because Grøstl
    walks `masked_chain` exactly as RIPEMD-160 does and so reads like a
    raw-bytes MD digest. Grøstl is the more dangerous of the two — it is
    RECOGNIZED and routed on the pinned plugin today, so wiring it here would be
    a perfect no-op while the flag is off and would move a live marker onto an
    envelope with no length operand the moment it flips.

    The `primitive` composite attribute is this migration's other half, and the
    three callers emit it alongside their own names rather than only after the
    flip. That is what makes the flip a rename and nothing else: the plugin
    resolves the family through the attribute once the name stops carrying it,
    so a caller that stopped emitting it would silently decline into its
    decomposition instead of failing. `extension/md.py`'s `chain` states the
    same arrangement for the words-in schema.
    """
    if _OPERATION_NAMED_BYTES_DIGEST:
        return BYTES_DIGEST_MARKER, BYTES_DIGEST_MARKER_VERSION
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
    # The operation-named words-in Merkle-Damgard digest; the per-family
    # spellings below it are retiring and stay recognized for one pin cycle.
    Marker(
        MD_DIGEST_MARKER,
        MD_DIGEST_MARKER_VERSION,
        MarkerKind.DIGEST,
        MarkerNaming.OPERATION,
    ),
    # The stream FINALIZE: a whole hash too, finished from a midstate rather
    # than from an IV. Born operation-named -- there is no per-family spelling
    # to retire, because this schema has never shipped under one.
    Marker(
        STREAM_FINALIZE_MARKER,
        STREAM_FINALIZE_MARKER_VERSION,
        MarkerKind.DIGEST,
        MarkerNaming.OPERATION,
    ),
    # The operation-named RAW-BYTES Merkle-Damgard digest. Its own row rather
    # than a version of the one above because the two are different wire ABIs:
    # words-in takes pre-padded blocks, this takes the message and a zero tail.
    Marker(
        BYTES_DIGEST_MARKER,
        BYTES_DIGEST_MARKER_VERSION,
        MarkerKind.DIGEST,
        MarkerNaming.OPERATION,
    ),
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

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The marker registry is held to the emitting modules, row by row.

`hash_frx.markers` restates names and versions as literals so reading the wire
surface costs none of the hashes' dependencies; this test is where the
restatement is held to the constants it mirrors, and where the enumeration is
held complete — a marker constant landing or moving without its registry row
fails here rather than drifting silently.
"""

from __future__ import annotations

import re
from unittest import mock

from absl.testing import absltest

from hash_frx import markers, sponge
from hash_frx.ascon import ascon
from hash_frx.ascon import permutation as ascon_permutation
from hash_frx.blake2b import blake2b
from hash_frx.blake2s import blake2s
from hash_frx.blake3 import rows as blake3_rows
from hash_frx.blake3 import streaming as blake3_streaming
from hash_frx.grostl import grostl
from hash_frx.keccak import permutation as keccak_permutation
from hash_frx.keccak import sponge as keccak_sponge
from hash_frx.markers import (
    COMPRESS_NAMESPACE,
    DIGEST_NAMESPACE,
    MARKERS,
    PERM_NAMESPACE,
    MarkerKind,
    MarkerNaming,
)
from hash_frx.poseidon import poseidon, sparse
from hash_frx.poseidon2 import poseidon2
from hash_frx.ripemd160 import ripemd160
from hash_frx.sha256 import sha256
from hash_frx.sha512 import sha512
from hash_frx.sm3 import sm3
from hash_frx.testing.package_sweep import declared_anywhere
from hash_frx.vision import vision

# Every emitting module's (name, version) pair, read from the constants the
# markers actually ride the wire with.
_MODULE_CONSTANTS = {
    poseidon.POSEIDON_MARKER: poseidon.POSEIDON_MARKER_VERSION,
    sparse.POSEIDON_SPARSE_MARKER: sparse.POSEIDON_SPARSE_MARKER_VERSION,
    poseidon2.POSEIDON2_MARKER: poseidon2.POSEIDON2_MARKER_VERSION,
    keccak_permutation.KECCAK_F_MARKER: keccak_permutation.KECCAK_F_MARKER_VERSION,
    vision.VISION_MARKER: vision.VISION_MARKER_VERSION,
    blake3_rows.BLAKE3_MARKER: blake3_rows.BLAKE3_MARKER_VERSION,
    blake3_rows.BLAKE3_PARENT_MARKER: blake3_rows.BLAKE3_PARENT_MARKER_VERSION,
    # The streaming compression's marker sits with its only emitter, which is
    # the rule this file's module docstring states.
    blake3_streaming.BLAKE3_COMPRESS_MARKER: (
        blake3_streaming.BLAKE3_COMPRESS_MARKER_VERSION
    ),
    sha256.SHA256_MARKER: sha256.SHA256_MARKER_VERSION,
    sha256.SHA256_BYTES_MARKER: sha256.SHA256_BYTES_MARKER_VERSION,
    sha512.SHA512_MARKER: sha512.SHA512_MARKER_VERSION,
    keccak_sponge.KECCAK_SPONGE_MARKER: keccak_sponge.KECCAK_SPONGE_MARKER_VERSION,
    sponge.SPONGE_HASH_MARKER: sponge.SPONGE_HASH_MARKER_VERSION,
    grostl.GROSTL256_MARKER: grostl.GROSTL256_MARKER_VERSION,
    ascon.ASCON_HASH256_MARKER: ascon.ASCON_HASH256_MARKER_VERSION,
    ascon.ASCON_XOF128_MARKER: ascon.ASCON_XOF128_MARKER_VERSION,
    ascon_permutation.ASCON_P_MARKER: ascon_permutation.ASCON_P_MARKER_VERSION,
    ripemd160.RIPEMD160_MARKER: ripemd160.RIPEMD160_MARKER_VERSION,
    blake2b.BLAKE2B_MARKER: blake2b.BLAKE2B_MARKER_VERSION,
    blake2s.BLAKE2S_MARKER: blake2s.BLAKE2S_MARKER_VERSION,
    sm3.SM3_MARKER: sm3.SM3_MARKER_VERSION,
    # The operation-named permute marker's "emitting module" is `markers`
    # itself: six permutations emit it, so no one of them owns it, and the
    # constant lives where the choice between spellings is made.
    markers.PERMUTE_MARKER: markers.PERMUTE_MARKER_VERSION,
    # Same arrangement one kind over: the words-in digest marker is emitted by
    # every Merkle-Damgard family whose message arrives already packed, so its
    # constant lives with the choice rather than with any one of them.
    markers.MD_DIGEST_MARKER: markers.MD_DIGEST_MARKER_VERSION,
    # And one schema over again: the stream FINALIZE is emitted by
    # `extension/md.py` for every MD family, so the constant lives here for the
    # same reason -- no single family owns an operation name.
    markers.STREAM_FINALIZE_MARKER: markers.STREAM_FINALIZE_MARKER_VERSION,
    # And the raw-bytes one beside it, for the same reason: three families emit
    # it, so its constant lives with the choice rather than with any of them.
    markers.BYTES_DIGEST_MARKER: markers.BYTES_DIGEST_MARKER_VERSION,
}


# The three families that ride `bytes_in_digest_marker`: the STATIC-length
# raw-bytes schema. SHA-256's raw-bytes form is deliberately absent -- it is the
# runtime-LENGTH schema, a third wire ABI that keeps its own name.
_BYTES_IN_FAMILIES = (
    "hash_frx.digest.ripemd160",
    "hash_frx.digest.blake2s",
    "hash_frx.digest.blake2b",
)


class MarkerRegistryTest(absltest.TestCase):
    def test_every_row_matches_its_module_constant(self) -> None:
        self.assertEqual({m.name: m.version for m in MARKERS}, _MODULE_CONSTANTS)

    def test_names_are_unique_and_package_prefixed(self) -> None:
        names = [m.name for m in MARKERS]
        self.assertCountEqual(names, set(names))
        for name in names:
            # The prefix names the repo that owns the primitive; the generic
            # `zorch.*` regions are deliberately not rows (`hash_frx.markers`).
            self.assertStartsWith(name, "hash_frx.")

    def test_the_enumeration_is_complete(self) -> None:
        # Sweep the package sources for module-level `<X>_MARKER = "hash_frx.…"`
        # constants: a new wire name landing without a registry row is exactly
        # the drift the registry cannot catch about itself. Testing modules are
        # excluded — a test fake's marker never reaches the wire. The sweep
        # sees what this test's BUILD deps place in runfiles, so a new hash
        # package joins those deps when it joins the registry.
        pattern = re.compile(r'^[A-Z0-9_]*MARKER = "(hash_frx\.[a-z0-9_.]+)"', re.M)
        found = declared_anywhere(pattern)
        self.assertEqual(found, {m.name for m in MARKERS})

    def test_every_row_is_spelled_in_its_kinds_namespace(self) -> None:
        # The inverse of the pre-flip pin: every row now carries the namespace
        # its kind names, so a marker landing outside its namespace — or a
        # kind disagreeing with the spelling — is the wire-ABI drift this
        # catches.
        namespace_of = {
            MarkerKind.PERM: PERM_NAMESPACE,
            MarkerKind.COMPRESS: COMPRESS_NAMESPACE,
            MarkerKind.DIGEST: DIGEST_NAMESPACE,
        }
        for marker in MARKERS:
            with self.subTest(marker=marker.name):
                if marker.naming is MarkerNaming.OPERATION:
                    # An operation name IS the kind, so it has no primitive to
                    # nest behind — it must be a bare `hash_frx.<operation>`
                    # and must NOT sit inside its kind's namespace. Checked
                    # rather than skipped: `hash_frx.digest` would otherwise
                    # pass for the wrong reason, and every operation name the
                    # relayering adds is held to the same spelling.
                    self.assertRegex(marker.name, r"^hash_frx\.[a-z0-9_]+$")
                    self.assertNotStartsWith(marker.name, namespace_of[marker.kind])
                else:
                    self.assertStartsWith(marker.name, namespace_of[marker.kind])


class OperationNamedMdDigestTest(absltest.TestCase):
    """The digest-side flip, now shipped ON.

    `OperationNamedPermuteTest`'s sibling, and no longer its mirror: that flag
    is still off. Worth its own class because the two flip for different
    reasons — the permute one to avoid losing fusion the old spellings already
    have, this one to keep the wire spelling and `fusion_path` moving together
    — so neither is a proxy for the other.
    """

    def test_off_reports_the_familys_own_spelling(self) -> None:
        # Patched off, because the flag now ships ON: the pinned plugin routes
        # the operation name. Kept rather than deleted — it is the rollback
        # path, and the one that fails if the fallback ever stops returning the
        # family's own spelling verbatim.
        with mock.patch.object(markers, "_OPERATION_NAMED_MD_DIGEST", False):
            self.assertEqual(
                markers.words_in_digest_marker("hash_frx.digest.sha512", 1),
                ("hash_frx.digest.sha512", 1),
            )

    def test_on_reports_the_operation_name_for_every_family(self) -> None:
        # One name and one version whatever the family was; the primitive
        # reaches the plugin as the `primitive` attribute instead. Only the
        # WORDS-IN families migrate, so the list is explicit rather than every
        # DIGEST row — the raw-bytes forms and the sponges are different wire
        # ABIs that keep their own names.
        # Unpatched: this is the shipped state now, so the assertion runs
        # against the flag as configured rather than against a mocked one.
        words_in = ["hash_frx.digest.sha512", "hash_frx.digest.sm3"]
        for name in words_in:
            with self.subTest(marker=name):
                self.assertEqual(
                    markers.words_in_digest_marker(name, 1),
                    (markers.MD_DIGEST_MARKER, markers.MD_DIGEST_MARKER_VERSION),
                )

    def test_the_operation_name_is_not_inside_the_digest_namespace(self) -> None:
        # The bare stem and the namespace differ by one trailing dot, and a
        # `startswith` written against the wrong one would silently make every
        # `hash_frx.digest.*` row look operation-named.
        self.assertEqual(markers.MD_DIGEST_MARKER, DIGEST_NAMESPACE.rstrip("."))
        self.assertNotStartsWith(markers.MD_DIGEST_MARKER, DIGEST_NAMESPACE)


class OperationNamedBytesDigestTest(absltest.TestCase):
    """The raw-bytes flip, still OFF.

    `OperationNamedMdDigestTest`'s sibling one schema over, and currently in the
    opposite state: that flag ships ON, this one waits for a `frx>=` floor
    carrying the recognizer (fractalyze/xla#635) and the registry entries it
    resolves through (#636/#639/#642).
    """

    def test_off_reports_the_familys_own_spelling(self) -> None:
        # The shipped state, so this runs unpatched: while the flag is off every
        # family must get its own spelling back VERBATIM. A fallback that
        # altered the name would silently change the wire on a pin that still
        # reads the old one.
        for name in _BYTES_IN_FAMILIES:
            with self.subTest(marker=name):
                self.assertEqual(markers.bytes_in_digest_marker(name, 1), (name, 1))

    def test_on_reports_the_operation_name_for_every_family(self) -> None:
        # One name and one version whatever the family was; the primitive
        # reaches the plugin as the `primitive` attribute instead.
        with mock.patch.object(markers, "_OPERATION_NAMED_BYTES_DIGEST", True):
            for name in _BYTES_IN_FAMILIES:
                with self.subTest(marker=name):
                    self.assertEqual(
                        markers.bytes_in_digest_marker(name, 1),
                        (
                            markers.BYTES_DIGEST_MARKER,
                            markers.BYTES_DIGEST_MARKER_VERSION,
                        ),
                    )

    def test_the_flag_is_off_until_the_pin_carries_the_recognizer(self) -> None:
        # The gate itself. Flipping this before the floor moves does not fail --
        # the families swap one unrecognized name for another and keep inlining,
        # which is a green suite with no kernel. So the constant is pinned here
        # and moves in the same commit as the floor.
        self.assertFalse(markers._OPERATION_NAMED_BYTES_DIGEST)

    def test_it_is_a_separate_operation_from_the_words_in_one(self) -> None:
        # Two schemas, two names. Collapsing them would hand a raw-bytes region
        # to an envelope expecting pre-padded blocks -- it would read the
        # message where a block count belongs.
        self.assertNotEqual(markers.BYTES_DIGEST_MARKER, markers.MD_DIGEST_MARKER)

    def test_the_operation_name_is_not_inside_the_digest_namespace(self) -> None:
        # Flat, like `hash_frx.permute` and `hash_frx.digest`. The dotted
        # spelling would put a LIVE name inside the namespace the RETIRING
        # per-family spellings live in, which is slated for deletion as a group.
        self.assertNotStartsWith(markers.BYTES_DIGEST_MARKER, DIGEST_NAMESPACE)


class OperationNamedPermuteTest(absltest.TestCase):
    """The flip itself, which is otherwise unexercised while the flag is off.

    The constant is false until the pinned plugin recognizes the name, so
    without these the operation-named path ships never having run.
    """

    def test_off_reports_the_primitives_own_spelling(self) -> None:
        self.assertEqual(
            markers.dedicated_permute_marker("hash_frx.perm.poseidon2", 2),
            ("hash_frx.perm.poseidon2", 2),
        )

    def test_on_reports_the_operation_name_for_every_primitive(self) -> None:
        # One name and one version whatever the primitive was: that IS the
        # change, and it is why the primitive has to reach the plugin as an
        # attribute instead. Walked off the registry rather than a hand-listed
        # table, so a version bump or a seventh permutation cannot strand a
        # stale row here.
        retiring = [
            m
            for m in MARKERS
            if m.kind is MarkerKind.PERM and m.naming is MarkerNaming.PRIMITIVE
        ]
        self.assertLen(retiring, 6)
        with mock.patch.object(markers, "_OPERATION_NAMED_PERMUTE", True):
            for m in retiring:
                with self.subTest(marker=m.name):
                    self.assertEqual(
                        markers.dedicated_permute_marker(m.name, m.version),
                        (markers.PERMUTE_MARKER, markers.PERMUTE_MARKER_VERSION),
                    )

    def test_flipping_keeps_the_permutation_attribute_on_the_wire(self) -> None:
        # The plugin reads the primitive from `permutation` once the name stops
        # carrying it, so the attribute every permute marker already emits is
        # what makes the rename safe. If a primitive ever stopped emitting it,
        # flipping would silently decline that permutation into its
        # decomposition rather than fail. One primitive is enough to pin the
        # precondition HERE, where a reader of the rename looks; each family's
        # own test pins its full attribute set off the traced equation.
        self.assertEqual(keccak_permutation._marker_attrs()["permutation"], "keccak_f")

    def test_the_flag_is_off_until_the_pin_carries_the_recognizer(self) -> None:
        # Guards the ordering the whole dual-spelling cycle exists for: emitting
        # a name the pinned plugin does not know loses fusion silently rather
        # than failing. Flip this together with the `frx>=` floor.
        self.assertFalse(markers._OPERATION_NAMED_PERMUTE)


if __name__ == "__main__":
    absltest.main()

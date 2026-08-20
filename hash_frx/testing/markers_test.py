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
from pathlib import Path

from absl.testing import absltest

import hash_frx
from hash_frx import sha256, sha512, sponge
from hash_frx.blake3 import blake3
from hash_frx.grostl import grostl
from hash_frx.keccak import permutation as keccak_permutation
from hash_frx.keccak import sponge as keccak_sponge
from hash_frx.markers import (
    COMPRESS_NAMESPACE,
    DIGEST_NAMESPACE,
    MARKERS,
    PERM_NAMESPACE,
    MarkerKind,
)
from hash_frx.poseidon import poseidon, sparse
from hash_frx.poseidon2 import poseidon2
from hash_frx.vision import vision

# Every emitting module's (name, version) pair, read from the constants the
# markers actually ride the wire with.
_MODULE_CONSTANTS = {
    poseidon.POSEIDON_MARKER: poseidon.POSEIDON_MARKER_VERSION,
    sparse.POSEIDON_SPARSE_MARKER: sparse.POSEIDON_SPARSE_MARKER_VERSION,
    poseidon2.POSEIDON2_MARKER: poseidon2.POSEIDON2_MARKER_VERSION,
    keccak_permutation.KECCAK_F_MARKER: keccak_permutation.KECCAK_F_MARKER_VERSION,
    vision.VISION_MARKER: vision.VISION_MARKER_VERSION,
    blake3.BLAKE3_MARKER: blake3.BLAKE3_MARKER_VERSION,
    blake3.BLAKE3_PARENT_MARKER: blake3.BLAKE3_PARENT_MARKER_VERSION,
    blake3.BLAKE3_COMPRESS_MARKER: blake3.BLAKE3_COMPRESS_MARKER_VERSION,
    sha256.SHA256_MARKER: sha256.SHA256_MARKER_VERSION,
    sha256.SHA256_BYTES_MARKER: sha256.SHA256_BYTES_MARKER_VERSION,
    sha512.SHA512_MARKER: sha512.SHA512_MARKER_VERSION,
    keccak_sponge.KECCAK_SPONGE_MARKER: keccak_sponge.KECCAK_SPONGE_MARKER_VERSION,
    sponge.SPONGE_HASH_MARKER: sponge.SPONGE_HASH_MARKER_VERSION,
    grostl.GROSTL256_MARKER: grostl.GROSTL256_MARKER_VERSION,
}


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
        root = Path(next(iter(hash_frx.__path__)))
        found: set[str] = set()
        for path in root.rglob("*.py"):
            # Package-relative parts: under bazel the absolute path runs
            # through this test's own `…/testing/markers_test.runfiles/…`
            # prefix, which would otherwise skip every file.
            if "testing" in path.relative_to(root).parts:
                continue
            found |= set(pattern.findall(path.read_text()))
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
            self.assertStartsWith(marker.name, namespace_of[marker.kind])


if __name__ == "__main__":
    absltest.main()

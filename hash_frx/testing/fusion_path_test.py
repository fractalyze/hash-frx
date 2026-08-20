# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The (hash, backend) → FusionPath matrix, asserted cell by cell.

Each device family keeps its own pin+backend switch next to its marker; this is
the one place the *facts* those switches encode — which backends carry which
dedicated emitters — are spelled together, independently of the production
tuples, so a tuple edit is a conscious two-place change rather than a silent
routing move. The states themselves are pinned with the cells: an absent
backend reads GENERIC (device, traceable, un-routed), never HOST, and a host
row reads HOST on every leg.

Per-family derivation mechanics (pin veto, backend veto, marker agreement) live
with each family — `keccak.testing.permutation_test.EmitterGateTest` is the
pattern — and are not repeated here.
"""

from __future__ import annotations

from unittest import mock

import frx
import numpy as np
from absl.testing import absltest
from zk_dtypes import binary_field_t5, goldilocks_mont

from hash_frx import sha256 as sha256_mod
from hash_frx import sha512 as sha512_mod
from hash_frx.ascon import ascon as ascon_mod
from hash_frx.ascon.ascon import AsconHash256
from hash_frx.blake2b.byte_hashes import HostBlake2b
from hash_frx.blake3 import byte_hashes as blake3_rows
from hash_frx.blake3.byte_hashes import Blake3, HostBlake3
from hash_frx.compression import Compression, CompressionParams
from hash_frx.duplex_sponge import DuplexSponge
from hash_frx.fusion import FusionPath
from hash_frx.grostl import grostl as grostl_mod
from hash_frx.grostl.grostl import Grostl256
from hash_frx.keccak import permutation as keccak_perm_mod
from hash_frx.keccak.byte_hashes import HostSha3_256, Sha3_256
from hash_frx.keccak.permutation import KeccakF1600
from hash_frx.poseidon import poseidon as poseidon_mod
from hash_frx.poseidon import sparse as sparse_mod
from hash_frx.poseidon2 import poseidon2 as poseidon2_mod
from hash_frx.poseidon2.testing.koalabear16 import koalabear16_perm
from hash_frx.rescue import rescue as rescue_mod
from hash_frx.rescue.params import rescue_rpo128_params
from hash_frx.rescue.rescue import Rescue
from hash_frx.sha256 import HostSha256, Sha256
from hash_frx.sha512 import HostSha512, Sha512
from hash_frx.sponge import Sponge, SpongeParams
from hash_frx.vision import vision as vision_mod
from hash_frx.vision.params import vision_mark32_params
from hash_frx.vision.vision import Vision

# The matrix rows: which backends the pinned plugin's dedicated emitters cover,
# per family. Keccak's arms are GPU-only; the rest run wherever the
# ZorchFusedRegionRewriter (cpu+gpu compilers) routes them — except sparse
# Poseidon, whose CPU mis-routing cost is measured in `poseidon.sparse`, and
# Vision, Rescue, Grøstl and Ascon, for which no plugin ships an emitter at
# all.
_MATRIX = {
    poseidon_mod: ("cpu", "gpu"),
    poseidon2_mod: ("cpu", "gpu"),
    sparse_mod: ("gpu",),
    keccak_perm_mod: ("gpu",),
    vision_mod: (),
    rescue_mod: (),
    sha256_mod: ("cpu", "gpu"),
    sha512_mod: (),
    blake3_rows: ("cpu", "gpu"),
    grostl_mod: (),
    ascon_mod: (),
}


class MatrixFactsTest(absltest.TestCase):
    def test_the_production_tuples_are_the_documented_matrix(self) -> None:
        for module, backends in _MATRIX.items():
            with self.subTest(module=module.__name__):
                self.assertEqual(module._EMITTER_BACKENDS, backends)
                # Premise for every cell below: wherever any backend has an
                # arm the pin half of the switch is on, so the backend half is
                # what decides; a family with no emitter anywhere keeps the
                # pin off with the tuple (Vision today).
                self.assertEqual(module._DEDICATED_EMITTER_AVAILABLE, bool(backends))

    def test_the_enum_property_table(self) -> None:
        # Pinned once; the per-impl cases below assert only the member, since
        # `is_one_kernel` / `is_traceable` are functions of the enum alone.
        self.assertEqual(
            [(p.is_one_kernel, p.is_traceable) for p in FusionPath],
            [(True, True), (False, True), (False, False)],
        )


class DeviceCellTest(absltest.TestCase):
    def test_this_leg_reads_its_cell(self) -> None:
        backend = frx.default_backend()
        rows = (
            (KeccakF1600(), _MATRIX[keccak_perm_mod]),
            (koalabear16_perm(), _MATRIX[poseidon2_mod]),
            (Sha256(), _MATRIX[sha256_mod]),
            (Sha3_256(), _MATRIX[keccak_perm_mod]),
            (Blake3(), _MATRIX[blake3_rows]),
            # The empty rows: GENERIC on every leg, by the same derivation as
            # an absent backend — never HOST.
            (Vision(vision_mark32_params(binary_field_t5)), _MATRIX[vision_mod]),
            (Rescue(rescue_rpo128_params(goldilocks_mont)), _MATRIX[rescue_mod]),
            (Grostl256(), _MATRIX[grostl_mod]),
            (Sha512(), _MATRIX[sha512_mod]),
            (AsconHash256(), _MATRIX[ascon_mod]),
        )
        for impl, backends in rows:
            with self.subTest(impl=type(impl).__name__):
                expected = (
                    FusionPath.DEDICATED if backend in backends else FusionPath.GENERIC
                )
                self.assertIs(impl.fusion_path, expected)

    def test_an_absent_backend_reads_generic_not_host(self) -> None:
        # The Metal-shaped cell: a device hash on a backend without the arm is
        # still a device hash — traceable, un-fused — and collapsing it into
        # HOST is the exact conflation the three-state enum retires. Keccak is
        # absent from the loop: its family gate test
        # (`keccak.testing.permutation_test.EmitterGateTest`) already owns the
        # backend-veto mock; these three families have no gate test of their
        # own.
        for module, build in (
            (poseidon2_mod, koalabear16_perm),
            (sha256_mod, Sha256),
            (blake3_rows, Blake3),
        ):
            with self.subTest(module=module.__name__):
                with mock.patch.object(module, "_EMITTER_BACKENDS", ("nonesuch",)):
                    impl = build()
                self.assertIs(impl.fusion_path, FusionPath.GENERIC)


class HostCellTest(absltest.TestCase):
    def test_host_rows_are_host_on_every_leg(self) -> None:
        for row in (
            HostSha256(),
            HostSha512(),
            HostSha3_256(),
            HostBlake3(),
            HostBlake2b(),
        ):
            with self.subTest(row=type(row).__name__):
                self.assertIs(row.fusion_path, FusionPath.HOST)


class ConstructionDelegationTest(absltest.TestCase):
    def test_constructions_read_their_permutation(self) -> None:
        perm = koalabear16_perm()
        for c in (
            Sponge(perm, SpongeParams(rate=8, out=8)),
            Compression(perm, CompressionParams(arity=2, chunk=8)),
            DuplexSponge(perm, rate=8),
        ):
            with self.subTest(construction=type(c).__name__):
                self.assertIs(c.fusion_path, perm.fusion_path)


class TraceabilityTest(absltest.TestCase):
    def test_is_traceable_agrees_with_the_return_type(self) -> None:
        # The seam's rule: the return type is the authority and the attribute
        # answers to it. One cheap pair proves the tie on both sides.
        msg = np.zeros((1, 3), dtype=np.uint8)
        device, host = Sha256(), HostSha256()
        self.assertTrue(device.fusion_path.is_traceable)
        self.assertNotIsInstance(device.digest(msg), np.ndarray)
        self.assertFalse(host.fusion_path.is_traceable)
        self.assertIsInstance(host.digest(msg), np.ndarray)


if __name__ == "__main__":
    absltest.main()

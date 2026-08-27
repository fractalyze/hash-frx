# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The (hash, backend) → FusionPath matrix, asserted cell by cell.

Each device family keeps its own pin+backend switch next to its marker; this is
the one place the *facts* those switches encode — which backends carry which
dedicated emitters — are spelled together, independently of the production
tuples, so a tuple edit is a conscious two-place change rather than a silent
routing move. The states themselves are pinned with the cells: an absent
backend reads GENERIC (traceable, un-routed) rather than losing its cell.

Per-family derivation mechanics (pin veto, backend veto, marker agreement) live
with each family — `keccak.testing.permutation_test.EmitterGateTest` is the
pattern — and are not repeated here.
"""

from __future__ import annotations

from unittest import mock

import frx
from absl.testing import absltest
from zk_dtypes import binary_field_t5

from hash_frx.ascon import ascon as ascon_mod
from hash_frx.ascon.ascon import AsconCxof128, AsconHash256, AsconXof128
from hash_frx.blake2b import blake2b as blake2b_mod
from hash_frx.blake2b.blake2b import Blake2b
from hash_frx.blake2s import blake2s as blake2s_mod
from hash_frx.blake2s.blake2s import Blake2s
from hash_frx.blake3 import rows as blake3_rows
from hash_frx.blake3.rows import Blake3
from hash_frx.compression import Compression, CompressionParams
from hash_frx.duplex_sponge import DuplexSponge
from hash_frx.fusion import FusionPath
from hash_frx.grostl import grostl as grostl_mod
from hash_frx.grostl.grostl import Grostl256
from hash_frx.keccak import permutation as keccak_perm_mod
from hash_frx.keccak.byte_hashes import Sha3_256
from hash_frx.keccak.permutation import KeccakF1600
from hash_frx.poseidon import poseidon as poseidon_mod
from hash_frx.poseidon import sparse as sparse_mod
from hash_frx.poseidon2 import poseidon2 as poseidon2_mod
from hash_frx.poseidon2.testing.koalabear16 import koalabear16_perm
from hash_frx.ripemd160 import ripemd160 as ripemd160_mod
from hash_frx.ripemd160.ripemd160 import Ripemd160
from hash_frx.sha256 import sha256 as sha256_mod
from hash_frx.sha256.sha256 import Sha256
from hash_frx.sha512 import sha512 as sha512_mod
from hash_frx.sha512.sha512 import (
    Sha384,
    Sha512,
    Sha512_256,
)
from hash_frx.sm3 import sm3 as sm3_mod
from hash_frx.sm3.sm3 import Sm3
from hash_frx.sponge import Sponge, SpongeParams
from hash_frx.vision import vision as vision_mod
from hash_frx.vision.params import vision_mark32_params
from hash_frx.vision.vision import Vision

# The matrix rows: which backends the pinned plugin's dedicated emitters cover,
# per family. They run wherever the ZorchFusedRegionRewriter (cpu+gpu
# compilers) routes them — except sparse Poseidon, whose CPU mis-routing cost
# is measured in `poseidon.sparse`, and Vision, Ascon, RIPEMD-160 and the
# BLAKE2 pair, for which no plugin ships an emitter at all.
#
# SHA-512 and SM3 arrive with BOTH arms at once, unlike every family above
# them. They are not per-family emitters: the plugin routes them through its
# shared words-in Merkle-Damgard envelope, which both compilers already gate
# on, so registering the compression lit up cpu and gpu in one step. That is
# also why neither could ship one backend at a time the way Grøstl did — there
# is no per-backend arm to ship separately.
#
# Keccak covers both legs only from the wheel carrying the CPU sponge emitter.
# Its two arms had to arrive together: one tuple gates the permute marker and
# the whole-hash one, and routing the coarser marker to a leg that cannot honour
# it costs the whole absorb-and-squeeze trace (`keccak.permutation` records the
# measurement).
_MATRIX = {
    poseidon_mod: ("cpu", "gpu"),
    poseidon2_mod: ("cpu", "gpu"),
    sparse_mod: ("gpu",),
    keccak_perm_mod: ("cpu", "gpu"),
    vision_mod: (),
    sha256_mod: ("cpu", "gpu"),
    sha512_mod: ("cpu", "gpu"),
    blake3_rows: ("cpu", "gpu"),
    grostl_mod: ("cpu", "gpu"),
    ascon_mod: (),
    ripemd160_mod: (),
    blake2b_mod: (),
    blake2s_mod: (),
    sm3_mod: ("cpu", "gpu"),
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
        # `is_one_kernel` is a function of the enum alone.
        self.assertEqual(
            [p.is_one_kernel for p in FusionPath],
            [True, False],
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
            (Grostl256(), _MATRIX[grostl_mod]),
            # The empty rows: GENERIC on every leg, by the same derivation as
            # an absent backend.
            (Vision(vision_mark32_params(binary_field_t5)), _MATRIX[vision_mod]),
            (Sha512(), _MATRIX[sha512_mod]),
            # The truncated variants read the sha512 module's switch — one
            # family row serves all three (the h0-as-operand design).
            (Sha384(), _MATRIX[sha512_mod]),
            (Sha512_256(), _MATRIX[sha512_mod]),
            (AsconHash256(), _MATRIX[ascon_mod]),
            (AsconXof128(32), _MATRIX[ascon_mod]),
            (AsconCxof128(b"", 32), _MATRIX[ascon_mod]),
            (Ripemd160(), _MATRIX[ripemd160_mod]),
            (Blake2b(), _MATRIX[blake2b_mod]),
            (Blake2s(), _MATRIX[blake2s_mod]),
            (Sm3(), _MATRIX[sm3_mod]),
        )
        for impl, backends in rows:
            with self.subTest(impl=type(impl).__name__):
                expected = (
                    FusionPath.DEDICATED if backend in backends else FusionPath.GENERIC
                )
                self.assertIs(impl.fusion_path, expected)

    def test_an_absent_backend_reads_generic(self) -> None:
        # The Metal-shaped cell: a hash on a backend without the arm is still
        # traceable and un-fused rather than unavailable. Keccak is
        # absent from the loop: its family gate test
        # (`keccak.testing.permutation_test.EmitterGateTest`) already owns the
        # backend-veto mock, in combinations this loop cannot express; these
        # four families have no gate test of their own.
        for module, build in (
            (poseidon2_mod, koalabear16_perm),
            (sha256_mod, Sha256),
            (blake3_rows, Blake3),
            (grostl_mod, Grostl256),
        ):
            with self.subTest(module=module.__name__):
                with mock.patch.object(module, "_EMITTER_BACKENDS", ("nonesuch",)):
                    impl = build()
                self.assertIs(impl.fusion_path, FusionPath.GENERIC)


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


if __name__ == "__main__":
    absltest.main()

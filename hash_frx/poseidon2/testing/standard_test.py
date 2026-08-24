# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The shipped standard parameter sets, held to the references they name.

The point of shipping a set is that the byte-match assertion and the constants
stop living in different repos. These tests are that assertion: they run the
*exported* member, not a fixture copy of it.
"""

from __future__ import annotations

import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

import hash_frx
from hash_frx.poseidon2.poseidon2 import Poseidon2
from hash_frx.poseidon2.standard import KOALABEAR16_PLONKY3_COMMIT, KoalaBear16
from hash_frx.poseidon2.testing.koalabear16 import KOALABEAR16_EXPECTED


class KoalaBear16ExportTest(absltest.TestCase):
    def test_exports_from_the_package_root(self) -> None:
        self.assertIs(hash_frx.KoalaBear16, KoalaBear16)

    def test_is_a_ready_permutation(self) -> None:
        """A ready instance, not a params bundle or a factory — a consumer
        naming the set wants to permute with it, not assemble it."""
        self.assertIsInstance(KoalaBear16, Poseidon2)
        self.assertEqual(KoalaBear16.width, 16)

    def test_carries_its_source_citation(self) -> None:
        """The revision the constants were generated from is a value, not a
        comment: 'which Plonky3 is this' becomes a question the package
        answers."""
        self.assertEqual(
            KOALABEAR16_PLONKY3_COMMIT, "4318eba062fd1cbca3dbe98904ad18ad950f3b49"
        )


class KoalaBear16ReferenceTest(absltest.TestCase):
    """Byte-match against the Plonky3 revision the set names."""

    def test_permute_matches_plonky3_vector(self) -> None:
        got = KoalaBear16.permute(fnp.arange(16, dtype=F))
        self.assertEqual([int(x) for x in got], [int(x) for x in KOALABEAR16_EXPECTED])


if __name__ == "__main__":
    absltest.main()

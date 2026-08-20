# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The oracle and the shipped tables are anchored before anything is held to them.

`rescue_test` checks the frx Rescue against `reference.py` over the shipped
RPO-128 tables, which only means something if both are right. Oracle and
tables were both written here, so agreement between them would also be the
signature of one transcription error applied twice.

The published digests are the independent party: printed in the RPO paper's
Section 3.1 and hard-coded by miden-crypto's Rust implementation
(`reference.py` carries both provenances), they exercise every MDS entry,
every round constant, and both power maps through all 7 rounds of up to three
permutations — a wrong constant, matrix entry, or exponent does not survive
them. The SHAKE256 derivation then pins the shipped EXPANDED
`round_constants` to the Section 2.2 procedure, and the circulant law pins
the shipped `mds` to the published first row — the two generations the spec
defines and the parameter surface does not carry. (`alpha`/`inv_alpha` are
pinned in `params_test`, beside the surface that stores them.)
"""

from __future__ import annotations

from absl.testing import absltest
from zk_dtypes import goldilocks_mont as F

from hash_frx.rescue.params import rescue_rpo128_params
from hash_frx.rescue.testing.decode import int_rows
from hash_frx.rescue.testing.reference import (
    RPO128_CAPACITY,
    RPO128_M,
    RPO128_MDS_ROW,
    RPO128_P,
    RPO128_ROUNDS,
    RPO128_SECURITY_LEVEL,
    RPO128_TEST_VECTORS,
    circulant,
    get_round_constants,
    rpo_hash,
)

_P = rescue_rpo128_params(F)
_MDS = int_rows(_P.mds)
_ROUND_CONSTANTS = int_rows(_P.round_constants)


class ReferenceAnchorTest(absltest.TestCase):
    def test_rpo_hash_matches_every_published_digest(self) -> None:
        # All 19 rows, over the SHIPPED tables: hash([0..i]) covers one-block
        # padded (i < 8), exact-block (i = 7), and multi-block (i > 7) inputs,
        # so the whole permutation chain and the sponge schedule are pinned
        # together.
        for i, want in enumerate(RPO128_TEST_VECTORS):
            with self.subTest(input_length=i + 1):
                got = rpo_hash(
                    list(range(i + 1)),
                    _MDS,
                    _ROUND_CONSTANTS,
                    _P.alpha,
                    _P.inv_alpha,
                    RPO128_P,
                    RPO128_M,
                    RPO128_CAPACITY,
                )
                self.assertEqual(got, list(want))

    def test_round_constants_match_the_shake256_derivation(self) -> None:
        # The shipped surface carries the EXPANDED table; this is where it is
        # held to the Section 2.2 procedure it was expanded from. Every one of
        # the 2mN constants, not a spot check.
        flat = get_round_constants(
            RPO128_P,
            RPO128_M,
            RPO128_CAPACITY,
            RPO128_SECURITY_LEVEL,
            RPO128_ROUNDS,
        )
        want = [
            flat[i * RPO128_M : (i + 1) * RPO128_M] for i in range(2 * RPO128_ROUNDS)
        ]
        self.assertEqual(_ROUND_CONSTANTS, want)

    def test_mds_matches_the_published_circulant_row(self) -> None:
        # The factory materializes the circulant; this is where the full
        # (12, 12) table is held to the published first row and the
        # row-rotation law (Section 2.3).
        self.assertEqual(_MDS, circulant(list(RPO128_MDS_ROW)))


if __name__ == "__main__":
    absltest.main()

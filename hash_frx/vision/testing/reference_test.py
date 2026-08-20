# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The oracle and the shipped tables are anchored before anything is held to them.

`vision_test` checks the frx Vision against `reference.py` over the shipped
Mark-32 tables, which only means something if both are right. Oracle and tables
were both written here, so agreement between them would also be the signature
of one transcription error applied twice.

The published plaintext/ciphertext vectors are the independent party: produced
by the binius-models Sage generator, they exercise every table entry and both
S-boxes through all 16 steps — a wrong MDS byte, B coefficient, or schedule row
does not survive them. The generator recurrence then pins the shipped
`round_keys` to the (initial_constant, constants_matrix, constants_constant)
triple the instance was sampled with, and the affine-composition check pins
`b_inv` to `b` — the two derivations the paper specifies and the parameter
surface does not carry.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from zk_dtypes import binary_field_t5 as F

from hash_frx.vision.params import vision_mark32_params
from hash_frx.vision.testing.reference import (
    CONSTANTS_CONSTANT,
    CONSTANTS_MATRIX,
    INITIAL_CONSTANT,
    MARK32_RANDOM_CIPHERTEXT,
    MARK32_RANDOM_PLAINTEXT,
    MARK32_ZERO_CIPHERTEXT,
    MARK32_ZERO_PLAINTEXT,
    evaluate_affine,
    expand_round_keys,
    vision_permutation,
)


def _ints(arr: object) -> list[int]:
    return [int(v) for v in np.asarray(arr).astype(object)]


def _int_rows(arr: object) -> list[list[int]]:
    return [[int(v) for v in row] for row in np.asarray(arr).astype(object)]


_P = vision_mark32_params(F)
_B = _ints(_P.b)
_B_INV = _ints(_P.b_inv)
_MDS = _int_rows(_P.mds)
_ROUND_KEYS = _int_rows(_P.round_keys)


class ReferenceAnchorTest(absltest.TestCase):
    def test_permutation_of_the_zero_state_matches_the_published_vector(self) -> None:
        got = vision_permutation(
            list(MARK32_ZERO_PLAINTEXT), _B, _B_INV, _MDS, _ROUND_KEYS
        )
        self.assertEqual(got, list(MARK32_ZERO_CIPHERTEXT))

    def test_permutation_of_the_random_state_matches_the_published_vector(
        self,
    ) -> None:
        # A zero state stays schedule-determined until the first S-box, so
        # only the random vector exercises generic nonzero inverses everywhere.
        got = vision_permutation(
            list(MARK32_RANDOM_PLAINTEXT), _B, _B_INV, _MDS, _ROUND_KEYS
        )
        self.assertEqual(got, list(MARK32_RANDOM_CIPHERTEXT))

    def test_round_keys_match_the_schedule_recurrence(self) -> None:
        # The shipped surface carries the EXPANDED schedule; this is where it
        # is held to the generator triple it was expanded from (the paper's
        # key schedule under the all-zero master key).
        got = expand_round_keys(
            list(INITIAL_CONSTANT),
            [list(row) for row in CONSTANTS_MATRIX],
            list(CONSTANTS_CONSTANT),
            _B,
            _B_INV,
            _MDS,
            rounds=8,
        )
        self.assertEqual(got, _ROUND_KEYS)

    def test_b_inv_composes_with_b_to_the_identity(self) -> None:
        # b_inv is DATA on the parameter surface; that it really is B's
        # compositional inverse (https://eprint.iacr.org/2019/426, Section 4.1:
        # B invertible over F_{2^n}) is only pinned here. Both compositions,
        # since an affine polynomial has distinct left/right errors; 0 included
        # (the affine constant path).
        for x in (0, 1, 2, 0xDEADBEEF, 0x12345678, 0xFFFFFFFF):
            self.assertEqual(evaluate_affine(_B, evaluate_affine(_B_INV, x)), x)
            self.assertEqual(evaluate_affine(_B_INV, evaluate_affine(_B, x)), x)


if __name__ == "__main__":
    absltest.main()

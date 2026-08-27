# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""babybear-16 test-side companion to the shipped `BabyBear16` set.

The sibling of [`koalabear16.py`](koalabear16.py) and the same arrangement: the
constants ship from [`poseidon2/standard.py`](../standard.py) and this module
holds only the Plonky3 output vector, so a drift between what is tested and what
is published cannot open up.

`BABYBEAR16_EXPECTED` is `permute(0..15)` under plonky3
`default_babybear_poseidon2_16` at the revision `BABYBEAR16_PLONKY3_COMMIT`
names — tag v0.4.3. It is not transcribed from the Rust: it is the golden array
openvm-zorch's `tools/fixture-gen` dumped from that tag and tested against for
the life of its vendored copy, which is what makes it an *independent* oracle
for the set this package now ships rather than a restatement of it.
"""

from __future__ import annotations

import frx.numpy as fnp
from zk_dtypes import babybear_mont as F

from hash_frx.poseidon2.poseidon2 import Poseidon2
from hash_frx.poseidon2.standard import BABYBEAR16_PARAMS

BABYBEAR16_EXPECTED = fnp.array(
    [
        1906786279,
        1737026427,
        1959749225,
        700325316,
        1638050605,
        1021608788,
        1726691001,
        1761127344,
        1552405120,
        417318995,
        36799261,
        1215172152,
        614923223,
        1300746575,
        957311597,
        304856115,
    ],
    dtype=F,
)


def babybear16_perm() -> Poseidon2:
    """The shipped set as a permutation — built per call for the reason
    `standard.py`'s `__getattr__` gives: a module-scope instance would freeze
    its routing to whichever backend was default at import."""
    return Poseidon2(BABYBEAR16_PARAMS)

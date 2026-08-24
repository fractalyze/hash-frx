# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Standard Poseidon2 parameter sets, as named members.

`params.py` ships the *shape* — `Poseidon2Params` and `default_external_matrix`
— and this module ships *instances* of it: the parameterizations consumers
actually run, each pinned to the released implementation it reproduces. The
split mirrors [`adapter/duplex.py`](../adapter/duplex.py), where the
construction owns the axes and the adapter owns the members: a set is a claim
about an external codebase, and a claim about the outside world needs a name
and a citation rather than a comment.

**What this fixes is a duplication that fails silently.** Before this module
the constants lived in each consumer's own tree, verified in exactly one place
(hash-frx's test fixture, against Plonky3) and used in several. A round-constant
table that is wrong does not crash; it produces a hash that is merely
*different*, and the difference surfaces as a Merkle root mismatch three layers
away. Shipping the set with `testing/standard_test.py` holding it to the
reference is what makes the constants and their proof travel together.

**Which sets belong here.** The test is "is this set one scheme's?".
KoalaBear-16 below is a general Plonky3 parameterization, so it passes. A set
that is a particular prover's vendored instance does not, and stays with that
prover — the repo's application-agnosticism rule (README, "Two non-negotiables")
is not suspended for a constant table. Concretely: SP1's vendored KoalaBear-16,
which shares these round constants but carries a powers-of-two internal diagonal
and folds a Montgomery R^-1 into `internal_j_scale`, is *not* here.
"""

from __future__ import annotations

import frx.numpy as fnp
from zk_dtypes import koalabear_mont as _F

from hash_frx.poseidon2.params import Poseidon2Params
from hash_frx.poseidon2.poseidon2 import Poseidon2

# The revision these constants were generated from, as a value rather than a
# comment: "which Plonky3 is this" is then a question the package answers.
KOALABEAR16_PLONKY3_COMMIT = "4318eba062fd1cbca3dbe98904ad18ad950f3b49"

_WIDTH, _ER, _IR, _ALPHA = 16, 4, 20, 3

# Canonical-u32 constants from the pinned Plonky3 koala-bear poseidon2-16.
_EXTERNAL_INITIAL = [
    [
        2128964168,
        288780357,
        316938561,
        2126233899,
        426817493,
        1714118888,
        1045008582,
        1738510837,
        889721787,
        8866516,
        681576474,
        419059826,
        1596305521,
        1583176088,
        1584387047,
        1529751136,
    ],
    [
        1863858111,
        1072044075,
        517831365,
        1464274176,
        1138001621,
        428001039,
        245709561,
        1641420379,
        1365482496,
        770454828,
        693167409,
        757905735,
        136670447,
        436275702,
        525466355,
        1559174242,
    ],
    [
        1030087950,
        869864998,
        322787870,
        267688717,
        948964561,
        740478015,
        679816114,
        113662466,
        2066544572,
        1744924186,
        367094720,
        1380455578,
        1842483872,
        416711434,
        1342291586,
        1692058446,
    ],
    [
        1493348999,
        1113949088,
        210900530,
        1071655077,
        610242121,
        1136339326,
        2020858841,
        1019840479,
        678147278,
        1678413261,
        1361743414,
        61132629,
        1209546658,
        64412292,
        1936878279,
        1980661727,
    ],
]

_EXTERNAL_TERMINAL = [
    [
        1139268644,
        630873441,
        669538875,
        462500858,
        876500520,
        1214043330,
        383937013,
        375087302,
        636912601,
        307200505,
        390279673,
        1999916485,
        1518476730,
        1606686591,
        1410677749,
        1581191572,
    ],
    [
        1004269969,
        143426723,
        1747283099,
        1016118214,
        1749423722,
        66331533,
        1177761275,
        1581069649,
        1851371119,
        852520128,
        1499632627,
        1820847538,
        150757557,
        884787840,
        619710451,
        1651711087,
    ],
    [
        505263814,
        212076987,
        1482432120,
        1458130652,
        382871348,
        417404007,
        2066495280,
        1996518884,
        902934924,
        582892981,
        1337064375,
        1199354861,
        2102596038,
        1533193853,
        1436311464,
        2012303432,
    ],
    [
        839997195,
        1225781098,
        2011967775,
        575084315,
        1309329169,
        786393545,
        995788880,
        1702925345,
        1444525226,
        908073383,
        1811535085,
        1531002367,
        1635653662,
        1585100155,
        867006515,
        879151050,
    ],
]

_INTERNAL_RC = [
    1423960925,
    2101391318,
    1915532054,
    275400051,
    1168624859,
    1141248885,
    356546469,
    1165250474,
    1320543726,
    932505663,
    1204226364,
    1452576828,
    1774936729,
    926808140,
    1184948056,
    1186493834,
    843181003,
    185193011,
    452207447,
    510054082,
]

_INTERNAL_DIAG = [
    2130706431,
    1,
    2,
    1065353217,
    3,
    4,
    1065353216,
    2130706430,
    2130706429,
    2122383361,
    1864368129,
    2130706306,
    8323072,
    266338304,
    133169152,
    127,
]

# The parameter bundle behind `KoalaBear16`. Public so a consumer can build a
# variant off it (`dataclasses.replace`) without transcribing the tables again,
# but deliberately NOT in the package's export table: `KoalaBear16` is the one
# name this set answers to at the root, and a second spelling there is exactly
# what `adapter/mgf1.py` argues against.
KOALABEAR16_PARAMS = Poseidon2Params(
    width=_WIDTH,
    dtype=_F,
    alpha=_ALPHA,
    external_rounds=_ER,
    internal_rounds=_IR,
    external_constants_initial=fnp.array(_EXTERNAL_INITIAL, dtype=_F),
    external_constants_terminal=fnp.array(_EXTERNAL_TERMINAL, dtype=_F),
    internal_constants=fnp.array(_INTERNAL_RC, dtype=_F),
    internal_diag=fnp.array(_INTERNAL_DIAG, dtype=_F),
)

# Plonky3's KoalaBear width-16 Poseidon2, ready to permute.
KoalaBear16 = Poseidon2(KOALABEAR16_PARAMS)

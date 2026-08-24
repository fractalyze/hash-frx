# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Standard Poseidon2 parameter sets, as named members.

`params.py` ships the *shape* — `Poseidon2Params` and `default_external_matrix`
— and this module ships *instances* of it: the parameterizations consumers
actually run, each pinned to the released implementation it reproduces. A set
is a claim about an external codebase, and a claim about the outside world
needs a name and a citation rather than a comment.

The repo's precedent for a shipped named set is
[`vision/params.py`](../vision/params.py)'s `vision_mark32_params` — same
package as the primitive, which is why this lives under `poseidon2/` rather
than under `adapter/` (that layer is constructions built over a finished hash:
HMAC, HKDF, PBKDF2, MGF1). It differs from Vision's on one axis: Vision takes
`dtype` as a parameter because its constants are dtype-independent, while a
Poseidon2 set's round constants are field elements, so the field is baked in
and this target carries the `zk_dtypes` dep that the engine deliberately does
not.

**What this fixes is a duplication that fails silently.** Before this module
the constants lived in each consumer's own tree, verified in exactly one place
(hash-frx's test fixture, against Plonky3) and used in several. A round-constant
table that is wrong does not crash; it produces a hash that is merely
*different*, and the difference surfaces as a Merkle root mismatch three layers
away. `testing/poseidon2_test.py`'s `test_permute_byte_matches_plonky3` runs
these parameters, so the constants and the assertion that pins them to Plonky3
now live in one repo instead of two.

**Which sets belong here.** The test is "is this set one scheme's?".
KoalaBear-16 below is a general Plonky3 parameterization, so it passes. A set
that is a particular prover's vendored instance does not, and stays with that
prover — the repo's application-agnosticism rule (README, "Two non-negotiables")
is not suspended for a constant table. Concretely: SP1's vendored KoalaBear-16,
which shares these round constants but carries a powers-of-two internal diagonal
and folds a Montgomery R^-1 into `internal_j_scale`, is *not* here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _params() -> Poseidon2Params:
    """The parameter bundle, built once and cached into `globals()` so that
    `KoalaBear16` and the `KOALABEAR16_PARAMS` export share one object."""
    cached = globals().get("KOALABEAR16_PARAMS")
    if cached is not None:
        return cached
    params = Poseidon2Params(
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
    globals()["KOALABEAR16_PARAMS"] = params
    return params


def __getattr__(name: str) -> Any:
    """Build `KOALABEAR16_PARAMS` / `KoalaBear16` on first access (PEP 562).

    Not module-scope values, for the reason `fusion.routing` gives: constructing
    a `Poseidon2` reads `frx.default_backend()` to pick its marker, so building
    one at import freezes the routing to whichever backend happened to be
    default when this module loaded. `keccak/sponge.py` declines a module-level
    `KeccakF1600()` on the same grounds. Deferring also keeps the cost off a
    consumer that only wants `KOALABEAR16_PLONKY3_COMMIT` — importing this
    module was 95 ms and a live device before, and is now free.

    Each binding is cached in `globals()`, so this runs once per name and the
    root package's own lazy re-export sees a plain attribute afterwards. A named
    instance still snapshots its routing at that first access; what this fixes
    is the snapshot happening at import, before a consumer has had any chance to
    select a backend.
    """
    if name == "KOALABEAR16_PARAMS":
        return _params()
    if name == "KoalaBear16":
        perm = Poseidon2(_params())
        globals()["KoalaBear16"] = perm
        return perm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # mypy cannot see through the runtime `__getattr__`
    KOALABEAR16_PARAMS: Poseidon2Params
    KoalaBear16: Poseidon2

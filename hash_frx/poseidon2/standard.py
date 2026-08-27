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
KoalaBear-16 and BabyBear-16 below are general Plonky3 parameterizations, so
they pass. A set that is a particular prover's vendored instance does not, and
stays with that prover — the repo's application-agnosticism rule (README, "Two
non-negotiables") is not suspended for a constant table. Concretely: SP1's
vendored KoalaBear-16, which shares these round constants but carries a
powers-of-two internal diagonal and folds a Montgomery R^-1 into
`internal_j_scale`, is *not* here.

That test also draws the line *inside* a consumer's file rather than around it.
openvm-zorch's `poseidon2/babybear16.py` held both kinds: the round constants
below, which are Plonky3's and belong here, and a rate-8 / digest-8 / arity-2
sponge and compression, which are OpenVM's Merkle shape and stay there. Moving
a set out of a consumer is not the same as emptying its file.

**The two sets are spelled differently on purpose.** KoalaBear-16's constants
are decimal and BabyBear-16's are hex, because each matches the literal form its
own Plonky3 source uses — hex for `baby-bear/src/poseidon2.rs`. A table of 141
constants is verifiable by a reviewer only if it diffs cleanly against the file
it was taken from, and re-basing one of them to match the other would trade that
away for a consistency nothing reads. `poseidon2_test` holds both to their
revisions regardless, which is what makes the spelling a review aid rather than
the guarantee.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import frx.numpy as fnp
from zk_dtypes import babybear_mont as _BB
from zk_dtypes import koalabear_mont as _KB

from hash_frx.poseidon2.params import Poseidon2Params
from hash_frx.poseidon2.poseidon2 import Poseidon2

# The revision these constants were generated from, as a value rather than a
# comment: "which Plonky3 is this" is then a question the package answers.
KOALABEAR16_PLONKY3_COMMIT = "4318eba062fd1cbca3dbe98904ad18ad950f3b49"

_KB_WIDTH, _KB_ER, _KB_IR, _KB_ALPHA = 16, 4, 20, 3

# Canonical-u32 constants from the pinned Plonky3 koala-bear poseidon2-16.
_KB_EXTERNAL_INITIAL = [
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

_KB_EXTERNAL_TERMINAL = [
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

_KB_INTERNAL_RC = [
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

_KB_INTERNAL_DIAG = [
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


def _koalabear16_params() -> Poseidon2Params:
    """KoalaBear-16's bundle. Built on demand and never at module scope; the
    caching that makes `KoalaBear16` and `KOALABEAR16_PARAMS` share one object
    belongs to `_SETS` below, so this function only says what the set IS."""
    return Poseidon2Params(
        width=_KB_WIDTH,
        dtype=_KB,
        alpha=_KB_ALPHA,
        external_rounds=_KB_ER,
        internal_rounds=_KB_IR,
        external_constants_initial=fnp.array(_KB_EXTERNAL_INITIAL, dtype=_KB),
        external_constants_terminal=fnp.array(_KB_EXTERNAL_TERMINAL, dtype=_KB),
        internal_constants=fnp.array(_KB_INTERNAL_RC, dtype=_KB),
        internal_diag=fnp.array(_KB_INTERNAL_DIAG, dtype=_KB),
    )


# ---------------------------------------------------------------------------
# BabyBear-16 — plonky3's `default_babybear_poseidon2_16`.
#
# The instance openvm-stark-backend hashes every proof byte with, and the set
# openvm-zorch carried in its own tree until this module took it. Its round
# constants are the HorizenLabs BabyBear instance
# (`BABYBEAR_RC16_EXTERNAL_INITIAL` / `_FINAL`, `BABYBEAR_RC16_INTERNAL`),
# written in the same hex as the plonky3 source so the arrays diff cleanly
# against it — the module docstring states why that is worth an inconsistency
# with KoalaBear-16's decimals.
#
# S-box x^7: the least D with gcd(p - 1, D) = 1 for p = 15 * 2^27 + 1, where
# KoalaBear takes x^3. The internal layer is plonky3's `1 + Diag(V)` family
# with the optimized vector
#   [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8,
#    -1/16, -1/2^27]
# reduced to canonical form in `_BB_INTERNAL_DIAG` below. Unlike SP1's
# vendored KoalaBear kernel there is no Montgomery factor folded into the
# layer, so `internal_j_scale` stays at its default 1 — which is also what
# makes this set general rather than one prover's.
# ---------------------------------------------------------------------------

# The revision these constants were generated from, as a value rather than a
# comment, the same way `KOALABEAR16_PLONKY3_COMMIT` above is. This is tag
# v0.4.3 — the pin openvm-stark-backend v2.0.0's reference carries.
BABYBEAR16_PLONKY3_COMMIT = "90008383a99bdcbf725c91c91efbdf6775da7054"

_BB_WIDTH, _BB_ER, _BB_IR, _BB_ALPHA = 16, 4, 13, 7

_BB_EXTERNAL_INITIAL = [
    [
        0x69CBB6AF,
        0x46AD93F9,
        0x60A00F4E,
        0x6B1297CD,
        0x23189AFE,
        0x732E7BEF,
        0x72C246DE,
        0x2C941900,
        0x0557EEDE,
        0x1580496F,
        0x3A3EA77B,
        0x54F3F271,
        0x0F49B029,
        0x47872FE1,
        0x221E2E36,
        0x1AB7202E,
    ],
    [
        0x487779A6,
        0x3851C9D8,
        0x38DC17C0,
        0x209F8849,
        0x268DCEE8,
        0x350C48DA,
        0x5B9AD32E,
        0x0523272B,
        0x3F89055B,
        0x01E894B2,
        0x13DDEDDE,
        0x1B2EF334,
        0x7507D8B4,
        0x6CEEB94E,
        0x52EB6BA2,
        0x50642905,
    ],
    [
        0x05453F3F,
        0x06349EFC,
        0x6922787C,
        0x04BFFF9C,
        0x768C714A,
        0x3E9FF21A,
        0x15737C9C,
        0x2229C807,
        0x0D47F88C,
        0x097E0ECC,
        0x27EADBA0,
        0x2D7D29E4,
        0x3502AAA0,
        0x0F475FD7,
        0x29FBDA49,
        0x018AFFFD,
    ],
    [
        0x0315B618,
        0x6D4497D1,
        0x1B171D9E,
        0x52861ABD,
        0x2E5D0501,
        0x3EC8646C,
        0x6E5F250A,
        0x148AE8E6,
        0x17F5FA4A,
        0x3E66D284,
        0x0051AA3B,
        0x483F7913,
        0x2CFE5F15,
        0x023427CA,
        0x2CC78315,
        0x1E36EA47,
    ],
]

_BB_EXTERNAL_TERMINAL = [
    [
        0x7290A80D,
        0x6F7E5329,
        0x598EC8A8,
        0x76A859A0,
        0x6559E868,
        0x657B83AF,
        0x13271D3F,
        0x1F876063,
        0x0AEEAE37,
        0x706E9CA6,
        0x46400CEE,
        0x72A05C26,
        0x2C589C9E,
        0x20BD37A7,
        0x6A2D3D10,
        0x20523767,
    ],
    [
        0x5B8FE9C4,
        0x2AA501D6,
        0x1E01AC3E,
        0x1448BC54,
        0x5CE5AD1C,
        0x4918A14D,
        0x2C46A83F,
        0x4FCF6876,
        0x61D8D5C8,
        0x6DDF4FF9,
        0x11FDA4D3,
        0x02933A8F,
        0x170EAF81,
        0x5A9C314F,
        0x49A12590,
        0x35EC52A1,
    ],
    [
        0x58EB1611,
        0x5E481E65,
        0x367125C9,
        0x0EBA33BA,
        0x1FC28DED,
        0x066399AD,
        0x0CBEC0EA,
        0x75FD1AF0,
        0x50F5BF4E,
        0x643D5F41,
        0x6F4FE718,
        0x5B3CBBDE,
        0x1E3AFB3E,
        0x296FB027,
        0x45E1547B,
        0x4A8DB2AB,
    ],
    [
        0x59986D19,
        0x30BCDFA3,
        0x1DB63932,
        0x1D7C2824,
        0x53B33681,
        0x0673B747,
        0x038A98A3,
        0x2C5BCE60,
        0x351979CD,
        0x5008FB73,
        0x547BCA78,
        0x711AF481,
        0x3F93BF64,
        0x644D987B,
        0x3C8BCD87,
        0x608758B8,
    ],
]

_BB_INTERNAL_RC = [
    0x5A8053C0,
    0x693BE639,
    0x3858867D,
    0x19334F6B,
    0x128F0FD8,
    0x4E2B1CCB,
    0x61210CE0,
    0x3C318939,
    0x0B5B2F22,
    0x2EDB11D5,
    0x213EFFDF,
    0x0CAC4606,
    0x241AF16D,
]

_BB_INTERNAL_DIAG = [
    0x77FFFFFF,
    0x00000001,
    0x00000002,
    0x3C000001,
    0x00000003,
    0x00000004,
    0x3C000000,
    0x77FFFFFE,
    0x77FFFFFD,
    0x77880001,
    0x5A000001,
    0x69000001,
    0x77FFFFF2,
    0x00780000,
    0x07800000,
    0x0000000F,
]


def _babybear16_params() -> Poseidon2Params:
    """BabyBear-16's bundle, the sibling of `_koalabear16_params` above and
    subject to the same rule: it says what the set is, and `_SETS` below owns
    when it runs and what caches the result."""
    return Poseidon2Params(
        width=_BB_WIDTH,
        dtype=_BB,
        alpha=_BB_ALPHA,
        external_rounds=_BB_ER,
        internal_rounds=_BB_IR,
        external_constants_initial=fnp.array(_BB_EXTERNAL_INITIAL, dtype=_BB),
        external_constants_terminal=fnp.array(_BB_EXTERNAL_TERMINAL, dtype=_BB),
        # One constant per round, lane 0 (`params.py`'s contract). The set
        # arrived here from a consumer that expanded these thirteen into a
        # (13, 16) matrix with fifteen structurally-zero columns per row; that
        # shape predates the contract and is not reproduced.
        internal_constants=fnp.array(_BB_INTERNAL_RC, dtype=_BB),
        internal_diag=fnp.array(_BB_INTERNAL_DIAG, dtype=_BB),
    )


# name -> what builds it. The table IS the list of shipped sets: a set joins by
# adding two rows, and `standard_test` sweeps this rather than restating each
# set's properties, so joining is not optional the way it is with a hand-kept
# second list (`testing/rows.py` states that argument; `markers.py` and this
# package's own `_EXPORTS` are the same shape one level up).
#
# **The values are thunks, and that is the whole design.** Nothing here
# constructs a `Poseidon2Params` or a `Poseidon2` — a module-scope table of
# BUILT objects would be exactly the import-time materialization `__getattr__`
# below exists to prevent. A table of callables is a module-scope value that
# builds nothing until a name is asked for, so the deferral rule constrains
# what this table holds, not whether it exists.
#
# A permutation's thunk reaches its parameters through `_member` rather than
# calling the builder a second time, which is what keeps a set's two names one
# object: `Poseidon2(_koalabear16_params())` here would build a second equal
# bundle, and two equal-but-distinct bundles are two jit cache keys for one
# parameterization (`params.py` states what that costs).
_SETS: dict[str, Callable[[], Any]] = {
    "KOALABEAR16_PARAMS": _koalabear16_params,
    "KoalaBear16": lambda: Poseidon2(_member("KOALABEAR16_PARAMS")),
    "BABYBEAR16_PARAMS": _babybear16_params,
    "BabyBear16": lambda: Poseidon2(_member("BABYBEAR16_PARAMS")),
}


def _member(name: str) -> Any:
    """`name`, built and cached on the first ask and read from the cache after.

    The one place a shipped set is built, which is the point: a per-set copy of
    this is the thing that cannot be trusted. A builder that wrote the wrong
    cache key would rebuild its tables on every access, forever, while
    producing identical values — nothing fails, and the only guard would be
    whether whoever added the set also remembered to assert it. Written once,
    a new set cannot get it wrong.

    Reads `globals()` rather than a private dict because that is where
    `__getattr__` must leave the binding anyway: Python consults the module
    dict first, so a cached name never reaches `__getattr__` a second time.
    """
    cached = globals().get(name)
    if cached is not None:
        return cached
    value = _SETS[name]()
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    """Build a named set on first access (PEP 562).

    Not module-scope values, for the reason `fusion.routing` gives: constructing
    a `Poseidon2` reads `frx.default_backend()` to pick its marker, so building
    one at import freezes the routing to whichever backend happened to be
    default when this module loaded. `keccak/sponge.py` declines a module-level
    `KeccakF1600()` on the same grounds. Deferring also keeps the cost off a
    consumer that only wants a `*_PLONKY3_COMMIT` citation — importing this
    module was 95 ms and a live device before, and is now free.

    Each binding is cached in `globals()` by `_member`, so this runs once per
    name and the root package's own lazy re-export sees a plain attribute
    afterwards. A named instance still snapshots its routing at that first
    access; what this fixes is the snapshot happening at import, before a
    consumer has had any chance to select a backend.

    A name outside `_SETS` raises rather than being invented, which is what
    keeps a typo from resolving to a permutation.
    """
    if name not in _SETS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _member(name)


def __dir__() -> list[str]:
    """The module's own names plus the sets `__getattr__` will build, so
    `dir()` and tab-completion see a set that has not been asked for yet — the
    root package's `__dir__` exists for the same reason."""
    return sorted({*globals(), *_SETS})


if TYPE_CHECKING:  # mypy cannot see through the runtime `__getattr__`
    KOALABEAR16_PARAMS: Poseidon2Params
    KoalaBear16: Poseidon2
    BABYBEAR16_PARAMS: Poseidon2Params
    BabyBear16: Poseidon2

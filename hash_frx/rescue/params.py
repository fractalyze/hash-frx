# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""RescueParams — the fully-free parameter surface, and the RPO-128 tables.

Rescue is the prime-field Marvellous permutation family
(Aly-Ashur-Ben-Sasson-Dhooghe-Szepieniec, IACR ToSC 2020(3),
https://eprint.iacr.org/2019/426): rounds alternate the power S-box `x**alpha`
with its inverse `x**inv_alpha`, around an MDS layer and injected round
constants. It is a generator over a prime field, a width, an alpha and a
constant schedule — no enumerated member list — so, like Poseidon2 and Vision,
the parameterization is one value-compared object and `Rescue(params)` takes
it whole (docs/reference/conventions.md, "a parameter per choice").

The core treats `dtype` as opaque: any dtype whose `*` is the field
multiplication satisfies it. That is why `inv_alpha` is STORED rather than
derived — deriving `alpha**-1 mod (p - 1)` needs `p`, which the core cannot
read; `testing/params_test.py` derives it from `alpha` and the field size and
pins the shipped value.

`rescue_rpo128_params` is the shipped instance: **Rescue-Prime Optimized** at
the 128-bit level (Ashur-Kindi-Meier-Szepieniec-Threadbare, "Rescue-Prime
Optimized", https://eprint.iacr.org/2022/1577), Miden VM's native hash — p = 2^64 - 2^32
+ 1 (Goldilocks), m = 12, N = 7 rounds, alpha = 7. RPO won the Task 0
anchoring gate on fractalyze/hash-frx#190: its 19 published digests pin every
table end to end, where the vanilla SoK Rescue-Prime (eprint 2020/1143)
publishes a generator and no vectors. RPO's **half-round order differs** from
the SoK's — the structural consequence lives in `rescue.py`, not here: this
surface carries both families unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
import numpy as np
from frx import Array


@dataclass(frozen=True)
class RescueParams:
    """Fully-free parameter surface of a Rescue permutation.

    Contract (validated in __post_init__):
      rounds : N >= 1; a round is two half-rounds (the alpha half, then the
          inverse-alpha half).
      alpha : positive S-box exponent; caller guarantees gcd(alpha, p-1) == 1
          (the core does not know p, so it cannot check).
      inv_alpha : positive inverse exponent; caller guarantees it is
          alpha^-1 mod (p - 1). Stored, not derived — the core treats `dtype`
          as opaque and cannot read p; `testing/params_test.py` pins the
          shipped value against the derivation.
      mds : (width, width) over dtype; applied as `mds @ state`.
      round_constants : (2*rounds, width) over dtype; row 2r is injected in
          round r's first half (before the alpha S-box), row 2r+1 in its
          second (before the inverse one) — the flat 2mN list the spec
          derives, reshaped one row per half-round.
    """

    width: int
    dtype: Any
    rounds: int
    alpha: int
    inv_alpha: int
    mds: Array
    round_constants: Array

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError(f"rounds must be a positive int, got {self.rounds}")
        if self.alpha < 1:
            raise ValueError(f"alpha must be a positive int, got {self.alpha}")
        if self.inv_alpha < 1:
            raise ValueError(f"inv_alpha must be a positive int, got {self.inv_alpha}")
        w = self.width
        # One pass over `_ARRAY_FIELDS` so a new array field cannot silently
        # skip either the shape or the dtype check.
        want_shapes: dict[str, tuple[int, ...]] = {
            "mds": (w, w),
            "round_constants": (2 * self.rounds, w),
        }
        for name in self._ARRAY_FIELDS:
            arr = getattr(self, name)
            got = tuple(np.shape(arr))
            want = want_shapes[name]
            if got != want:
                raise ValueError(f"{name}: expected shape {want}, got {got}")
            if arr.dtype != self.dtype:
                raise ValueError(
                    f"{name}: expected dtype {self.dtype}, got {arr.dtype}"
                )

    # Value equality/hash: a permutation rides pytree aux, which must compare
    # by value — identity equality re-traces the enclosing jit zone per freshly
    # built instance (docs/reference/conventions.md "Pytree registration"). The
    # dataclass-derived __eq__ is unusable anyway: `==` on the Array fields is
    # elementwise. One cached host-side key serves both methods, as
    # `Poseidon2Params` lays out (jit dispatch calls __eq__ per call, so a live
    # device-array comparison there would sync per dispatch). Also the field
    # list `__post_init__` validates — one tuple, so the two sweeps cannot
    # drift apart.
    _ARRAY_FIELDS = ("mds", "round_constants")

    def _value_key(self) -> tuple:
        k = self.__dict__.get("_key")
        if k is None:
            k = (
                self.width,
                self.dtype,
                self.rounds,
                self.alpha,
                self.inv_alpha,
            ) + tuple(
                np.asarray(getattr(self, f)).tobytes() for f in self._ARRAY_FIELDS
            )
            object.__setattr__(self, "_key", k)
        return k

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, RescueParams):
            return NotImplemented
        return self._value_key() == other._value_key()

    def __hash__(self) -> int:
        # Memoized like `_key`: the permute jit zone hashes the params on every
        # dispatch, and CPython caches neither tuple nor bytes hashes, so a
        # bare hash(key) would re-SipHash the table bytes per permute call.
        h = self.__dict__.get("_hash")
        if h is None:
            h = hash(self._value_key())
            object.__setattr__(self, "_hash", h)
        return h


def rescue_rpo128_params(dtype: Any) -> RescueParams:
    """The RPO-128 instance over `dtype` — the shipped member (Miden's hash).

    `dtype` must be the 64-bit Goldilocks prime field p = 2^64 - 2^32 + 1 —
    `zk_dtypes.goldilocks_mont` — for the tables below to mean what RPO-128
    means; it is a parameter only because the core names no dtype package (the
    same field could be registered elsewhere). Taking a different field here
    would be a different (unanalyzed) hash, which is why the tables are not
    parameters of this factory.

    m = 12 state elements (the paper's sponge reads capacity in lanes 0..3 and
    rate in 4..11 — a consumer concern, not carried here), N = 7 rounds,
    alpha = 7, inv_alpha = 7^-1 mod (p-1), per the RPO paper
    (https://eprint.iacr.org/2022/1577, Section 2, Table 1).
    """
    return RescueParams(
        width=12,
        dtype=dtype,
        rounds=7,
        alpha=7,
        # 7^-1 mod (p - 1), printed in the RPO paper's Section 2.1. Note it
        # exceeds 2^63 - 1: it fits neither a signed i64 marker attribute nor
        # anything narrower than the Python int it rides here.
        inv_alpha=10540996611094048183,
        mds=fnp.array(
            [[_MDS_ROW[(j - i) % 12] for j in range(12)] for i in range(12)],
            dtype=dtype,
        ),
        round_constants=fnp.array(_ROUND_CONSTANTS, dtype=dtype),
    )


# The RPO-128 tables. The MDS is circulant (RPO paper Section 2.3): row i is
# the published first row rotated right i times, M[i][j] = row[(j - i) mod m]
# — the factory materializes the full matrix because the parameter surface is
# a free (width, width) array, not a circulant family. `_ROUND_CONSTANTS` is
# the Section 2.2 schedule — SHAKE256("RPO(p,m,c,level)") expanded to 2mN
# 9-byte little-endian chunks reduced mod p — transcribed from the production
# tables in 0xMiden/crypto (miden-crypto, MIT)
# src/hash/rescue/mod.rs @ v0.9.0 (ARK1/ARK2, interleaved one row per
# half-round). `testing/reference_test.py` anchors both end to end: it
# re-derives every constant from the SHAKE256 procedure and holds the
# permutation over these tables to the 19 digests published in the paper's
# Section 3.1.
# fmt: off
_MDS_ROW = (7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8)

_ROUND_CONSTANTS = (
    (
        5789762306288267392, 6522564764413701783, 17809893479458208203,
        107145243989736508, 6388978042437517382, 15844067734406016715,
        9975000513555218239, 3344984123768313364, 9959189626657347191,
        12960773468763563665, 9602914297752488475, 16657542370200465908,
    ),
    (
        6077062762357204287, 15277620170502011191, 5358738125714196705,
        14233283787297595718, 13792579614346651365, 11614812331536767105,
        14871063686742261166, 10148237148793043499, 4457428952329675767,
        15590786458219172475, 10063319113072092615, 14200078843431360086,
    ),
    (
        12987190162843096997, 653957632802705281, 4441654670647621225,
        4038207883745915761, 5613464648874830118, 13222989726778338773,
        3037761201230264149, 16683759727265180203, 8337364536491240715,
        3227397518293416448, 8110510111539674682, 2872078294163232137,
    ),
    (
        6202948458916099932, 17690140365333231091, 3595001575307484651,
        373995945117666487, 1235734395091296013, 14172757457833931602,
        707573103686350224, 15453217512188187135, 219777875004506018,
        17876696346199469008, 17731621626449383378, 2897136237748376248,
    ),
    (
        18072785500942327487, 6200974112677013481, 17682092219085884187,
        10599526828986756440, 975003873302957338, 8264241093196931281,
        10065763900435475170, 2181131744534710197, 6317303992309418647,
        1401440938888741532, 8884468225181997494, 13066900325715521532,
    ),
    (
        8023374565629191455, 15013690343205953430, 4485500052507912973,
        12489737547229155153, 9500452585969030576, 2054001340201038870,
        12420704059284934186, 355990932618543755, 9071225051243523860,
        12766199826003448536, 9045979173463556963, 12934431667190679898,
    ),
    (
        5674685213610121970, 5759084860419474071, 13943282657648897737,
        1352748651966375394, 17110913224029905221, 1003883795902368422,
        4141870621881018291, 8121410972417424656, 14300518605864919529,
        13712227150607670181, 17021852944633065291, 6252096473787587650,
    ),
    (
        18389244934624494276, 16731736864863925227, 4440209734760478192,
        17208448209698888938, 8739495587021565984, 17000774922218161967,
        13533282547195532087, 525402848358706231, 16987541523062161972,
        5466806524462797102, 14512769585918244983, 10973956031244051118,
    ),
    (
        4887609836208846458, 3027115137917284492, 9595098600469470675,
        10528569829048484079, 7864689113198939815, 17533723827845969040,
        5781638039037710951, 17024078752430719006, 109659393484013511,
        7158933660534805869, 2955076958026921730, 7433723648458773977,
    ),
    (
        6982293561042362913, 14065426295947720331, 16451845770444974180,
        7139138592091306727, 9012006439959783127, 14619614108529063361,
        1394813199588124371, 4635111139507788575, 16217473952264203365,
        10782018226466330683, 6844229992533662050, 7446486531695178711,
    ),
    (
        16308865189192447297, 11977192855656444890, 12532242556065780287,
        14594890931430968898, 7291784239689209784, 5514718540551361949,
        10025733853830934803, 7293794580341021693, 6728552937464861756,
        6332385040983343262, 13277683694236792804, 2600778905124452676,
    ),
    (
        3736792340494631448, 577852220195055341, 6689998335515779805,
        13886063479078013492, 14358505101923202168, 7744142531772274164,
        16135070735728404443, 12290902521256031137, 12059913662657709804,
        16456018495793751911, 4571485474751953524, 17200392109565783176,
    ),
    (
        7123075680859040534, 1034205548717903090, 7717824418247931797,
        3019070937878604058, 11403792746066867460, 10280580802233112374,
        337153209462421218, 13333398568519923717, 3596153696935337464,
        8104208463525993784, 14345062289456085693, 17036731477169661256,
    ),
    (
        17130398059294018733, 519782857322261988, 9625384390925085478,
        1664893052631119222, 7629576092524553570, 3485239601103661425,
        9755891797164033838, 15218148195153269027, 16460604813734957368,
        9643968136937729763, 3611348709641382851, 18256379591337759196,
    ),
)
# fmt: on

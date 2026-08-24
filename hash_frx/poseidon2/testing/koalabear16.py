"""koalabear-16 test-side companions to the shipped `KoalaBear16` set.

The constants themselves are no longer here: they ship from
[`poseidon2/standard.py`](../standard.py), and this module holds only what a
test needs and a consumer does not — the Plonky3 output vector, the expected
marker text, and the non-identity-scale variant. Everything below runs the
*shipped* parameters, so a drift between what is tested and what is published
cannot open up.
"""

from __future__ import annotations

from dataclasses import replace

import frx.numpy as fnp
from zk_dtypes import koalabear_mont as F

from hash_frx.poseidon2.params import Poseidon2Params
from hash_frx.poseidon2.poseidon2 import Poseidon2
from hash_frx.poseidon2.standard import KOALABEAR16_PARAMS

# Literals, deliberately not read off `KOALABEAR16_PARAMS`: these build the
# expected marker text, and an expectation derived from the object under test
# follows it silently when it changes.
_WIDTH, _ER, _IR, _ALPHA = 16, 4, 20, 3

# This parameterization's marker metadata as StableHLO prints it (dict keys
# alphabetical) — shared by the emission contract test and the vmap marker
# survival test so the expected text lives once. `external_m4` is the base M4
# (Plonky3's circ(2,3,1,1)) flattened row-major, which the emitter applies per
# 4-block.
# The default (identity) internal_j_scale carries its canonical value (1); the
# recognizer value-encodes it per field.
KOALABEAR16_POSEIDON2_ATTRS = (
    f"composite_attributes = {{alpha = {_ALPHA} : i64,"
    " external_m4 = dense<[2, 3, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 3, 1, 1, 2]> :"
    " tensor<16xi64>,"
    f" external_rounds = {_ER} : i64,"
    " internal_j_scale = 1 : i64,"
    f" internal_rounds = {_IR} : i64, width = {_WIDTH} : i64}}"
)

# The Plonky3 output for `permute(arange(16))` — the vector the shipped set is
# held to. A test vector rather than a parameter, so it stays on this side.
KOALABEAR16_EXPECTED = fnp.array(
    [
        1259554834,
        663463928,
        1989430097,
        476523442,
        836740795,
        1803459961,
        1229318262,
        2023956904,
        2054405130,
        1556655036,
        1455339712,
        1471465890,
        423337459,
        353979748,
        1203410294,
        1592576868,
    ],
    dtype=F,
)


def koalabear16_params() -> Poseidon2Params:
    """The shipped set's parameters."""
    return KOALABEAR16_PARAMS


def koalabear16_perm() -> Poseidon2:
    """A fresh permutation over the shipped koalabear-16 parameters.

    Fresh per call, not the module-level `KoalaBear16`: several tests build two
    and require them to compare equal by value (the static jit key) or to see a
    backend mock installed after import. Handing back one shared instance makes
    those pass on identity instead, which is the regression they exist to catch.
    """
    return Poseidon2(KOALABEAR16_PARAMS)


def koalabear16_scaled_perm() -> Poseidon2:
    """The golden instance with a non-identity `internal_j_scale`.

    The default instance's identity scale hides an entire bug class: a
    lowering that silently substitutes identity for the J term's scale (or
    re-encodes its Montgomery storage) is byte-invisible when the true scale
    is already one. The value here is R⁻¹ mod p — its Montgomery STORAGE is
    exactly 1, which is the trap a raw-bits/canonical mixup lands on — and it
    mirrors a consumer folding R⁻¹ out of an `R⁻¹·M·state` internal layer.
    """
    return Poseidon2(
        replace(koalabear16_params(), internal_j_scale=fnp.array(1057030144, F))
    )

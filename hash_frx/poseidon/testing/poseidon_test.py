"""Classic Poseidon: byte-matches a pure-Python reference and emits its marker.

A small width-3 config over a 31-bit field is enough to pin the round structure
(full/partial split, ARC every round, dense MDS every round, S-box on all vs the
last lane) against an independent numpy/Python reference, and to read the marker
the dedicated emitter consumes off the lowered StableHLO.
"""

from __future__ import annotations

import dataclasses

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from zk_dtypes import babybear_mont as F
from zk_dtypes import goldilocks_mont, pfinfo

from hash_frx.fusion import FusionPath
from hash_frx.permutation import Permutation
from hash_frx.poseidon.params import PoseidonParams
from hash_frx.poseidon.poseidon import (
    POSEIDON_MARKER,
    POSEIDON_MARKER_VERSION,
    Poseidon,
)
from hash_frx.testing.marker_recognized import assert_marker_recognized
from hash_frx.testing.marker_seam import assert_marker_matches_emission

_P = pfinfo(F).modulus  # field prime; canonical-int reference reduces mod this.

# A small width-3 config. alpha=7 is coprime to p-1 for this field, so the
# S-box is a permutation; the MDS is a small-int MDS matrix.
_WIDTH, _FULL, _PARTIAL, _ALPHA = 3, 2, 1, 7
_MDS = ((2, 3, 1), (1, 2, 3), (3, 1, 2))
# One full-width ARC vector per round (full_rounds + partial_rounds rows).
_ROUND_CONSTANTS = (
    (5, 7, 11),
    (13, 17, 19),  # the single partial round
    (23, 29, 31),
)

# Marker metadata as StableHLO prints it (dict keys alphabetical). `mds` is the
# 3x3 MDS flattened row-major, a numpy int64 value so it lowers to a
# DenseElementsAttr the XLA recognizer parses (GetCompositeAttrIntArray).
EXPECTED_ATTRS = (
    f"composite_attributes = {{alpha = {_ALPHA} : i64,"
    f" full_rounds = {_FULL} : i64,"
    " mds = dense<[2, 3, 1, 1, 2, 3, 3, 1, 2]> : tensor<9xi64>,"
    f" partial_rounds = {_PARTIAL} : i64, width = {_WIDTH} : i64}}"
)


def _to_field(canon: np.ndarray) -> fnp.ndarray:
    """Canonical ints -> field array (the dtype cast Montgomery-encodes)."""
    return fnp.asarray(canon.astype(np.int64).astype(F))


def _to_canon(arr: fnp.ndarray) -> np.ndarray:
    """Field array -> canonical ints (numpy object cast Montgomery-decodes,
    no frx x64 needed)."""
    return np.asarray(np.asarray(arr).astype(object), dtype=object)


def _poseidon_params() -> PoseidonParams:
    rc = np.array(_ROUND_CONSTANTS, dtype=np.int64)
    mds = np.array(_MDS, dtype=np.int64)
    return PoseidonParams(
        width=_WIDTH,
        dtype=F,
        alpha=_ALPHA,
        full_rounds=_FULL,
        partial_rounds=_PARTIAL,
        round_constants=_to_field(rc),
        mds=_to_field(mds),
    )


def _reference_permute(
    state_canon: list[int], mds_rows: tuple[tuple[int, ...], ...] = _MDS
) -> list[int]:
    """Independent classic-Poseidon reference in pure-Python int arithmetic mod p.

    Each round: ARC (add the round's full-width constants) -> S-box (x^alpha on
    all lanes in a full round, on the last lane only in a partial round) -> dense
    MDS (mds @ state). Rounds split full_rounds/2 full, partial_rounds partial,
    full_rounds/2 full; the dense MDS runs every round.
    """
    p, w, alpha = _P, _WIDTH, _ALPHA
    s = [x % p for x in state_canon]
    rounds = [list(r) for r in _ROUND_CONSTANTS]
    half_full = _FULL // 2

    def sbox(x: int) -> int:
        return pow(x % p, alpha, p)

    def mds(vec: list[int]) -> list[int]:
        return [sum(mds_rows[i][j] * vec[j] for j in range(w)) % p for i in range(w)]

    r = 0
    for _ in range(half_full):  # initial full rounds
        s = [(s[i] + rounds[r][i]) % p for i in range(w)]
        s = [sbox(x) for x in s]
        s = mds(s)
        r += 1
    for _ in range(_PARTIAL):  # partial rounds: S-box on the last lane only
        s = [(s[i] + rounds[r][i]) % p for i in range(w)]
        s[w - 1] = sbox(s[w - 1])
        s = mds(s)
        r += 1
    for _ in range(half_full):  # terminal full rounds
        s = [(s[i] + rounds[r][i]) % p for i in range(w)]
        s = [sbox(x) for x in s]
        s = mds(s)
        r += 1
    return s


class PoseidonReferenceByteMatchTest(absltest.TestCase):
    """Poseidon.permute byte-matches an independent pure-Python reference over
    random inputs (run eagerly on CPU)."""

    def test_byte_matches_reference(self) -> None:
        p = Poseidon(_poseidon_params())
        rng = np.random.default_rng(0)
        for _ in range(8):
            canon = rng.integers(0, _P, size=_WIDTH, dtype=np.int64)
            state = _to_field(canon)
            out = p.permute(state)
            got = [int(x) for x in _to_canon(out)]
            want = _reference_permute([int(x) for x in canon])
            self.assertEqual(got, want)


# An MDS whose entries sit above the dedicated emitter's add-chain bound. Any
# real MDS over a 31-bit field looks like this — the small-int matrix above is
# the exception, not the rule.
_WIDE_MDS = ((2, 3, 1), (1, 64, 3), (3, 1, _P - 1))
# A matrix the emitter also rejects for a second reason: an MDS has no zero row.
_ZERO_ROW_MDS = ((2, 3, 1), (0, 0, 0), (3, 1, 2))
_GOLDILOCKS_P = pfinfo(goldilocks_mont).modulus


def _to_goldilocks_field(rows: tuple[tuple[int, ...], ...]) -> fnp.ndarray:
    """Canonical ints -> Goldilocks. Unsigned, because an entry near `p` does not
    fit the int64 `_to_field` above casts through — which is the point."""
    return fnp.asarray(np.array(rows, dtype=np.uint64).astype(goldilocks_mont))


def _goldilocks_params() -> PoseidonParams:
    """The same width-3 shape over Goldilocks, with an MDS entry past
    `2**63 - 1` — #117's trigger, and the only way to reach it: a 31-bit field
    cannot hold a value that large, so `_WIDE_MDS` above cannot stand in.

    What this pins is an *ordering*, not a second rejection branch — it takes the
    same `[0, 64)` exit `_WIDE_MDS` does. The gate reads canonical Python ints
    and runs before `_poseidon_marker_attrs` casts them to int64, so #117's
    `OverflowError` is unreachable. Compute those attributes any earlier and this
    case raises while every other test still passes.

    `alpha` is 7 rather than 3 because 3 divides `p - 1`, and an S-box has to
    permute.
    """
    return PoseidonParams(
        width=_WIDTH,
        dtype=goldilocks_mont,
        alpha=7,
        full_rounds=_FULL,
        partial_rounds=_PARTIAL,
        round_constants=_to_goldilocks_field(_ROUND_CONSTANTS),
        mds=_to_goldilocks_field(((2, 3, 1), (1, 2, 3), (3, 1, _GOLDILOCKS_P - 1))),
    )


def _poseidon_params_with(mds_rows: tuple[tuple[int, ...], ...]) -> PoseidonParams:
    params = _poseidon_params()
    return dataclasses.replace(
        params, mds=_to_field(np.array(mds_rows, dtype=np.int64))
    )


class PoseidonUnroutableMdsTest(parameterized.TestCase):
    """A parameter set the dedicated emitter cannot express takes the generic
    marker rather than failing the compile, which is what routing one to it used
    to cost: an `unparsable composite.attributes` error naming a marker that was
    never malformed.

    `_poseidon_params()`'s own small matrix is the other side of the gate — the
    marker-emission tests below still pin it to the dedicated path — so this is a
    question the MDS answers rather than a blanket no.
    """

    @parameterized.named_parameters(
        ("entry_at_the_bound", _WIDE_MDS),
        ("all_zero_row", _ZERO_ROW_MDS),
    )
    def test_it_emits_the_generic_marker(
        self, mds_rows: tuple[tuple[int, ...], ...]
    ) -> None:
        p = Poseidon(_poseidon_params_with(mds_rows))

        self.assertIs(p.fusion_path, FusionPath.GENERIC)
        # The wire contract, not only the Python-side path: `fusion_path` staying
        # right while the emitted marker drifts is the failure this catches.
        assert_marker_matches_emission(self, p, fnp.arange(_WIDTH, dtype=F))

    def test_an_entry_wider_than_the_marker_attribute_also_falls_back(self) -> None:
        p = Poseidon(_goldilocks_params())

        self.assertIs(p.fusion_path, FusionPath.GENERIC)
        assert_marker_matches_emission(
            self, p, fnp.arange(_WIDTH, dtype=goldilocks_mont)
        )

    def test_the_generic_path_still_byte_matches_the_reference(self) -> None:
        p = Poseidon(_poseidon_params_with(_WIDE_MDS))
        rng = np.random.default_rng(1)

        for _ in range(4):
            canon = rng.integers(0, _P, size=_WIDTH, dtype=np.int64)
            out = p.permute(_to_field(canon))

            self.assertEqual(
                [int(x) for x in _to_canon(out)],
                _reference_permute([int(x) for x in canon], _WIDE_MDS),
            )


class PoseidonPermuteShapeTest(absltest.TestCase):
    def test_is_a_permutation(self) -> None:
        p = Poseidon(_poseidon_params())
        self.assertIsInstance(p, Permutation)
        self.assertEqual(p.width, _WIDTH)
        self.assertEqual(p.dtype, F)
        self.assertIs(p.fusion_path, FusionPath.DEDICATED)

    def test_permute_shape_and_vmap(self) -> None:
        p = Poseidon(_poseidon_params())
        x = fnp.arange(_WIDTH, dtype=F)
        out = p.permute(x)
        self.assertEqual(out.shape, (_WIDTH,))
        self.assertEqual(out.dtype, F)
        batch = fnp.stack([x, x + F(1)])
        bout = frx.vmap(p.permute)(batch)  # thread-per-hash
        self.assertEqual(bout.shape, (2, _WIDTH))
        self.assertEqual(bout.dtype, F)
        self.assertTrue(bool(fnp.array_equal(bout[0], out)))

    def test_permute_rejects_wrong_shape(self) -> None:
        p = Poseidon(_poseidon_params())
        with self.assertRaises(ValueError):
            p.permute(fnp.zeros((_WIDTH + 1,), dtype=F))  # width mismatch
        with self.assertRaises(ValueError):
            p.permute(fnp.zeros((2, _WIDTH), dtype=F))  # batched, not a 1-D state


class PoseidonMarkerEmissionTest(absltest.TestCase):
    def test_permute_emits_poseidon_named_composite(self) -> None:
        # The permute marks its region "hash_frx.perm.poseidon" so XLA routes it to the
        # dedicated Poseidon emitter; the permutation shape rides as
        # composite.attributes — all four ints plus the flat MDS are required by
        # the XLA recognizer.
        p = Poseidon(_poseidon_params())
        txt = frx.jit(p.permute).lower(fnp.arange(_WIDTH, dtype=F)).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn(f'"{POSEIDON_MARKER}"', composite_line)
        self.assertIn(EXPECTED_ATTRS, composite_line)
        self.assertIn(f"version = {POSEIDON_MARKER_VERSION}", composite_line)
        # Exactly the 2 ABI operands: the closed-over MDS must stay inline in
        # the decomposition (frx#218), never a leading operand.
        operands = composite_line.split(f'"{POSEIDON_MARKER}"')[1].split("{")[0]
        self.assertEqual(operands.count("%"), 2, composite_line)

    def test_mds_serializes_as_dense_i64_tensor(self) -> None:
        # The mds attribute must lower to a DenseElementsAttr
        # (`dense<[..]> : tensor<Nxi64>`), the form the XLA recognizer reads via
        # GetCompositeAttrIntArray — NOT a plain ArrayAttr (`mds = [..]`), which a
        # Python list/tuple would produce.
        p = Poseidon(_poseidon_params())
        txt = frx.jit(p.permute).lower(fnp.arange(_WIDTH, dtype=F)).as_text()
        self.assertIn("mds = dense<[2, 3, 1, 1, 2, 3, 3, 1, 2]> : tensor<9xi64>", txt)

    def test_marker_is_recognized_by_the_pinned_toolchain(self) -> None:
        p = Poseidon(_poseidon_params())
        assert_marker_recognized(
            self, "poseidon", p.permute, fnp.arange(p.width, dtype=F)
        )

    def test_seam_marker_matches_the_emission(self) -> None:
        p = Poseidon(_poseidon_params())
        assert_marker_matches_emission(self, p, fnp.arange(p.width, dtype=F))


if __name__ == "__main__":
    absltest.main()

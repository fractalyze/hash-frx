# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ascon-p[12] over uint32 word halves — values, seam conformance, and shape.

Values are checked against `reference.py`, which `reference_test` anchors to the
SP 800-232 KAT vectors, so agreement here means agreement with the standard
rather than with a second copy of one misreading. The bitsliced S-box circuit
keeps its own exhaustive case: it is the one component whose frx spelling shares
*nothing* with the oracle's table, and a wrong gate corrupts every digest
identically on both jit legs.

The lowering assertions are the half values cannot see. A missing marker, a
reduction, a gather, or a call all still compute the right bytes and only cost
the kernel.

**The two shapes must agree**, and one case here is only about that: the seam's
unbatched `(10,)` state and the `[B, 5]` word grids `ascon.ascon_hash256_bytes`
absorbs into run the *same* round body, so the interleave law
(word `i` at elements `2i` / `2i + 1`) is what keeps a permute reached through
the seam and one reached through the digest from being two different functions.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array

from hash_frx.ascon import permutation as permutation_mod
from hash_frx.ascon.permutation import (
    ASCON_P_MARKER,
    ASCON_P_MARKER_VERSION,
    ROUNDS,
    WIDTH,
    WORDS,
    AsconP,
    _abi_operands,
    _pack,
    _permute_body,
    _rounds,
    _substitution,
    _unpack,
    masks,
)
from hash_frx.ascon.testing.reference import INITIAL_STATE, SBOX
from hash_frx.ascon.testing.reference import permutation as reference_permutation
from hash_frx.fusion import FUSED_REGION_MARKER, FusionPath
from hash_frx.permutation import Permutation
from hash_frx.testing.composite_eqn import (
    composite_attrs,
    composite_eqn,
)
from hash_frx.testing.fusion_ready import assert_fusion_ready
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.testing.marker_seam import assert_marker_matches_emission
from hash_frx.word import split

_STATES: dict[str, tuple[int, ...]] = {
    # All-zero, the degenerate input every step must still move.
    "zeros": (0, 0, 0, 0, 0),
    # Ascon-Hash256's own initial state, so one case runs the words the shipped
    # hash actually permutes. Taken from the oracle, which DERIVES it by
    # permuting IV ‖ 0^256 (`reference.INITIAL_STATE`) and is anchored to SP
    # 800-232 Table 12 in `reference_test` — a third hand-transcription here
    # could disagree with both and still pass every case in this file.
    "hash256_init": INITIAL_STATE,
    # Every byte position distinct, so a misplaced half or a wrong-way rotation
    # cannot cancel out.
    "counter": (
        0x0001020304050607,
        0x08090A0B0C0D0E0F,
        0x1011121314151617,
        0x18191A1B1C1D1E1F,
        0x2021222324252627,
    ),
}


def _to_state(words: tuple[int, ...]) -> np.ndarray:
    """Five 64-bit words to the seam's `(10,)` interleaved halves."""
    out = np.zeros(WIDTH, dtype=np.uint32)
    for i, w in enumerate(words):
        lo, hi = split(w)
        out[2 * i], out[2 * i + 1] = lo, hi
    return out


def _from_state(state: np.ndarray) -> tuple[int, ...]:
    """The seam's `(10,)` interleaved halves back to five 64-bit words."""
    return tuple(
        int(state[2 * i]) | (int(state[2 * i + 1]) << 32) for i in range(WORDS)
    )


def _device_state(words: tuple[int, ...]) -> Array:
    return fnp.asarray(_to_state(words))


@contextlib.contextmanager
def _dedicated_emitter() -> Iterator[None]:
    """A leg where the pin and the backend both carry the emitter — the routing
    no plugin ships yet, constructed so the dedicated arm is testable before it
    exists (the Vision arrangement)."""
    with (
        mock.patch.object(permutation_mod, "_DEDICATED_EMITTER_AVAILABLE", True),
        mock.patch.object(
            permutation_mod, "_EMITTER_BACKENDS", (frx.default_backend(),)
        ),
    ):
        yield


@contextlib.contextmanager
def _generic_emitter() -> Iterator[None]:
    with mock.patch.object(permutation_mod, "_DEDICATED_EMITTER_AVAILABLE", False):
        yield


class AsconPTest(absltest.TestCase):
    def test_satisfies_the_permutation_seam(self) -> None:
        perm = AsconP()
        self.assertIsInstance(perm, Permutation)
        self.assertEqual(perm.width, WIDTH)
        self.assertEqual(perm.dtype, fnp.uint32)

    def test_value_equality_across_fresh_instances(self) -> None:
        # The pytree-aux contract: identity equality re-traces the enclosing jit
        # zone on every freshly built instance, which does not error.
        self.assertEqual(AsconP(), AsconP())
        self.assertEqual(hash(AsconP()), hash(AsconP()))

    def test_matches_the_reference_permutation(self) -> None:
        for label, words in _STATES.items():
            with self.subTest(state=label):
                got = np.asarray(AsconP().permute(_device_state(words)))
                self.assertEqual(
                    _from_state(got), tuple(reference_permutation(list(words)))
                )

    def test_jit_matches_eager(self) -> None:
        state = _device_state(_STATES["counter"])
        np.testing.assert_array_equal(
            np.asarray(frx.jit(AsconP().permute)(state)),
            np.asarray(AsconP().permute(state)),
        )

    def test_batches_under_vmap(self) -> None:
        # The seam promises a batched call lowers through the same body, so the
        # rows must agree with the same permutation applied one at a time.
        rows = np.stack([_to_state(w) for w in _STATES.values()])
        got = np.asarray(frx.vmap(AsconP().permute)(fnp.asarray(rows)))
        for i, words in enumerate(_STATES.values()):
            self.assertEqual(
                _from_state(got[i]), tuple(reference_permutation(list(words)))
            )

    def test_rejects_a_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            AsconP().permute(fnp.zeros(WIDTH + 1, dtype=fnp.uint32))
        # A `[B, 10]` batch is `frx.vmap`'s job, not a state.
        with self.assertRaises(ValueError):
            AsconP().permute(fnp.zeros((2, WIDTH), dtype=fnp.uint32))

    def test_rejects_a_wrong_dtype(self) -> None:
        # A word is two uint32 halves; a caller reaching for one 64-bit word is
        # the `keccak/lane.py` mistake and gets told so rather than truncated.
        with self.assertRaises(TypeError):
            AsconP().permute(fnp.zeros(WIDTH, dtype=fnp.int32))

    def test_the_body_traces_once_across_instances(self) -> None:
        state = _device_state(_STATES["zeros"])
        assert_single_trace(
            self,
            _permute_body,
            [lambda: AsconP().permute(state) for _ in range(3)],
        )


class AsconPStateLayoutTest(absltest.TestCase):
    """The interleave law, which is what lets one round body serve both shapes."""

    def test_unpack_and_pack_round_trip(self) -> None:
        state = _device_state(_STATES["counter"])
        lo, hi = _unpack(state)
        self.assertEqual(lo.shape, (WORDS,))
        self.assertEqual(hi.shape, (WORDS,))
        np.testing.assert_array_equal(np.asarray(_pack(lo, hi)), np.asarray(state))

    def test_word_i_sits_at_elements_2i_and_2i_plus_one(self) -> None:
        words = _STATES["counter"]
        state = _to_state(words)
        for i, w in enumerate(words):
            self.assertEqual(int(state[2 * i]), w & 0xFFFFFFFF)
            self.assertEqual(int(state[2 * i + 1]), w >> 32)

    def test_the_seam_and_the_digest_grids_run_one_permutation(self) -> None:
        # The case the two shapes exist for. `ascon.ascon_hash256_bytes` absorbs
        # into `[B, 5]` grids and never goes through `permute`; both call the
        # same `permutation`, so a layout that disagreed would be two functions
        # under one name — right bytes on one path, wrong on the other, and no
        # value test on either side would see it.
        words = _STATES["hash256_init"]
        seam = _from_state(np.asarray(AsconP().permute(_device_state(words))))

        halves = np.array([split(w) for w in words], dtype=np.uint32)
        lo = fnp.asarray(np.broadcast_to(halves[:, 0], (1, WORDS)))
        hi = fnp.asarray(np.broadcast_to(halves[:, 1], (1, WORDS)))
        out_lo, out_hi = frx.jit(
            lambda a, b: permutation_mod.permutation(a, b, masks())
        )(lo, hi)
        grid = tuple(
            int(np.asarray(out_lo)[0, i]) | (int(np.asarray(out_hi)[0, i]) << 32)
            for i in range(WORDS)
        )
        self.assertEqual(seam, grid)


class SboxCircuitTest(absltest.TestCase):
    def test_the_circuit_matches_the_standard_table_for_all_32_inputs(
        self,
    ) -> None:
        # The masked-roll grid circuit against the table-defined S-box,
        # exhaustively — the two sides share no spelling (the oracle's table
        # is transcribed from Table 6 and corner-anchored in
        # `reference_test`). Word i's low half packs bit x_i of every 5-bit
        # value, bit position j carrying input j — the bitsliced orientation
        # the state grid has, x0 the most significant index bit (Table 6's
        # convention); the high halves ride the same gates, so zeros there
        # only re-check column 0x00.
        planes = [0, 0, 0, 0, 0]
        for j in range(32):
            for i in range(WORDS):
                planes[i] |= ((j >> (4 - i)) & 1) << j
        lo = fnp.asarray(np.array([planes], dtype=np.uint32))
        hi = fnp.asarray(np.zeros((1, WORDS), dtype=np.uint32))
        out_lo, _ = frx.jit(lambda a, b: _substitution(a, b, masks()))(lo, hi)
        out = np.asarray(out_lo)[0]
        got = []
        for j in range(32):
            y = 0
            for i in range(WORDS):
                y |= (int(out[i]) >> j & 1) << (4 - i)
            got.append(y)
        self.assertEqual(tuple(got), SBOX)


class AsconPMarkerTest(absltest.TestCase):
    """The operand ABI an Ascon-p emitter will read, pinned before it exists."""

    def test_the_marked_region_captures_no_constants(self) -> None:
        # The property the ABI rests on: an array the body materialises on the
        # host is lifted into an unnamed operand AHEAD of the declared ones, one
        # per site. The round constants are scalar literals and the S-box masks
        # are counted from `iota`, so the state is the whole operand list — and
        # that is what makes it a one-operand ABI rather than one with anonymous
        # constants in front. Both routings, because the generic rewriter reads
        # the same list.
        state = _device_state(_STATES["counter"])
        for label, ctx in (
            ("generic", _generic_emitter()),
            ("dedicated", _dedicated_emitter()),
        ):
            with self.subTest(routing=label), ctx:
                eqn = composite_eqn(AsconP().permute, state)
                self.assertLen(eqn.invars, 1)
                self.assertEqual(eqn.invars[0].aval.shape, (WIDTH,))

    def test_seam_marker_matches_the_emission(self) -> None:
        assert_marker_matches_emission(self, AsconP(), _device_state(_STATES["zeros"]))

    def test_the_marked_decomposition_is_fusion_ready(self) -> None:
        # The generic rewriter accepts a straight-line element-wise body only, so
        # the decomposition INSIDE the marker is what has to hold to that —
        # `permute` itself is the composite. This is the shared whitelist rather
        # than a local blacklist: it also catches `call`, `scatter`, `dot` and
        # anything else new, and it reads the LOWERED module.
        assert_fusion_ready(_rounds, *_abi_operands(_device_state(_STATES["zeros"])))

    def test_the_dedicated_marker_carries_its_name_version_and_attrs(self) -> None:
        with _dedicated_emitter():
            eqn = composite_eqn(AsconP().permute, _device_state(_STATES["zeros"]))
        self.assertEqual(eqn.params["name"], ASCON_P_MARKER)
        self.assertEqual(eqn.params["version"], ASCON_P_MARKER_VERSION)
        attrs = composite_attrs(eqn)
        self.assertEqual(
            attrs, {"permutation": "ascon_p", "width": WIDTH, "rounds": ROUNDS}
        )

    def test_the_generic_marker_carries_no_contract(self) -> None:
        # What every leg gets today. A version or an attribute on the generic
        # region would claim a contract the marker does not have.
        with _generic_emitter():
            eqn = composite_eqn(AsconP().permute, _device_state(_STATES["zeros"]))
        self.assertEqual(eqn.params["name"], FUSED_REGION_MARKER)
        self.assertEqual(eqn.params["version"], 0)
        self.assertEmpty(eqn.params["attributes"])

    def test_both_routings_agree_with_the_reference(self) -> None:
        # The dedicated marker inlines its decomposition where no emitter
        # recognizes it, so both routings must compute the standard's bytes —
        # which is what makes the marker a routing choice rather than a
        # behaviour one.
        words = _STATES["counter"]
        want = tuple(reference_permutation(list(words)))
        for label, ctx in (
            ("generic", _generic_emitter()),
            ("dedicated", _dedicated_emitter()),
        ):
            with self.subTest(routing=label), ctx:
                got = np.asarray(AsconP().permute(_device_state(words)))
                self.assertEqual(_from_state(got), want)

    def test_the_spec_hands_out_the_abi_only_on_the_dedicated_path(self) -> None:
        # What a consumer assembling a whole-region composite reads. The inert
        # stub names no layout, which is what keeps a non-dedicated permutation
        # from being wrapped in one.
        state = _device_state(_STATES["zeros"])
        with _generic_emitter():
            operands, _permute, attrs = AsconP().fused_region_spec(state)
        self.assertLen(operands, 1)
        self.assertEqual(attrs, {})

        with _dedicated_emitter():
            operands, permute, attrs = AsconP().fused_region_spec(state)
        self.assertLen(operands, 1)
        self.assertEqual(attrs["permutation"], "ascon_p")
        # The permute the spec hands back must run off those operands and match
        # the seam's own call — that is the whole point of publishing a layout.
        np.testing.assert_array_equal(
            np.asarray(frx.jit(permute)(*operands)),
            np.asarray(frx.jit(AsconP().permute)(state)),
        )

    def test_the_abi_operands_are_the_state_alone(self) -> None:
        state = _device_state(_STATES["zeros"])
        self.assertEqual(len(_abi_operands(state)), 1)
        np.testing.assert_array_equal(
            np.asarray(_abi_operands(state)[0]), np.asarray(state)
        )
        np.testing.assert_array_equal(
            np.asarray(frx.jit(_rounds)(state)),
            np.asarray(frx.jit(AsconP().permute)(state)),
        )

    def test_the_marker_is_part_of_the_permutation_identity(self) -> None:
        # The parameter surface is empty, so without the marker in `__eq__` the
        # two routings collide in `_permute_body`'s static-arg cache and the
        # second would be served the first's marker.
        with _generic_emitter():
            generic = AsconP()
        with _dedicated_emitter():
            dedicated = AsconP()
        self.assertNotEqual(generic, dedicated)
        self.assertNotEqual(hash(generic), hash(dedicated))


class EmitterGateTest(absltest.TestCase):
    """The routing is a conjunction: the pin AND a backend that has the arm.

    No plugin ships an Ascon-p emitter, so the shipped answer is GENERIC on
    every leg. Both halves are still exercised, because a gate that collapses to
    the pin is byte-neutral — the marker is emitted, the rewriter declines it,
    the composite inlines — and only the trace cost moves.
    """

    def test_no_leg_routes_an_ascon_p_marker_yet(self) -> None:
        self.assertFalse(permutation_mod._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(permutation_mod._EMITTER_BACKENDS, ())
        perm = AsconP()
        self.assertIs(perm.fusion_path, FusionPath.GENERIC)
        self.assertEqual(perm.fused_region_marker, (FUSED_REGION_MARKER, 0))

    def test_a_backend_without_an_arm_takes_the_generic_marker(self) -> None:
        with (
            mock.patch.object(permutation_mod, "_DEDICATED_EMITTER_AVAILABLE", True),
            mock.patch.object(permutation_mod, "_EMITTER_BACKENDS", ("nonesuch",)),
        ):
            perm = AsconP()
        self.assertIs(perm.fusion_path, FusionPath.GENERIC)
        self.assertEqual(perm.fused_region_marker, (FUSED_REGION_MARKER, 0))

    def test_the_pin_still_vetoes_a_backend_that_has_the_arm(self) -> None:
        with (
            mock.patch.object(
                permutation_mod, "_EMITTER_BACKENDS", (frx.default_backend(),)
            ),
            mock.patch.object(permutation_mod, "_DEDICATED_EMITTER_AVAILABLE", False),
        ):
            perm = AsconP()
        self.assertIs(perm.fusion_path, FusionPath.GENERIC)

    def test_both_halves_true_is_what_routes_dedicated(self) -> None:
        with _dedicated_emitter():
            perm = AsconP()
        self.assertIs(perm.fusion_path, FusionPath.DEDICATED)
        self.assertEqual(
            perm.fused_region_marker, (ASCON_P_MARKER, ASCON_P_MARKER_VERSION)
        )


if __name__ == "__main__":
    absltest.main()

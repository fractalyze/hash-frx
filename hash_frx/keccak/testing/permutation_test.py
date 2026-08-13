# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Keccak-f[1600] over uint32 lane halves — values, seam conformance, and shape.

Values are checked against `reference.py`, which `reference_test` anchors to
`hashlib` so that agreement here means agreement with FIPS 202 rather than with
a second copy of the same misreading.

The lowering assertions are not decoration. A missing marker, a reduction, a
gather, or a call all still compute the right bytes and only cost the kernel, so
values alone cannot catch any of them. They are read off the *lowered* module: a
compiled one has already inlined the calls a marked body may not contain, which
is how a body with 288 call boundaries once passed a compiled-module check.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hash_frx.fusion import FUSED_REGION_MARKER
from hash_frx.keccak import permutation as permutation_mod
from hash_frx.keccak.permutation import (
    KECCAK_F_MARKER,
    KECCAK_F_MARKER_VERSION,
    KeccakF1600,
    _abi_operands,
    _permute_body,
    _rounds,
)
from hash_frx.keccak.testing.reference import (
    from_state,
    keccak_f1600,
    to_state,
)
from hash_frx.permutation import Permutation
from hash_frx.testing.fusion_ready import assert_fusion_ready
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.testing.marker_seam import assert_marker_matches_emission

_ALL_ONES = 0xFFFFFFFFFFFFFFFF

_STATES: dict[str, list[int]] = {
    "zeros": [0] * 25,
    # Every bit set: exercises chi's complement and every rotation carry.
    "ones": [_ALL_ONES] * 25,
    "counter": [(i * 0x0123456789ABCDEF) & _ALL_ONES for i in range(25)],
    # Only the top bit of the last lane: catches a rotation that drops the
    # cross-half carry, which a small-value state would hide.
    "one_hot_high": [0] * 24 + [1 << 63],
}


def _device_state(lanes: list[int]) -> frx.Array:
    return fnp.asarray(np.asarray(to_state(lanes), dtype=np.uint32))


class KeccakF1600Test(absltest.TestCase):
    def test_satisfies_the_permutation_seam(self) -> None:
        k = KeccakF1600()
        self.assertIsInstance(k, Permutation)
        self.assertEqual(k.width, 50)
        self.assertEqual(k.dtype, fnp.uint32)

    def test_value_equality_across_fresh_instances(self) -> None:
        # Parameterless, so instances must be interchangeable as jit-zone keys;
        # identity equality would re-trace on every freshly built instance.
        self.assertEqual(KeccakF1600(), KeccakF1600())
        self.assertEqual(hash(KeccakF1600()), hash(KeccakF1600()))

    def test_matches_the_reference_permutation(self) -> None:
        k = KeccakF1600()
        for name, lanes in _STATES.items():
            with self.subTest(state=name):
                got = from_state(
                    [int(v) for v in np.asarray(k.permute(_device_state(lanes)))]
                )
                self.assertEqual(got, keccak_f1600(lanes))

    def test_jit_matches_eager(self) -> None:
        k = KeccakF1600()
        x = _device_state(_STATES["counter"])
        np.testing.assert_array_equal(
            np.asarray(frx.jit(k.permute)(x)), np.asarray(k.permute(x))
        )

    def test_batches_under_vmap(self) -> None:
        # Batched over a leading axis with no Python loop over the batch.
        k = KeccakF1600()
        names = list(_STATES)
        batch = [_STATES[n] for n in names]
        stacked = fnp.asarray(
            np.stack([np.asarray(to_state(b), dtype=np.uint32) for b in batch])
        )
        out = np.asarray(frx.jit(frx.vmap(k.permute))(stacked))
        self.assertEqual(out.shape, (len(batch), 50))
        for i, lanes in enumerate(batch):
            with self.subTest(row=names[i]):
                self.assertEqual(
                    from_state([int(v) for v in out[i]]), keccak_f1600(lanes)
                )

    def test_permute_emits_one_fused_region(self) -> None:
        # The contract's unit. Without the marker the body still computes the
        # right bytes, so nothing but the lowered module shows the unit is gone.
        # Named for whichever routing is pinned — one region either way; which
        # marker carries it is `KeccakF1600MarkerTest`'s subject.
        k = KeccakF1600()
        text = frx.jit(k.permute).lower(_device_state(_STATES["zeros"])).as_text()
        self.assertEqual(text.count("stablehlo.composite"), 1)
        self.assertIn(f'"{k.fused_region_marker[0]}"', text)

    def test_seam_marker_matches_the_emission(self) -> None:
        assert_marker_matches_emission(
            self, KeccakF1600(), _device_state(_STATES["zeros"])
        )

    def test_the_marked_decomposition_is_fusion_ready(self) -> None:
        # The generic rewriter accepts a straight-line element-wise body only, so
        # the decomposition inside the marker is what has to hold to that. This
        # is the shared whitelist rather than a local blacklist: it also catches
        # `call`, `scatter`, `dot` and anything else new, and it reads the
        # LOWERED module — a compiled one has already inlined the calls that a
        # marked body may not contain.
        assert_fusion_ready(_rounds, *_abi_operands(_device_state(_STATES["zeros"])))

    def test_the_body_traces_once_across_instances(self) -> None:
        # The zone's cache is keyed on (permutation, aval), and `KeccakF1600`
        # carries no parameters beyond its marker, so freshly built instances at
        # one emitter setting must share one trace.
        x = _device_state(_STATES["counter"])
        assert_single_trace(
            self,
            _permute_body,
            [lambda: KeccakF1600().permute(x) for _ in range(3)],
        )

    def test_rejects_a_wrong_shape(self) -> None:
        k = KeccakF1600()
        with self.assertRaises(ValueError):
            k.permute(fnp.zeros(49, dtype=fnp.uint32))
        with self.assertRaises(ValueError):
            k.permute(fnp.zeros((5, 10), dtype=fnp.uint32))

    def test_rejects_a_wrong_dtype(self) -> None:
        # A uint64 state is the mistake this representation exists to prevent,
        # and int32 is the accidental one; both must fail loudly.
        k = KeccakF1600()
        with self.assertRaises(TypeError):
            k.permute(fnp.zeros(50, dtype=fnp.int32))


def _composite(fn: object, *args: frx.Array) -> Any:
    """The one composite eqn in `fn`'s jaxpr — read without lowering to MLIR."""
    eqns = [
        e
        for e in frx.make_jaxpr(fn)(*args).jaxpr.eqns
        if e.primitive.name == "composite"
    ]
    assert len(eqns) == 1, f"expected one composite, got {len(eqns)}"
    return eqns[0]


@contextlib.contextmanager
def _dedicated_emitter() -> Iterator[None]:
    """Route as if the pinned plugin shipped the Keccak emitters — the default.

    `_DEDICATED_EMITTER_AVAILABLE` is read in `__init__`, so the patch has to
    wrap construction, not just the call. Kept explicit even though it now
    matches the default: a test that names the routing it exercises still says
    so when the pin moves again.
    """
    with mock.patch.object(permutation_mod, "_DEDICATED_EMITTER_AVAILABLE", True):
        yield


@contextlib.contextmanager
def _generic_emitter() -> Iterator[None]:
    """Route as if the pinned plugin lacked them, falling back to the generic
    region marker.

    This is what a `contextlib.nullcontext()` used to mean here and no longer
    does — the default is the dedicated routing now, so a "generic" arm has to
    patch for it or it silently runs the dedicated one twice.
    """
    with mock.patch.object(permutation_mod, "_DEDICATED_EMITTER_AVAILABLE", False):
        yield


class KeccakF1600MarkerTest(absltest.TestCase):
    """The operand ABI a KeccakFusion emitter reads (#21)."""

    def test_the_marked_region_captures_no_constants(self) -> None:
        # The property the ABI rests on, and the one nothing else shows: an array
        # the body materialises on the host is lifted into an unnamed operand
        # AHEAD of the declared ones, one per site. Closing over rho's offsets
        # put 96 of them in front of the state, so the operand list a recognizer
        # would read was neither documented nor stable. Both routings, because
        # the generic rewriter reads the same list.
        state = _device_state(_STATES["counter"])
        for label, ctx in (
            ("generic", _generic_emitter()),
            ("dedicated", _dedicated_emitter()),
        ):
            with self.subTest(routing=label), ctx:
                eqn = _composite(KeccakF1600().permute, state)
                self.assertLen(eqn.invars, 2)
                self.assertEqual(eqn.invars[0].aval.shape, (50,))
                self.assertEqual(eqn.invars[1].aval.shape, (5, 5))

    def test_the_dedicated_marker_carries_its_name_version_and_attrs(self) -> None:
        with _dedicated_emitter():
            eqn = _composite(KeccakF1600().permute, _device_state(_STATES["zeros"]))
        self.assertEqual(eqn.params["name"], KECCAK_F_MARKER)
        self.assertEqual(eqn.params["version"], KECCAK_F_MARKER_VERSION)
        attrs = {key: leaves[0] for key, leaves, _ in eqn.params["attributes"]}
        self.assertEqual(attrs, {"permutation": "keccak_f", "width": 50, "rounds": 24})

    def test_the_dedicated_marker_is_the_default(self) -> None:
        # The pinned plugin ships the emitters, so this is what a consumer gets
        # without asking. Pinned as its own test because every other assertion
        # here names its routing explicitly and would still pass if the default
        # silently went back to generic — which costs the kernel and nothing else.
        k = KeccakF1600()
        self.assertTrue(k.has_dedicated_fusion)
        self.assertEqual(
            k.fused_region_marker, (KECCAK_F_MARKER, KECCAK_F_MARKER_VERSION)
        )

    def test_the_generic_marker_carries_no_contract(self) -> None:
        # The fallback where a plugin lacks the emitter. It is not a slower
        # kernel but no kernel: the rewriter declines a bare fused_region with no
        # live-width operand, so it inlines. Carrying no version and no attrs is
        # what says the marker claims nothing an emitter could read.
        with _generic_emitter():
            k = KeccakF1600()
            self.assertFalse(k.has_dedicated_fusion)
            self.assertEqual(k.fused_region_marker, (FUSED_REGION_MARKER, 0))
            eqn = _composite(k.permute, _device_state(_STATES["zeros"]))
        self.assertEqual(eqn.params["name"], FUSED_REGION_MARKER)
        self.assertEqual(eqn.params["version"], 0)
        self.assertEqual(eqn.params["attributes"], ())

    def test_both_routings_agree_with_the_reference(self) -> None:
        # A marker chooses a kernel, never a result. Checked against the
        # independent reference rather than against each other, so a shared
        # mistake in the decomposition cannot pass.
        for name, lanes_in in _STATES.items():
            for label, ctx in (
                ("generic", _generic_emitter()),
                ("dedicated", _dedicated_emitter()),
            ):
                with self.subTest(state=name, routing=label), ctx:
                    out = frx.jit(KeccakF1600().permute)(_device_state(lanes_in))
                    self.assertEqual(
                        from_state([int(v) for v in np.asarray(out)]),
                        keccak_f1600(lanes_in),
                    )

    def test_the_spec_hands_out_the_abi_only_on_the_dedicated_path(self) -> None:
        # What a consumer assembling a whole-region composite reads. The inert
        # stub is what keeps a non-dedicated permutation from being wrapped in
        # one: it names no layout, so there is nothing for an emitter to read.
        state = _device_state(_STATES["zeros"])
        with _generic_emitter():
            operands, _permute, attrs = KeccakF1600().fused_region_spec(state)
        self.assertLen(operands, 1)
        self.assertEqual(attrs, {})

        with _dedicated_emitter():
            operands, permute, attrs = KeccakF1600().fused_region_spec(state)
        self.assertLen(operands, 2)
        self.assertEqual(attrs["permutation"], "keccak_f")
        # The permute the spec hands back must run off those operands and match
        # the seam's own call — that is the whole point of publishing a layout.
        np.testing.assert_array_equal(
            np.asarray(frx.jit(permute)(*operands)),
            np.asarray(frx.jit(KeccakF1600().permute)(state)),
        )

    def test_the_marker_is_part_of_the_permutation_identity(self) -> None:
        # The parameter surface is empty, so without the marker in `__eq__` the
        # two routings collide in `_permute_body`'s static-arg cache and the
        # second silently reuses the first's marker.
        with _generic_emitter():
            generic = KeccakF1600()
        with _dedicated_emitter():
            dedicated = KeccakF1600()
        self.assertNotEqual(generic, dedicated)
        self.assertNotEqual(hash(generic), hash(dedicated))
        self.assertEqual(dedicated, KeccakF1600())


if __name__ == "__main__":
    absltest.main()

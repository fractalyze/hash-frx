# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Vision Mark-32 — values against the oracle, seam conformance, and lowering.

Values are checked against `reference.py`, which `reference_test` anchors to
the vectors published in binius-models, so agreement here means agreement with
Vision Mark-32 rather than with a second copy of the same misreading.

The lowering assertions are not decoration. A missing marker, a reduction, a
gather, or a call all still compute the right bytes and only cost the kernel,
so values alone cannot catch any of them. No backend routes a Vision marker
yet, so today's half is the generic-marker behavior; the dedicated half is
pinned through the same mock routing Keccak uses, so the plumbing an emitter
will read is held true before one exists.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array
from zk_dtypes import binary_field_t5 as F

from hash_frx.fusion import FUSED_REGION_MARKER, FusionPath
from hash_frx.permutation import Permutation
from hash_frx.testing.composite_eqn import (
    composite_attrs,
    composite_eqn,
    composite_shapes,
)
from hash_frx.testing.fusion_ready import assert_fusion_ready
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.testing.marker_seam import assert_marker_matches_emission
from hash_frx.vision import vision as vision_mod
from hash_frx.vision.params import VisionParams, vision_mark32_params
from hash_frx.vision.testing.decode import int_rows, ints
from hash_frx.vision.testing.reference import (
    MARK32_RANDOM_CIPHERTEXT,
    MARK32_RANDOM_PLAINTEXT,
    MARK32_ZERO_CIPHERTEXT,
    vision_permutation,
)
from hash_frx.vision.vision import (
    VISION_MARKER,
    VISION_MARKER_VERSION,
    Vision,
    _abi_operands,
    _permutation_body,
    _permute_body,
)

_W = 24

# Edge states alongside the seeded draws: all-zeros exercises 0 |-> 0 through
# the first inverse, all-ones the multiplicative identity, the full mask every
# bit of the top tower level, and the single-lane state the MDS diffusion of
# an almost-zero state.
_STATES: dict[str, list[int]] = {
    "zeros": [0] * _W,
    "ones": [1] * _W,
    "all_bits": [0xFFFFFFFF] * _W,
    "single_lane": [0] * 23 + [0xDEADBEEF],
}
for seed in (0, 1, 2):
    _STATES[f"seed{seed}"] = [
        int(v) for v in np.random.default_rng(seed).integers(0, 1 << 32, size=_W)
    ]


def _perm() -> Vision:
    # Deliberately fresh params per call: the value-equality and single-trace
    # cases below are about freshly built instances being interchangeable.
    return Vision(vision_mark32_params(F))


def _device_state(lanes: list[int]) -> Array:
    return fnp.array(lanes, dtype=F)


def _oracle_for(p: VisionParams, lanes: list[int]) -> list[int]:
    return vision_permutation(
        lanes, ints(p.b), ints(p.b_inv), int_rows(p.mds), int_rows(p.round_keys)
    )


# The Mark-32 tables, decoded once — an oracle run is ~0.6 s of recursive
# tower arithmetic per state, so results are cached too: the vmap batch asks
# for the exact states the direct comparison already paid for.
_MARK32 = vision_mark32_params(F)
_MARK32_TABLES = (
    ints(_MARK32.b),
    ints(_MARK32.b_inv),
    int_rows(_MARK32.mds),
    int_rows(_MARK32.round_keys),
)


@functools.cache
def _cached_oracle(lanes: tuple[int, ...]) -> list[int]:
    return vision_permutation(list(lanes), *_MARK32_TABLES)


def _oracle(lanes: list[int]) -> list[int]:
    return _cached_oracle(tuple(lanes))


def _small_params() -> VisionParams:
    """A synthetic width-3, one-round instance for the routing-plumbing cases.

    NOT a Vision anyone would ship (the tables are arbitrary; b_inv is not
    B's inverse — the permutation reads both as data): it exists because the
    marker/ABI plumbing is instance-independent while a Mark-32 compile is
    minutes of GPU budget per fresh static key."""
    return VisionParams(
        width=3,
        dtype=F,
        rounds=1,
        b=fnp.array([9, 8, 7, 6], dtype=F),
        b_inv=fnp.array([1, 2, 3, 4, 5], dtype=F),
        mds=fnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=F),
        round_keys=fnp.array([[1, 0, 2], [3, 1, 4], [5, 9, 2]], dtype=F),
    )


class VisionMark32Test(absltest.TestCase):
    def test_satisfies_the_permutation_seam(self) -> None:
        p = _perm()
        self.assertIsInstance(p, Permutation)
        self.assertEqual(p.width, _W)
        self.assertEqual(p.dtype, F)

    def test_value_equality_across_fresh_instances(self) -> None:
        # Freshly built same-params instances must be interchangeable as
        # jit-zone keys; identity equality would re-trace per instance.
        self.assertEqual(_perm(), _perm())
        self.assertEqual(hash(_perm()), hash(_perm()))

    def test_permute_matches_the_published_vectors(self) -> None:
        # Directly against the binius-models vectors, not through the oracle:
        # a transcription error shared by oracle and tables cannot pass here.
        p = _perm()
        self.assertEqual(
            ints(p.permute(_device_state([0] * _W))), list(MARK32_ZERO_CIPHERTEXT)
        )
        self.assertEqual(
            ints(p.permute(_device_state(list(MARK32_RANDOM_PLAINTEXT)))),
            list(MARK32_RANDOM_CIPHERTEXT),
        )

    def test_permute_matches_the_oracle(self) -> None:
        p = _perm()
        for name, lanes in _STATES.items():
            with self.subTest(state=name):
                got = ints(p.permute(_device_state(lanes)))
                self.assertEqual(got, _oracle(lanes))

    def test_batches_under_vmap(self) -> None:
        # Batched over a leading axis with no Python loop over the batch.
        p = _perm()
        names = list(_STATES)
        stacked = fnp.array([_STATES[n] for n in names], dtype=F)
        out = frx.jit(frx.vmap(p.permute))(stacked)
        self.assertEqual(out.shape, (len(names), _W))
        rows = int_rows(out)
        for i, name in enumerate(names):
            with self.subTest(row=name):
                self.assertEqual(rows[i], _oracle(_STATES[name]))

    def test_permute_reuses_one_trace_across_instances(self) -> None:
        # The zone's cache is keyed on (permutation, aval) with the permutation
        # compared by value, so freshly built same-params instances must share
        # one trace.
        x = _device_state(_STATES["seed0"])
        calls = [functools.partial(_perm().permute, x) for _ in range(3)]
        assert_single_trace(self, _permute_body, calls)

    def test_rejects_a_wrong_shape(self) -> None:
        p = _perm()
        with self.assertRaises(ValueError):
            p.permute(fnp.zeros(_W - 1, dtype=F))
        with self.assertRaises(ValueError):
            p.permute(fnp.zeros((2, _W), dtype=F))

    def test_rejects_a_wrong_dtype(self) -> None:
        p = _perm()
        with self.assertRaises(TypeError):
            p.permute(fnp.zeros(_W, dtype=fnp.uint32))


@contextlib.contextmanager
def _routing(dedicated: bool) -> Iterator[None]:
    """Pin which marker `Vision` picks. No leg has a Vision emitter, so the
    dedicated arm exists only under this patch — which is the point: the ABI
    an emitter will read is pinned before the emitter exists, off the jaxpr,
    which needs no recognizer. The decision is read in `__init__`, so this
    wraps construction rather than just the call."""
    with mock.patch.object(
        vision_mod, "_routes_to_dedicated_emitter", lambda: dedicated
    ):
        yield


class VisionMarkerTest(absltest.TestCase):
    """The marker contract, generic today and dedicated under mock routing."""

    def test_no_leg_routes_a_vision_marker_yet(self) -> None:
        # The pre-emitter pin: both module flags say "no emitter", so every
        # unpatched instance is on the generic path on every backend. When an
        # emitter lands these flip with the frx floor, and this case flips to
        # the keccak-style backend gate.
        self.assertFalse(vision_mod._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(vision_mod._EMITTER_BACKENDS, ())
        p = _perm()
        self.assertIs(p.fusion_path, FusionPath.GENERIC)
        self.assertEqual(p.fused_region_marker, (FUSED_REGION_MARKER, 0))

    def test_permute_emits_one_fused_region(self) -> None:
        # The contract's unit: without the marker the body still computes the
        # right bytes, so only the lowered module shows the unit is gone.
        p = _perm()
        txt = frx.jit(p.permute).lower(_device_state(_STATES["zeros"])).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1)
        self.assertIn(f'"{p.fused_region_marker[0]}"', txt)

    def test_vmap_permute_keeps_one_region(self) -> None:
        # A vmapped permute must stay ONE composite over a batched operand,
        # not a region per element.
        p = _perm()
        states = fnp.array([_STATES["seed0"], _STATES["seed1"]], dtype=F)
        txt = frx.jit(lambda x: frx.vmap(p.permute)(x)).lower(states).as_text()
        comp = [ln for ln in txt.splitlines() if "stablehlo.composite" in ln]
        self.assertLen(comp, 1, txt)
        self.assertIn("2x24x", comp[0])

    def test_seam_marker_matches_the_emission(self) -> None:
        assert_marker_matches_emission(self, _perm(), _device_state(_STATES["zeros"]))

    def test_the_marked_decomposition_is_fusion_ready(self) -> None:
        # The generic rewriter accepts a straight-line element-wise body only.
        # The inverse S-box lowers to an element-wise `divide` (whitelisted for
        # exactly this body); everything else is add/multiply/slice — so the
        # whole 16-step decomposition holds to the whitelist with no reduce.
        p = _perm()
        assert_fusion_ready(
            functools.partial(_permutation_body, p),
            *_abi_operands(p, _device_state(_STATES["zeros"])),
        )

    def test_the_generic_marker_carries_no_contract(self) -> None:
        # Carrying no version and no attrs is what says the generic marker
        # claims nothing an emitter could read.
        eqn = composite_eqn(_perm().permute, _device_state(_STATES["zeros"]))
        self.assertEqual(eqn.params["name"], FUSED_REGION_MARKER)
        self.assertEqual(eqn.params["version"], 0)
        self.assertEqual(eqn.params["attributes"], ())

    def test_the_marked_region_captures_no_constants(self) -> None:
        # The property the ABI rests on: an array the body materialises on the
        # host is lifted into an unnamed operand AHEAD of the declared ones,
        # one per site (the rho-offsets lesson in `keccak.permutation`). Both
        # routings, because the generic rewriter reads the same list.
        state = _device_state(_STATES["seed0"])
        for label, dedicated in (("generic", False), ("dedicated", True)):
            with self.subTest(routing=label), _routing(dedicated):
                eqn = composite_eqn(Vision(vision_mark32_params(F)).permute, state)
                self.assertLen(eqn.invars, 5)
                shapes = composite_shapes(eqn)
                self.assertEqual(shapes, [(24,), (4,), (33,), (24, 24), (17, 24)])

    def test_the_dedicated_marker_carries_its_name_version_and_attrs(self) -> None:
        with _routing(True):
            eqn = composite_eqn(
                Vision(vision_mark32_params(F)).permute,
                _device_state(_STATES["zeros"]),
            )
        self.assertEqual(eqn.params["name"], VISION_MARKER)
        self.assertEqual(eqn.params["version"], VISION_MARKER_VERSION)
        attrs = composite_attrs(eqn)
        self.assertEqual(attrs, {"permutation": "vision", "width": 24, "rounds": 8})

    def test_both_routings_agree_with_the_reference(self) -> None:
        # A marker chooses a kernel, never a result. Checked against the
        # independent reference rather than against each other, so a shared
        # mistake in the decomposition cannot pass. On the small instance:
        # the plumbing is instance-independent, and the dedicated routing is
        # a fresh static key whose Mark-32 compile is minutes of GPU budget
        # the value tests already spend on the generic one.
        p = _small_params()
        lanes = [5, 6, 7]
        want = _oracle_for(p, lanes)
        for label, dedicated in (("generic", False), ("dedicated", True)):
            with self.subTest(routing=label), _routing(dedicated):
                out = frx.jit(Vision(p).permute)(_device_state(lanes))
                self.assertEqual(ints(out), want)

    def test_the_spec_hands_out_the_abi_only_on_the_dedicated_path(self) -> None:
        # The inert stub is what keeps a non-dedicated permutation from being
        # wrapped in a whole-region composite: it names no layout. On the
        # small instance because the layout check EXECUTES the spec's permute,
        # a raw un-marked body — for Mark-32 that graph is a very-slow-compile
        # on the GPU leg (it exists on-device only inside a marker).
        p = _small_params()
        state = _device_state([1, 2, 3])
        with _routing(False):
            operands, _permute, attrs = Vision(p).fused_region_spec(state)
        self.assertLen(operands, 1)
        self.assertEqual(attrs, {})

        with _routing(True):
            operands, permute, attrs = Vision(p).fused_region_spec(state)
        self.assertLen(operands, 5)
        self.assertEqual(attrs["permutation"], "vision")
        # The permute the spec hands back must run off those operands and
        # match the seam's own call — the whole point of publishing a layout.
        np.testing.assert_array_equal(
            np.asarray(frx.jit(permute)(*operands)),
            np.asarray(Vision(p).permute(state)),
        )

    def test_the_marker_is_part_of_the_permutation_identity(self) -> None:
        # Without the marker in `__eq__` the two routings collide in
        # `_permute_body`'s static-arg cache and the second silently reuses
        # the first's marker.
        with _routing(False):
            generic = Vision(vision_mark32_params(F))
        with _routing(True):
            dedicated = Vision(vision_mark32_params(F))
        self.assertNotEqual(generic, dedicated)
        self.assertNotEqual(hash(generic), hash(dedicated))
        # No emitter exists anywhere, so an unpatched construction is the
        # generic one on every leg.
        self.assertEqual(generic, _perm())


if __name__ == "__main__":
    absltest.main()

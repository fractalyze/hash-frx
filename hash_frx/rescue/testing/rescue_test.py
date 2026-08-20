# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Rescue RPO-128 — values against the oracle, seam conformance, and lowering.

Values are checked against `reference.py`, which `reference_test` anchors to
the digests published in the RPO paper (and hard-coded by miden-crypto), so
agreement here means agreement with Rescue-Prime Optimized rather than with a
second copy of the same misreading. One case pins the device permutation to
the published digests DIRECTLY, through the paper's own sponge schedule, so
not even the oracle sits between the frx spelling and the publication.

The lowering assertions are not decoration. A missing marker, a reduction, a
gather, or a call all still compute the right bytes and only cost the kernel,
so values alone cannot catch any of them. No backend routes a Rescue marker
yet, so today's half is the generic-marker behavior; the dedicated half is
pinned through the same mock routing Keccak uses, so the plumbing an emitter
will read is held true before one exists.

One toolchain caveat, and it is the GPU leg doing exactly the job the fusion
contract gives it: the pinned plugin's GPU codegen miscompiles
`(a + b) * (a + b)` over the 64-bit Goldilocks Montgomery dtype — the square
of a sum inside one fused kernel, which every Rescue S-box layer contains by
construction (`(state + rc) ** alpha`), with no authoring dodge (the
field-aware simplifier folds every respelling back). Tracked with a minimal
repro and analysis at https://github.com/fractalyze/xla/issues/564. The
value-executing cases here guard on `_gl64_square_of_add_miscompiles`, a
runtime probe of that exact pattern on the current backend: today it skips
them on the GPU leg only, and the moment a pinned plugin computes the probe
correctly they run again everywhere — no stale skip to remember. Every
lowering, marker, ABI, and trace-count case runs on both legs regardless.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator
from typing import Any
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest
from frx import Array
from zk_dtypes import goldilocks_mont as F

from hash_frx.fusion import FUSED_REGION_MARKER, FusionPath
from hash_frx.linear import apply_matrix
from hash_frx.permutation import Permutation
from hash_frx.rescue import rescue as rescue_mod
from hash_frx.rescue.params import RescueParams, rescue_rpo128_params
from hash_frx.rescue.rescue import (
    RESCUE_MARKER,
    RESCUE_MARKER_VERSION,
    Rescue,
    _abi_operands,
    _permutation_body,
    _permute_body,
)
from hash_frx.rescue.testing.decode import int_rows, ints
from hash_frx.rescue.testing.reference import (
    RPO128_CAPACITY,
    RPO128_M,
    RPO128_P,
    RPO128_TEST_VECTORS,
    rescue_permutation,
)
from hash_frx.sponge import Sponge, SpongeParams
from hash_frx.testing.fusion_ready import assert_fusion_ready, assert_input_uses
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.testing.marker_seam import assert_marker_matches_emission
from hash_frx.testing.random_field import rand_field

_W = 12

# Edge states alongside the seeded draws: all-zeros exercises 0 fixed by both
# power maps until the first constants land, all-ones the multiplicative
# identity, all-(p-1) the top of the field (where the Montgomery storage and
# the canonical value differ most), and the single-lane state the MDS
# diffusion of an almost-zero state. Seeded draws come from `rand_field`, the
# production Montgomery encoding.
_STATES: dict[str, list[int]] = {
    "zeros": [0] * _W,
    "ones": [1] * _W,
    "max": [RPO128_P - 1] * _W,
    "single_lane": [0] * 11 + [0xDEADBEEF],
}
for seed in (0, 1, 2):
    _STATES[f"seed{seed}"] = ints(rand_field(seed, (_W,), F))


def _perm() -> Rescue:
    # Deliberately fresh params per call: the value-equality and single-trace
    # cases below are about freshly built instances being interchangeable.
    return Rescue(rescue_rpo128_params(F))


def _device_state(lanes: list[int]) -> Array:
    return fnp.array(lanes, dtype=F)


def _oracle_for(p: RescueParams, lanes: list[int]) -> list[int]:
    return rescue_permutation(
        lanes,
        int_rows(p.mds),
        int_rows(p.round_constants),
        p.alpha,
        p.inv_alpha,
        zk_dtypes.pfinfo(p.dtype).modulus,
    )


_RPO128 = rescue_rpo128_params(F)


def _oracle(lanes: list[int]) -> list[int]:
    return _oracle_for(_RPO128, lanes)


@functools.cache
def _gl64_square_of_add_miscompiles() -> bool:
    """Whether this backend's toolchain miscompiles the square-of-a-sum over
    the 64-bit Goldilocks Montgomery dtype (fractalyze/xla#564) — the pattern
    every Rescue S-box input is. Probed at runtime on the exact pinned plugin
    the suite runs, so the guard evaporates the moment a fixed plugin lands
    rather than living on as a stale skip."""
    a, b = [3, 5, 7], [11, 13, 17]
    got = ints(
        frx.jit(lambda x, y: (x + y) * (x + y))(_device_state(a), _device_state(b))
    )
    return got != [(x + y) ** 2 % RPO128_P for x, y in zip(a, b)]


def _require_trustworthy_values(test: absltest.TestCase) -> None:
    """Skip a value-asserting case where the toolchain is known to miscompile
    the body's arithmetic (see the module docstring): a red that only restates
    xla#564 hides a real regression, and a green is unobtainable. Lowering and
    marker cases never call this — they must hold everywhere."""
    if _gl64_square_of_add_miscompiles():
        test.skipTest(
            "pinned plugin miscompiles goldilocks_mont (a+b)*(a+b) on this "
            "backend — https://github.com/fractalyze/xla/issues/564"
        )


def _small_params() -> RescueParams:
    """A synthetic width-3, one-round instance for the routing-plumbing cases.

    NOT a Rescue anyone would ship (the tables are arbitrary; inv_alpha is not
    alpha's inverse — the permutation reads both as data): it exists because
    the marker/ABI plumbing is instance-independent while an RPO-128 compile
    is real GPU budget per fresh static key."""
    return RescueParams(
        width=3,
        dtype=F,
        rounds=1,
        alpha=3,
        inv_alpha=5,
        mds=fnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=F),
        round_constants=fnp.array([[1, 0, 2], [3, 1, 4]], dtype=F),
    )


class RescueRpo128Test(absltest.TestCase):
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
        _require_trustworthy_values(self)
        # Directly against the paper's digests, not through the oracle: the
        # published sponge schedule (capacity lanes 0..3, overwrite absorb,
        # 1-then-zeros padding with the domain bit — reference.py cites the
        # sections) is spelled here over the DEVICE permute, so a
        # transcription error shared by oracle and tables cannot pass. Inputs
        # [0] and [0..18] cover the padded one-permute and the exact-block
        # three-permute chains.
        p = _perm()
        rate = RPO128_M - RPO128_CAPACITY
        for i in (0, 18):
            with self.subTest(input_length=i + 1):
                seq = list(range(i + 1))
                state = [0] * RPO128_M
                if len(seq) % rate != 0:
                    seq.append(1)
                    while len(seq) % rate != 0:
                        seq.append(0)
                    state[0] = 1
                s = _device_state(state)
                for k in range(0, len(seq), rate):
                    block = _device_state(seq[k : k + rate])
                    s = p.permute(fnp.concatenate([s[:RPO128_CAPACITY], block]))
                digest = ints(s)[RPO128_CAPACITY : RPO128_CAPACITY + rate // 2]
                self.assertEqual(digest, list(RPO128_TEST_VECTORS[i]))

    def test_permute_matches_the_oracle(self) -> None:
        _require_trustworthy_values(self)
        p = _perm()
        for name, lanes in _STATES.items():
            with self.subTest(state=name):
                got = ints(p.permute(_device_state(lanes)))
                self.assertEqual(got, _oracle(lanes))

    def test_batches_under_vmap(self) -> None:
        _require_trustworthy_values(self)
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
            p.permute(fnp.zeros(_W, dtype=fnp.uint64))

    def test_sponge_over_rescue_runs(self) -> None:
        # The seam's promise: `Sponge` reads width/dtype and calls permute, so
        # a Rescue drops in with no construction code of its own. On the small
        # instance (the construction is instance-independent); a GENERIC-path
        # permutation runs the while_loop absorb, so this also exercises
        # permute inside traced control flow.
        sponge = Sponge(Rescue(_small_params()), SpongeParams(rate=2, out=2))
        digest = sponge.hash(fnp.array([5, 6, 7, 8], dtype=F))
        self.assertEqual(digest.shape, (2,))
        self.assertEqual(digest.dtype, F)


class RescueLayerContractTest(absltest.TestCase):
    """The body's two building blocks hold the fusion whitelist and the
    chained-input read limit on their own — `hash_frx.linear` states the rule
    (a re-read compounds per chained layer, and Rescue chains 14 per permute).
    """

    def test_the_power_chains_are_fusion_ready_and_read_limited(self) -> None:
        # `fnp.power` with a STATIC integer exponent (the Poseidon-family
        # spelling) must lower to a square-and-multiply chain of whitelist
        # multiplies — this is where that lowering is held to the contract
        # rather than assumed, the 64-bit inverse exponent being the sharp
        # case no other family exercises.
        x = _device_state(_STATES["seed0"])
        for label, e in (("alpha", 7), ("inv_alpha", _RPO128.inv_alpha)):
            with self.subTest(exponent=label):
                assert_fusion_ready(lambda v, e=e: fnp.power(v, e), x)
                # The base feeds the first squaring (both factors) and the
                # bit-0 fold: THREE reads no matter the exponent width.
                assert_input_uses(lambda v, e=e: fnp.power(v, e), x, limit=3)

    def test_the_mds_stays_normal_form(self) -> None:
        x = _device_state(_STATES["seed0"])
        assert_fusion_ready(lambda v: apply_matrix(_RPO128.mds, v), x)
        assert_input_uses(lambda v: apply_matrix(_RPO128.mds, v), x, limit=_W)
        # The matrix form reduces, so the gate must bite — else the whole-body
        # check below proves nothing.
        with self.assertRaises(AssertionError):
            assert_fusion_ready(lambda v: _RPO128.mds @ v, x)


def _composite(fn: object, *args: Array) -> Any:
    """The one composite eqn in `fn`'s jaxpr — read without lowering to MLIR."""
    eqns = [
        e
        for e in frx.make_jaxpr(fn)(*args).jaxpr.eqns
        if e.primitive.name == "composite"
    ]
    assert len(eqns) == 1, f"expected one composite, got {len(eqns)}"
    return eqns[0]


@contextlib.contextmanager
def _routing(dedicated: bool) -> Iterator[None]:
    """Pin which marker `Rescue` picks. No leg has a Rescue emitter, so the
    dedicated arm exists only under this patch — which is the point: the ABI
    an emitter will read is pinned before the emitter exists, off the jaxpr,
    which needs no recognizer. The decision is read in `__init__`, so this
    wraps construction rather than just the call."""
    with mock.patch.object(
        rescue_mod, "_routes_to_dedicated_emitter", lambda: dedicated
    ):
        yield


class RescueMarkerTest(absltest.TestCase):
    """The marker contract, generic today and dedicated under mock routing."""

    def test_no_leg_routes_a_rescue_marker_yet(self) -> None:
        # The pre-emitter pin: both module flags say "no emitter", so every
        # unpatched instance is on the generic path on every backend. When an
        # emitter lands these flip with the frx floor, and this case flips to
        # the keccak-style backend gate.
        self.assertFalse(rescue_mod._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(rescue_mod._EMITTER_BACKENDS, ())
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
        self.assertIn("2x12x", comp[0])

    def test_seam_marker_matches_the_emission(self) -> None:
        assert_marker_matches_emission(self, _perm(), _device_state(_STATES["zeros"]))

    def test_the_marked_decomposition_is_fusion_ready(self) -> None:
        # The generic rewriter accepts a straight-line element-wise body only,
        # and THIS body is the family's sharp case: the inverse S-box unrolls
        # to ~100 multiplies per layer, and all ~860 multiplies across the 7
        # rounds must hold to the whitelist with no reduce, call, or gather.
        p = _perm()
        assert_fusion_ready(
            functools.partial(_permutation_body, p),
            *_abi_operands(p, _device_state(_STATES["zeros"])),
        )

    def test_the_generic_marker_carries_no_contract(self) -> None:
        # Carrying no version and no attrs is what says the generic marker
        # claims nothing an emitter could read.
        eqn = _composite(_perm().permute, _device_state(_STATES["zeros"]))
        self.assertEqual(eqn.params["name"], FUSED_REGION_MARKER)
        self.assertEqual(eqn.params["version"], 0)
        self.assertEqual(eqn.params["attributes"], ())

    def test_the_marked_region_captures_no_constants(self) -> None:
        # The property the ABI rests on: an array the body materialises on the
        # host is lifted into an unnamed operand AHEAD of the declared ones,
        # one per site (the rho-offsets lesson in `keccak.permutation`). Both
        # routings, because the generic rewriter reads the same list. The
        # S-box exponents must leave no trace here: they are static ints the
        # power chains unroll over, not arrays.
        state = _device_state(_STATES["seed0"])
        for label, dedicated in (("generic", False), ("dedicated", True)):
            with self.subTest(routing=label), _routing(dedicated):
                eqn = _composite(Rescue(rescue_rpo128_params(F)).permute, state)
                self.assertLen(eqn.invars, 3)
                shapes = [tuple(v.aval.shape) for v in eqn.invars]
                self.assertEqual(shapes, [(12,), (12, 12), (14, 12)])

    def test_the_dedicated_marker_carries_its_name_version_and_attrs(self) -> None:
        # `inv_alpha` is deliberately NOT an attr: it exceeds 2^63 - 1 (no
        # signed i64 carries it) and an emitter derives it from `alpha` and
        # its field — `rescue._marker_attrs` states the contract.
        with _routing(True):
            eqn = _composite(
                Rescue(rescue_rpo128_params(F)).permute,
                _device_state(_STATES["zeros"]),
            )
        self.assertEqual(eqn.params["name"], RESCUE_MARKER)
        self.assertEqual(eqn.params["version"], RESCUE_MARKER_VERSION)
        attrs = {key: leaves[0] for key, leaves, _ in eqn.params["attributes"]}
        self.assertEqual(
            attrs,
            {"permutation": "rescue", "width": 12, "rounds": 7, "alpha": 7},
        )

    def test_both_routings_agree_with_the_reference(self) -> None:
        _require_trustworthy_values(self)
        # A marker chooses a kernel, never a result. Checked against the
        # independent reference rather than against each other, so a shared
        # mistake in the decomposition cannot pass. On the small instance:
        # the plumbing is instance-independent, and the dedicated routing is
        # a fresh static key whose RPO-128 compile is GPU budget the value
        # tests already spend on the generic one.
        p = _small_params()
        lanes = [5, 6, 7]
        want = _oracle_for(p, lanes)
        for label, dedicated in (("generic", False), ("dedicated", True)):
            with self.subTest(routing=label), _routing(dedicated):
                out = frx.jit(Rescue(p).permute)(_device_state(lanes))
                self.assertEqual(ints(out), want)

    def test_the_spec_hands_out_the_abi_only_on_the_dedicated_path(self) -> None:
        _require_trustworthy_values(self)
        # The inert stub is what keeps a non-dedicated permutation from being
        # wrapped in a whole-region composite: it names no layout. On the
        # small instance because the layout check EXECUTES the spec's permute.
        p = _small_params()
        state = _device_state([1, 2, 3])
        with _routing(False):
            operands, _permute, attrs = Rescue(p).fused_region_spec(state)
        self.assertLen(operands, 1)
        self.assertEqual(attrs, {})

        with _routing(True):
            operands, permute, attrs = Rescue(p).fused_region_spec(state)
        self.assertLen(operands, 3)
        self.assertEqual(attrs["permutation"], "rescue")
        # The permute the spec hands back must run off those operands and
        # match the seam's own call — the whole point of publishing a layout.
        np.testing.assert_array_equal(
            np.asarray(frx.jit(permute)(*operands)),
            np.asarray(Rescue(p).permute(state)),
        )

    def test_the_marker_is_part_of_the_permutation_identity(self) -> None:
        # Without the marker in `__eq__` the two routings collide in
        # `_permute_body`'s static-arg cache and the second silently reuses
        # the first's marker.
        with _routing(False):
            generic = Rescue(rescue_rpo128_params(F))
        with _routing(True):
            dedicated = Rescue(rescue_rpo128_params(F))
        self.assertNotEqual(generic, dedicated)
        self.assertNotEqual(hash(generic), hash(dedicated))
        # No emitter exists anywhere, so an unpatched construction is the
        # generic one on every leg.
        self.assertEqual(generic, _perm())


if __name__ == "__main__":
    absltest.main()

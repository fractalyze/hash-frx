"""Every shipped Poseidon2 set byte-matches the honest Plonky3 reference, and
the koalabear-16 set carries the engine's lowering and marker contract.

The per-set half is `ShippedSetTest`, parameterized over `_SETS` — a set that
ships without a vector is then a thing that cannot happen, which is the
argument `testing/rows.py` makes for its own table. Everything else stays on
koalabear-16 and is deliberately NOT swept: `frx.vmap`'s batching, the
five-operand ABI and the trace-sharing zone are properties of the engine rather
than of a parameterization, so running them per set buys another compile of the
same code path.

The one exception is marker RECOGNITION, which is swept: the marker carries
`alpha` and `internal_rounds` as attributes, so each set presents a different
attribute tuple to the pinned toolchain and a set the recognizer declines is
byte-invisible (`testing/marker_recognized.py` states why).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any
from unittest import mock

import frx
import frx.numpy as fnp
from absl.testing import absltest, parameterized
from zk_dtypes import babybear_mont as BB
from zk_dtypes import koalabear_mont as F

from hash_frx.fusion import FUSED_REGION_MARKER, FusionPath
from hash_frx.poseidon2 import poseidon2 as poseidon2_mod
from hash_frx.poseidon2.poseidon2 import (
    POSEIDON2_MARKER,
    POSEIDON2_MARKER_VERSION,
    Poseidon2,
    _permute_body,
)
from hash_frx.poseidon2.testing.babybear16 import (
    BABYBEAR16_EXPECTED,
    babybear16_perm,
)
from hash_frx.poseidon2.testing.koalabear16 import (
    KOALABEAR16_EXPECTED,
    KOALABEAR16_POSEIDON2_ATTRS,
    koalabear16_params,
    koalabear16_perm,
    koalabear16_scaled_perm,
)
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.testing.marker_recognized import assert_marker_recognized
from hash_frx.testing.marker_seam import assert_marker_matches_emission

# Each shipped set: its factory, its dtype, and the Plonky3 vector it is held
# to. Rows, not classes — the table is what makes a new set's coverage
# automatic rather than dependent on remembering to write a class.
_SETS = (
    ("koalabear16", koalabear16_perm, F, KOALABEAR16_EXPECTED),
    ("babybear16", babybear16_perm, BB, BABYBEAR16_EXPECTED),
)


def _permute_routed(build: Any) -> Any:
    """Build a permutation whose `permute` takes the dedicated marker, whatever
    the backend routes.

    Only the CPU routes the standalone permute marker; the cases below assert
    the marker TEXT rather than compile it, so forcing the route is what lets
    them state that contract from either runner. The route is decided in
    `__init__`, so the patch has to wrap construction.
    """
    with mock.patch.object(poseidon2_mod, "_routes_to_dedicated_emitter", lambda: True):
        return build()


class ShippedSetTest(parameterized.TestCase):
    """What every shipped set owes, at every set."""

    @parameterized.named_parameters(*_SETS)
    def test_permute_byte_matches_plonky3(
        self, factory: Any, dtype: Any, expected: Any
    ) -> None:
        # Runs the SHIPPED parameters, so this is what holds each set's
        # published constants to the revision `standard.py` names for it. Where
        # each vector came from is stated beside the vector, in the fixture
        # module that owns it.
        out = factory().permute(fnp.arange(16, dtype=dtype))
        self.assertTrue(bool(fnp.array_equal(out, expected)))

    @parameterized.named_parameters(*_SETS)
    def test_marker_is_recognized_by_the_pinned_toolchain(
        self, factory: Any, dtype: Any, expected: Any
    ) -> None:
        # Swept where the value tests above would not need to be: `alpha` and
        # `internal_rounds` ride the marker as ATTRIBUTES, so babybear-16
        # (alpha 7, 13 internal rounds) presents a tuple koalabear-16
        # (alpha 3, 20) never did. A recognizer that declines it is invisible
        # to every value test — the marker inlines and the bytes stay right.
        #
        # Which marker is on the wire is the leg's business; "one dedicated
        # kernel" is owed on both. A leg that routes no permute marker earns its
        # kernel by having the round loops converted to a single kCustom
        # `static_while` fusion instead, and a converter that starts declining
        # them is the same silent loss — right bytes, kernel gone — this case
        # exists to catch.
        permute = factory().permute
        x = fnp.arange(16, dtype=dtype)
        if poseidon2_mod._routes_to_dedicated_emitter():
            assert_marker_recognized(self, "poseidon2", permute, x)
            return
        lines = [
            ln
            for ln in frx.jit(permute).lower(x).compile().as_text().splitlines()
            if "kind=kCustom" in ln
        ]
        self.assertLen(lines, 1, lines)
        self.assertIn("%static_while_fusion", lines[0])


class Poseidon2Koalabear16Test(absltest.TestCase):
    """The engine's contract, exercised through one set.

    These are properties of the Poseidon2 implementation rather than of a
    parameterization — batching, the operand ABI, the trace-sharing zone — so
    they are asserted once. `ShippedSetTest` above is where a second set earns
    its coverage.
    """

    def test_vmap_batch_matches(self) -> None:
        p = koalabear16_perm()
        x = fnp.arange(16, dtype=F)
        batch = frx.vmap(p.permute)(fnp.stack([x, x]))
        self.assertTrue(bool(fnp.array_equal(batch[0], KOALABEAR16_EXPECTED)))
        self.assertTrue(bool(fnp.array_equal(batch[1], KOALABEAR16_EXPECTED)))

    def test_permute_reuses_one_trace_across_instances(self) -> None:
        # Freshly built same-params permutations must share one module-level
        # permute trace — the static key compares by value. Without the zone,
        # every composite emission re-traces the permutation body, which
        # dominates the first-trace-per-config cost for any consumer emitting
        # many identical-aval permutes.
        x = fnp.arange(16, dtype=F)
        calls = [functools.partial(koalabear16_perm().permute, x) for _ in (0, 1)]
        assert_single_trace(self, _permute_body, calls)

    def test_permute_emits_poseidon2_named_composite(self) -> None:
        # The standard-MDS permute marks its region "hash_frx.perm.poseidon2" so XLA
        # routes it to the dedicated Poseidon2Fusion emitter; the permutation
        # shape rides as composite.attributes — all four ints are required by
        # the XLA recognizer. W=16, E=4, I=20, alpha=3 for koalabear-16.
        p = _permute_routed(koalabear16_perm)
        txt = frx.jit(p.permute).lower(fnp.arange(16, dtype=F)).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn(f'"{POSEIDON2_MARKER}"', composite_line)
        self.assertIn(KOALABEAR16_POSEIDON2_ATTRS, composite_line)
        self.assertIn(f"version = {POSEIDON2_MARKER_VERSION}", composite_line)
        # Exactly the 5 ABI operands: the J scale rides as an attribute and any
        # closed-over constant must stay inline (frx#218), never a 6th operand.
        operands = composite_line.split(f'"{POSEIDON2_MARKER}"')[1].split("{")[0]
        self.assertEqual(operands.count("%"), 5, composite_line)

    def test_seam_marker_matches_the_emission(self) -> None:
        # `fusion_path` is named rather than derived: it reports whether the
        # primitive can be DRIVEN here, which every backend does for an
        # M4-structured set, while the marker follows the narrower question of
        # whether this backend routes a standalone permute kernel.
        assert_marker_matches_emission(
            self,
            koalabear16_perm(),
            fnp.arange(16, dtype=F),
            expected_path=FusionPath.DEDICATED,
        )

    def test_non_identity_j_scale_stays_five_operands_canonical(self) -> None:
        # A non-identity J scale must ride as the CANONICAL attribute value.
        # koalabear16_scaled_perm's scale is R⁻¹: canonical
        # 1057030144, Montgomery STORAGE 1 — so the attribute must read 1057030144,
        # not the storage 1 a raw-bits/canonical mixup would emit.
        p = _permute_routed(koalabear16_scaled_perm)
        txt = frx.jit(p.permute).lower(fnp.arange(16, dtype=F)).as_text()
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn("internal_j_scale = 1057030144 : i64", composite_line)
        operands = composite_line.split(f'"{POSEIDON2_MARKER}"')[1].split("{")[0]
        self.assertEqual(operands.count("%"), 5, composite_line)

    def test_vmap_permute_keeps_dedicated_marker(self) -> None:
        # If frx's composite batching rule regresses, vmap silently falls back to
        # generic loop fusion — the dedicated kernel lost with no error.
        p = _permute_routed(koalabear16_perm)
        states = fnp.arange(5 * 16, dtype=F).reshape(5, 16)
        txt = frx.jit(lambda x: frx.vmap(p.permute)(x)).lower(states).as_text()
        comp = [ln for ln in txt.splitlines() if "stablehlo.composite" in ln]
        self.assertEqual(len(comp), 1, txt)  # one composite over the whole batch
        self.assertIn(f'"{POSEIDON2_MARKER}"', comp[0])  # dedicated, not generic
        self.assertIn("5x16x", comp[0])  # batched operand (b=5), not per-element

    def test_free_form_external_matrix_uses_generic_marker(self) -> None:
        # The Poseidon2Fusion emitter assumes an (I + J_blocks) ⊗ M4 external
        # layer (M4 rides as a marker attribute), so a free-form matrix that is
        # NOT M4-block-structured must NOT take the hash_frx.poseidon2 route — it
        # falls back to the generic zorch.fused_region marker (LoopFusion lowers
        # the real body) to stay correct. (An M4-block-structured matrix — e.g.
        # the HorizenLabs reference — does take the dedicated route.)
        custom = fnp.arange(16 * 16, dtype=F).reshape(16, 16)
        p = Poseidon2(dataclasses.replace(koalabear16_params(), external_matrix=custom))
        txt = frx.jit(p.permute).lower(fnp.arange(16, dtype=F)).as_text()
        self.assertNotIn(POSEIDON2_MARKER, txt)
        self.assertIn("zorch.fused_region", txt)

    def test_non_plonky3_m4_takes_dedicated_route(self) -> None:
        # A non-default but M4-block-structured matrix (here the HorizenLabs
        # reference M4) must take the dedicated hash_frx.poseidon2 route, carrying its
        # own M4 as the external_m4 attribute — not fall back to the generic
        # marker — so the dedicated emitter serves any M4 without a special case.
        hl_m4 = [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]]
        w = 16
        mds = fnp.array(
            [
                [hl_m4[i % 4][j % 4] * (2 if i // 4 == j // 4 else 1) for j in range(w)]
                for i in range(w)
            ],
            dtype=F,
        )
        p = _permute_routed(
            lambda: Poseidon2(
                dataclasses.replace(koalabear16_params(), external_matrix=mds)
            )
        )
        txt = frx.jit(p.permute).lower(fnp.arange(w, dtype=F)).as_text()
        self.assertIn(POSEIDON2_MARKER, txt)
        self.assertIn(
            "external_m4 = dense<[5, 7, 1, 3, 4, 6, 1, 1, 1, 3, 5, 7, 1, 1, 4, 6]> :"
            " tensor<16xi64>",
            txt,
        )


class Poseidon2RoutingTest(absltest.TestCase):
    """Two routing questions, answered separately.

    Driving the primitive (running its round schedule from the params a marker
    carries) is what a whole-hash envelope over the permutation needs; routing
    the standalone permute marker to a kernel of its own is narrower. Reading
    the second where the first is meant is what silently costs an enclosing
    sponge its kernel, so these hold them apart.
    """

    def test_the_production_tuples_are_the_documented_matrix(self) -> None:
        self.assertTrue(poseidon2_mod._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(poseidon2_mod._EMITTER_BACKENDS, ("cpu", "gpu"))
        self.assertEqual(poseidon2_mod._PERMUTE_EMITTER_BACKENDS, ("cpu",))

    def test_each_question_reads_its_own_tuple(self) -> None:
        with mock.patch.object(poseidon2_mod, "_EMITTER_BACKENDS", ("nonesuch",)):
            self.assertFalse(poseidon2_mod._drives_the_primitive())
            self.assertEqual(
                poseidon2_mod._routes_to_dedicated_emitter(),
                frx.default_backend() in poseidon2_mod._PERMUTE_EMITTER_BACKENDS,
            )
        with mock.patch.object(
            poseidon2_mod, "_PERMUTE_EMITTER_BACKENDS", ("nonesuch",)
        ):
            self.assertFalse(poseidon2_mod._routes_to_dedicated_emitter())
            self.assertEqual(
                poseidon2_mod._drives_the_primitive(),
                frx.default_backend() in poseidon2_mod._EMITTER_BACKENDS,
            )

    def test_an_unrouted_permute_marker_keeps_the_envelope_expandable(self) -> None:
        # The regression this split exists to prevent: a backend that drives the
        # primitive but routes no permute marker must still report DEDICATED, or
        # every envelope over it (sponge, Merkle compress) falls back to the
        # generic absorb and loses its kernel.
        with mock.patch.object(poseidon2_mod, "_PERMUTE_EMITTER_BACKENDS", ()):
            perm = koalabear16_perm()
        self.assertEqual(perm.fused_region_marker, (FUSED_REGION_MARKER, 0))
        self.assertIs(perm.fusion_path, FusionPath.DEDICATED)
        self.assertTrue(perm.fusion_path.is_one_kernel)
        # The envelope ABI stays populated: an inert spec names no layout for an
        # emitter to read, which is how the loss used to show up.
        operands, _, attrs = perm.fused_region_spec(fnp.arange(16, dtype=F))
        self.assertTrue(operands)
        self.assertEqual(attrs["permutation"], "poseidon2")

    def test_a_backend_that_cannot_drive_it_reports_generic(self) -> None:
        with mock.patch.object(poseidon2_mod, "_EMITTER_BACKENDS", ()):
            perm = koalabear16_perm()
        self.assertIs(perm.fusion_path, FusionPath.GENERIC)
        self.assertFalse(perm.fusion_path.is_one_kernel)


if __name__ == "__main__":
    absltest.main()

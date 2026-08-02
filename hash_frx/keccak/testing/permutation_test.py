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

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hash_frx.fusion import FUSED_REGION_MARKER
from hash_frx.keccak.permutation import KeccakF1600, _permute_body, _rounds
from hash_frx.keccak.testing.reference import (
    from_state,
    keccak_f1600,
    to_state,
)
from hash_frx.permutation import Permutation
from hash_frx.testing.fusion_ready import assert_fusion_ready
from hash_frx.testing.jit_cache import assert_single_trace

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
        k = KeccakF1600()
        text = frx.jit(k.permute).lower(_device_state(_STATES["zeros"])).as_text()
        self.assertEqual(text.count("stablehlo.composite"), 1)
        self.assertIn(f'"{FUSED_REGION_MARKER}"', text)

    def test_the_marked_decomposition_is_fusion_ready(self) -> None:
        # The generic rewriter accepts a straight-line element-wise body only, so
        # the decomposition inside the marker is what has to hold to that. This
        # is the shared whitelist rather than a local blacklist: it also catches
        # `call`, `scatter`, `dot` and anything else new, and it reads the
        # LOWERED module — a compiled one has already inlined the calls that a
        # marked body may not contain.
        assert_fusion_ready(_rounds, _device_state(_STATES["zeros"]))

    def test_the_body_traces_once_across_instances(self) -> None:
        # The zone's cache is keyed on the aval, and `KeccakF1600` carries no
        # parameters, so freshly built instances must share one trace.
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


if __name__ == "__main__":
    absltest.main()

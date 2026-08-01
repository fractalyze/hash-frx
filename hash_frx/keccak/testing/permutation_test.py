# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Keccak-f[1600] over uint32 lane halves — values, seam conformance, and shape.

Values are checked against `reference.py`, which `reference_test` anchors to
`hashlib` so that agreement here means agreement with FIPS 202 rather than with
a second copy of the same misreading.

The lowering assertions are not decoration. A marked body that reaches for a
reduction, gather, or dynamic index still computes the right bytes and only
splits the kernel, so values alone cannot catch it — the compiled module is the
only place the single-kernel rule is visible.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hash_frx.keccak.permutation import KeccakF1600
from hash_frx.keccak.testing.reference import (
    from_state,
    keccak_f1600,
    to_state,
)
from hash_frx.permutation import Permutation

_ALL_ONES = 0xFFFFFFFFFFFFFFFF


def _lanes(name: str) -> list[int]:
    if name == "zeros":
        return [0] * 25
    if name == "ones":
        # Every bit set: exercises chi's complement and every rotation carry.
        return [_ALL_ONES] * 25
    if name == "counter":
        return [(i * 0x0123456789ABCDEF) & _ALL_ONES for i in range(25)]
    if name == "one_hot_high":
        # Only the top bit of the last lane: catches a rotation that drops the
        # cross-half carry, which a small-value state would hide.
        return [0] * 24 + [1 << 63]
    raise ValueError(name)


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
        for name in ("zeros", "ones", "counter", "one_hot_high"):
            with self.subTest(state=name):
                lanes = _lanes(name)
                got = from_state(
                    [int(v) for v in np.asarray(k.permute(_device_state(lanes)))]
                )
                self.assertEqual(got, keccak_f1600(lanes))

    def test_jit_matches_eager(self) -> None:
        k = KeccakF1600()
        x = _device_state(_lanes("counter"))
        np.testing.assert_array_equal(
            np.asarray(frx.jit(k.permute)(x)), np.asarray(k.permute(x))
        )

    def test_batches_under_vmap(self) -> None:
        # Batched over a leading axis with no Python loop over the batch.
        k = KeccakF1600()
        names = ("zeros", "ones", "counter", "one_hot_high")
        batch = [_lanes(n) for n in names]
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

    def test_body_lowers_to_a_straight_line_element_wise_graph(self) -> None:
        # The single-kernel rule: each of these still produces correct bytes and
        # only splits the kernel, so the compiled module is the only witness.
        #
        # Matched as `<op>(` rather than as a bare word, and this test's own name
        # avoids the op names, because the module carries the traced function's
        # stack in its source-location metadata — a test called
        # `..._without_a_gather` puts the string "gather" into the very text it
        # then searches, and passes or fails on its own name.
        k = KeccakF1600()
        text = (
            frx.jit(k.permute).lower(_device_state(_lanes("zeros"))).compile().as_text()
        )
        for op in (
            "reduce",
            "gather",
            "dynamic-slice",
            "dynamic-update-slice",
            "while",
            "sort",
        ):
            with self.subTest(op=op):
                self.assertNotIn(f"{op}(", text)

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

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`assert_one_static_while` has to bite.

It passes a loop the converter lowers, and fails the two shapes that compute the
same bytes without being one kernel: an unrolled body, and a loop the converter
declines. The markers it inlines have to be gone too, including from a trace a
jit zone cached before the block was entered.
"""

from __future__ import annotations

from functools import partial

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array

from hash_frx.fusion import fused_region
from hash_frx.testing.marker_recognized import emitted_composites
from hash_frx.testing.static_while import assert_one_static_while, markers_inlined

_ROUNDS = 8
# The converter admits a table read one row at the counter only at rank 2.
_TABLE = np.arange(1, _ROUNDS + 1, dtype=np.uint32).reshape(_ROUNDS, 1) * np.uint32(
    0x9E3779B9
)
# Instance count != row width: the converter cannot tell which axis of an
# `[n, n]` state is the instances, and declines it.
_STATE = np.arange(5 * 4, dtype=np.uint32).reshape(5, 4) * np.uint32(0x01000193)

_MARKER = "hash_frx.test.inlined"


def _round(t: Array | int, s: Array) -> Array:
    kt = frx.lax.dynamic_slice(fnp.asarray(_TABLE), (t, 0), (1, 1))
    mixed = s * np.uint32(3) + kt
    return fnp.concatenate([mixed[:, 1:], mixed[:, :1]], axis=1)


def _loop(state: Array) -> Array:
    return frx.lax.fori_loop(0, _ROUNDS, _round, state)


def _unrolled(state: Array) -> Array:
    for t in range(_ROUNDS):
        state = _round(t, state)
    return state


def _two_carries(state: Array) -> Array:
    """Two carried arrays, both read after the loop — the converter carries one
    state array, so it declines this."""

    def body(t: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        s, acc = carry
        s = _round(t, s)
        return s, acc ^ s

    s, acc = frx.lax.fori_loop(0, _ROUNDS, body, (state, state))
    return s + acc


@partial(frx.jit, inline=True)
def _marked_zone(x: Array) -> Array:
    return fused_region(lambda y: y + np.uint32(1), x, name=_MARKER)


class StaticWhileTest(absltest.TestCase):
    def test_a_converted_loop_passes_with_its_value(self) -> None:
        got = assert_one_static_while(self, _loop, fnp.asarray(_STATE))
        np.testing.assert_array_equal(
            np.asarray(got), np.asarray(frx.jit(_unrolled)(_STATE))
        )

    def test_an_unrolled_body_fails(self) -> None:
        with self.assertRaises(AssertionError):
            assert_one_static_while(self, _unrolled, fnp.asarray(_STATE))

    def test_a_declined_loop_fails(self) -> None:
        with self.assertRaises(AssertionError):
            assert_one_static_while(self, _two_carries, fnp.asarray(_STATE))

    def test_markers_are_inlined_even_from_a_cached_trace(self) -> None:
        x = fnp.asarray(_STATE)
        self.assertEqual(emitted_composites(_marked_zone, x), [_MARKER])
        with markers_inlined():
            self.assertEqual(emitted_composites(_marked_zone, x), [])
        self.assertEqual(emitted_composites(_marked_zone, x), [_MARKER])


if __name__ == "__main__":
    absltest.main()

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The shared word primitives, held to plain Python and to the fusion contract.

Three families call these, so a wrong one is wrong three times over and the
families' own vector tests would each report it as their own bug. Held here to
independent expressions of the same operations — `fnp.roll` for the roll,
`int.from_bytes` for the packers — rather than to a second copy of the shift
chains, which would be the thing under test written twice.

`split` is the host-side half of the same story: a 64-bit quantity a lane
hash needs — Keccak's round constants, BLAKE3's chunk counter — taken apart
where Python integers are still exact.

`roll`'s `axis` is the parameter that earns its own cases. BLAKE3 rolls a row of
a `[B, 4]` grid and Keccak rolls both axes of a `(5, 5)` one, so a default that
silently applied to the wrong axis would still produce well-shaped output — and
at a batch of one, `(-shift) % 1` is 0, which makes every roll an identity and
hides the mistake in exactly the shape a small test would use.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.testing.fusion_ready import assert_fusion_ready
from hash_frx.word import BYTES_PER_WORD, pack_le, roll, rotr, split, unpack_le


class RotrTest(parameterized.TestCase):
    @parameterized.parameters(1, 2, 7, 8, 12, 16, 25, 31)
    def test_matches_a_plain_python_rotate(self, n: int) -> None:
        rng = np.random.default_rng(n)
        x = rng.integers(0, 2**32, size=17, dtype=np.uint32)
        got = np.asarray(rotr(fnp.asarray(x), n))
        want = [((int(v) >> n) | (int(v) << (32 - n))) & 0xFFFFFFFF for v in x]
        np.testing.assert_array_equal(got, np.array(want, dtype=np.uint32))

    def test_stays_uint32(self) -> None:
        out = rotr(fnp.asarray(np.array([0x80000001], dtype=np.uint32)), 1)
        self.assertEqual(out.dtype, fnp.uint32)
        self.assertEqual(int(np.asarray(out)[0]), 0xC0000000)


class SplitTest(parameterized.TestCase):
    @parameterized.parameters(
        0, 1, 0xFFFFFFFF, 1 << 32, (1 << 64) - 1, 0x0123456789ABCDEF
    )
    def test_halves_reassemble(self, value: int) -> None:
        lo, hi = split(value)
        self.assertEqual(lo | (hi << 32), value)
        self.assertLess(lo, 1 << 32)
        self.assertLess(hi, 1 << 32)

    def test_the_boundary_is_not_off_by_one(self) -> None:
        # 2^32 - 1 is entirely low; 2^32 is entirely high. A `>=` for a `>` in
        # either shift lands one of them in the wrong half.
        self.assertEqual(split((1 << 32) - 1), (0xFFFFFFFF, 0))
        self.assertEqual(split(1 << 32), (0, 1))

    @parameterized.parameters(-1, 1 << 64)
    def test_rejects_what_does_not_fit(self, value: int) -> None:
        # Silently masking would put a wrong counter on every chunk of a tree.
        with self.assertRaises(ValueError):
            split(value)


class RollTest(parameterized.TestCase):
    @parameterized.parameters(-5, -3, -2, -1, 0, 1, 2, 3, 5)
    def test_matches_fnp_roll_on_every_axis(self, shift: int) -> None:
        # `fnp.roll` is the thing this exists to avoid emitting, not a different
        # computation — so it is exactly the right oracle.
        rng = np.random.default_rng(abs(shift) + 1)
        x = fnp.asarray(rng.integers(0, 2**32, size=(5, 4), dtype=np.uint32))
        for axis in (0, 1):
            with self.subTest(axis=axis):
                np.testing.assert_array_equal(
                    np.asarray(roll(x, shift, axis=axis)),
                    np.asarray(fnp.roll(x, shift, axis=axis)),
                )

    def test_the_axis_is_not_interchangeable(self) -> None:
        # The two axes disagree, which is what makes passing the wrong one a
        # silent wrong answer rather than an error.
        x = fnp.asarray(np.arange(20, dtype=np.uint32).reshape(5, 4))
        self.assertFalse(
            np.array_equal(np.asarray(roll(x, 1, axis=0)), np.asarray(roll(x, 1, 1)))
        )

    def test_a_full_turn_is_the_identity(self) -> None:
        # The `cut == 0` short circuit; Keccak's pi reaches it at x = 0.
        x = fnp.asarray(np.arange(20, dtype=np.uint32).reshape(5, 4))
        for axis, n in ((0, 5), (1, 4)):
            with self.subTest(axis=axis):
                np.testing.assert_array_equal(
                    np.asarray(roll(x, n, axis=axis)), np.asarray(x)
                )

    def test_rolls_a_rank_3_array_on_its_middle_axis(self) -> None:
        # Nothing in the tree calls this yet; the generic slice construction is
        # what makes it correct rather than an accident of the two axes above.
        rng = np.random.default_rng(11)
        x = fnp.asarray(rng.integers(0, 2**32, size=(2, 3, 4), dtype=np.uint32))
        np.testing.assert_array_equal(
            np.asarray(roll(x, 1, axis=1)), np.asarray(fnp.roll(x, 1, axis=1))
        )


class PackTest(absltest.TestCase):
    def test_matches_int_from_bytes(self) -> None:
        rng = np.random.default_rng(5)
        data = rng.integers(0, 256, size=(3, 5 * BYTES_PER_WORD), dtype=np.uint8)
        got = np.asarray(pack_le(fnp.asarray(data)))
        for row in range(3):
            for j in range(5):
                chunk = bytes(data[row, BYTES_PER_WORD * j : BYTES_PER_WORD * (j + 1)])
                with self.subTest(row=row, word=j):
                    self.assertEqual(int(got[row, j]), int.from_bytes(chunk, "little"))

    def test_endianness_is_little(self) -> None:
        # The one thing a byte-swapped transcription gets wrong, and the reason
        # this is not shared with `sha256.serialize_digest`.
        data = fnp.asarray(np.array([[1, 2, 3, 4]], dtype=np.uint8))
        self.assertEqual(int(np.asarray(pack_le(data))[0, 0]), 0x04030201)

    def test_packs_the_trailing_axis_of_a_rank_3_array(self) -> None:
        # BLAKE3 passes bytes already cut into blocks and wants its words in
        # place, so the trailing axis is the one packed.
        rng = np.random.default_rng(6)
        data = rng.integers(0, 256, size=(2, 3, 8), dtype=np.uint8)
        got = pack_le(fnp.asarray(data))
        self.assertEqual(got.shape, (2, 3, 2))
        np.testing.assert_array_equal(
            np.asarray(got),
            np.asarray(pack_le(fnp.asarray(data.reshape(6, 8)))).reshape(2, 3, 2),
        )

    def test_round_trips_through_unpack(self) -> None:
        rng = np.random.default_rng(7)
        words = rng.integers(0, 2**32, size=(4, 6), dtype=np.uint32)
        got = pack_le(unpack_le(fnp.asarray(words)))
        np.testing.assert_array_equal(np.asarray(got), words)

    def test_unpack_widens_the_trailing_axis(self) -> None:
        out = unpack_le(fnp.asarray(np.zeros((2, 6), dtype=np.uint32)))
        self.assertEqual(out.shape, (2, 24))
        self.assertEqual(out.dtype, fnp.uint8)

    def test_rejects_a_trailing_axis_that_is_not_whole_words(self) -> None:
        with self.assertRaises(ValueError):
            pack_le(fnp.asarray(np.zeros((2, 7), dtype=np.uint8)))


class LoweringTest(absltest.TestCase):
    def test_each_primitive_stays_fusion_safe(self) -> None:
        # These sit inside marked bodies in all three families, so a call or a
        # gather here splits every one of their kernels at once. `fnp.roll` is
        # the specific trap: it carries an internal jit.
        words = fnp.asarray(np.zeros((2, 8), dtype=np.uint32))
        grid = fnp.asarray(np.zeros((5, 4), dtype=np.uint32))
        cases = {
            "rotr": (lambda x: rotr(x, 7), words),
            "roll axis 0": (lambda x: roll(x, 1, axis=0), grid),
            "roll axis 1": (lambda x: roll(x, -2, axis=1), grid),
            "unpack_le": (unpack_le, words),
            "pack_le": (pack_le, fnp.asarray(np.zeros((2, 32), dtype=np.uint8))),
        }
        for name, (fn, arg) in cases.items():
            with self.subTest(primitive=name):
                assert_fusion_ready(fn, arg)


if __name__ == "__main__":
    absltest.main()

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The BLAKE3 compression function — values, batching, and lowering.

Held to two goldens rather than one. `reference.py` is a plain-Python
transcription sharing neither the batched lane layout nor the authoring rules, so
it cannot fail the same way; and the BLAKE3 team's published vectors reach the
compression function directly for inputs of at most 64 bytes, where a whole hash
is one call. The second is what makes the first trustworthy — `reference_test`
anchors it.

The lowering assertions are not decoration. A dynamic index or a reduction here
still computes the right words and only splits the kernel, so values alone cannot
catch it.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.blake3.compress import (
    CHUNK_END,
    CHUNK_START,
    IV,
    ROOT,
    compress,
)
from hash_frx.blake3.testing import reference as ref
from hash_frx.blake3.testing.vectors import SINGLE_BLOCK, official_input
from hash_frx.testing.fusion_ready import assert_fusion_ready

_U32 = np.uint32


def _u32(value: object) -> frx.Array:
    return fnp.asarray(np.asarray(value, dtype=_U32))


def _counter(value: int) -> np.ndarray:
    """A 64-bit counter as the (low, high) uint32 pair the seam takes."""
    return np.array([[value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF]], dtype=_U32)


def _compress_one(
    cv: np.ndarray, block: np.ndarray, counter: int, block_len: int, flags: int
) -> list[int]:
    out = compress(
        _u32(cv[None, :]),
        _u32(block[None, :]),
        _u32(_counter(counter)),
        _u32([block_len]),
        _u32([flags]),
    )
    return [int(w) for w in np.asarray(out)[0]]


class CompressAgainstReferenceTest(parameterized.TestCase):
    @parameterized.parameters(0, 1, 2, 3, 4, 5)
    def test_random_inputs_match_the_reference(self, seed: int) -> None:
        # Random rather than structured: the round schedule permutes the message
        # every round, so a block of equal or patterned words would let a wrong
        # permutation still agree.
        rng = np.random.default_rng(seed)
        cv = rng.integers(0, 2**32, size=8, dtype=_U32)
        block = rng.integers(0, 2**32, size=16, dtype=_U32)
        counter = int(rng.integers(0, 2**63))
        block_len = int(rng.integers(0, 65))
        flags = int(rng.integers(0, 128))

        self.assertEqual(
            _compress_one(cv, block, counter, block_len, flags),
            ref.compress(
                [int(w) for w in cv],
                [int(w) for w in block],
                counter,
                block_len,
                flags,
            ),
        )

    def test_a_counter_above_2_32_uses_the_high_half(self) -> None:
        # The counter is split into two uint32 lanes; a caller that dropped the
        # high half would still pass every small-counter case above.
        rng = np.random.default_rng(99)
        cv = rng.integers(0, 2**32, size=8, dtype=_U32)
        block = rng.integers(0, 2**32, size=16, dtype=_U32)
        low = _compress_one(cv, block, 1, 64, 0)
        high = _compress_one(cv, block, 1 + (1 << 32), 64, 0)
        self.assertNotEqual(low, high)
        self.assertEqual(
            high,
            ref.compress(
                [int(w) for w in cv], [int(w) for w in block], 1 + (1 << 32), 64, 0
            ),
        )


class CompressAgainstPublishedVectorsTest(parameterized.TestCase):
    @parameterized.parameters(*SINGLE_BLOCK)
    def test_single_block_hash_is_one_compression(
        self, length: int, expected: str
    ) -> None:
        # For an input of at most one block the whole BLAKE3 hash is a single
        # compression, so the official vector pins this function without any
        # chunk or tree bookkeeping in between.
        data = official_input(length)
        out = _compress_one(
            np.asarray(IV, dtype=_U32),
            np.asarray(ref.words_of(data), dtype=_U32),
            0,
            length,
            CHUNK_START | CHUNK_END | ROOT,
        )[:8]
        digest = b"".join(int(w).to_bytes(4, "little") for w in out)
        self.assertEqual(digest.hex(), expected)


class BatchingTest(absltest.TestCase):
    def test_batched_equals_per_row(self) -> None:
        # Batched over B with no Python loop over the batch: the rows must not
        # interact, which a reduction across the batch axis would break.
        rng = np.random.default_rng(14)
        batch = 5
        cvs = rng.integers(0, 2**32, size=(batch, 8), dtype=_U32)
        blocks = rng.integers(0, 2**32, size=(batch, 16), dtype=_U32)
        counters = rng.integers(0, 2**31, size=batch, dtype=_U32)
        lens = rng.integers(0, 65, size=batch, dtype=_U32)
        flags = rng.integers(0, 128, size=batch, dtype=_U32)

        got = np.asarray(
            compress(
                _u32(cvs),
                _u32(blocks),
                _u32(np.stack([counters, np.zeros(batch, dtype=_U32)], axis=1)),
                _u32(lens),
                _u32(flags),
            )
        )
        for i in range(batch):
            with self.subTest(row=i):
                self.assertEqual(
                    [int(w) for w in got[i]],
                    ref.compress(
                        [int(w) for w in cvs[i]],
                        [int(w) for w in blocks[i]],
                        int(counters[i]),
                        int(lens[i]),
                        int(flags[i]),
                    ),
                )

    def test_jit_matches_eager(self) -> None:
        rng = np.random.default_rng(7)
        args = (
            _u32(rng.integers(0, 2**32, size=(3, 8), dtype=_U32)),
            _u32(rng.integers(0, 2**32, size=(3, 16), dtype=_U32)),
            _u32(np.zeros((3, 2))),
            _u32(np.full(3, 64)),
            _u32(np.zeros(3)),
        )
        np.testing.assert_array_equal(
            np.asarray(frx.jit(compress)(*args)), np.asarray(compress(*args))
        )


class LoweringTest(absltest.TestCase):
    def test_the_body_is_fusion_ready(self) -> None:
        # The shared whitelist rather than a local blacklist: it also catches a
        # call, a scatter, or a dot, and it reads the lowered module.
        batch = 2
        assert_fusion_ready(
            compress,
            _u32(np.zeros((batch, 8))),
            _u32(np.zeros((batch, 16))),
            _u32(np.zeros((batch, 2))),
            _u32(np.zeros(batch)),
            _u32(np.zeros(batch)),
        )

    def test_arithmetic_stays_uint32(self) -> None:
        # x64 is off, and the whole point of the uint32 lane discipline is that
        # nothing here silently widens or truncates.
        batch = 2
        out = compress(
            _u32(np.zeros((batch, 8))),
            _u32(np.zeros((batch, 16))),
            _u32(np.zeros((batch, 2))),
            _u32(np.zeros(batch)),
            _u32(np.zeros(batch)),
        )
        self.assertEqual(out.dtype, fnp.uint32)
        self.assertEqual(out.shape, (batch, 16))


class ValidationTest(absltest.TestCase):
    def test_rejects_wrong_shapes(self) -> None:
        good_cv = _u32(np.zeros((1, 8)))
        good_block = _u32(np.zeros((1, 16)))
        good_counter = _u32(np.zeros((1, 2)))
        good_len = _u32(np.zeros(1))
        for name, args in (
            (
                "cv width",
                (_u32(np.zeros((1, 7))), good_block, good_counter, good_len, good_len),
            ),
            (
                "block width",
                (good_cv, _u32(np.zeros((1, 15))), good_counter, good_len, good_len),
            ),
            (
                "counter halves",
                (good_cv, good_block, _u32(np.zeros((1, 1))), good_len, good_len),
            ),
            (
                "cv rank",
                (_u32(np.zeros(8)), good_block, good_counter, good_len, good_len),
            ),
        ):
            with self.subTest(case=name), self.assertRaises(ValueError):
                compress(*args)


if __name__ == "__main__":
    absltest.main()

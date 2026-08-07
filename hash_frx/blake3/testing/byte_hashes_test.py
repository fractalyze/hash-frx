# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""BLAKE3 through the `ByteHash` seam.

`blake3_test` pins the bytes; this pins the surface a consumer actually codes
against — that `digest_size` is what `digest` returns, that the instance is
re-trace-safe as pytree aux, and that a tracer gets through. The seam is the
whole point of the family: a consumer reads `digest_size` and calls `digest`,
and never learns which hash it got.

One published vector per layer runs here as well as against the free function —
a single compression, a chunk chain, a parent-node tree. That is the same bytes
twice only if the class is a thin wrapper, which is the claim worth pinning: a
wrapper that truncated, re-padded or transposed would still return well-formed
32-byte digests, and would fail on any one of the three. The whole table runs in
`blake3_test`, where the implementation it delegates to lives — a wrapper with no
length-dependent code cannot pass three lengths and fail thirty-five.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.blake3.byte_hashes import Blake3
from hash_frx.blake3.testing.vectors import (
    MULTI_BLOCK,
    MULTI_CHUNK,
    SINGLE_BLOCK,
    official_input,
)
from hash_frx.byte_hash import ByteHash


def _rows(*messages: bytes) -> frx.Array:
    return fnp.asarray(np.array([list(m) for m in messages], dtype=np.uint8))


# One per group, so each layer the groups name is on the delegated path: a
# single compression (64 B), a chunk chain (129 B), a parent-node tree (1025 B).
_PER_LAYER = (SINGLE_BLOCK[-1], MULTI_BLOCK[3], MULTI_CHUNK[0])


class ForwardedVectorTest(parameterized.TestCase):
    @parameterized.parameters(*_PER_LAYER)
    def test_official_vector_through_the_seam(self, length: int, expected: str) -> None:
        got = np.asarray(Blake3().digest(_rows(official_input(length))))
        self.assertEqual(bytes(got[0]).hex(), expected)


class SeamConformanceTest(absltest.TestCase):
    def test_the_implementation_satisfies_the_byte_hash_protocol(self) -> None:
        h = Blake3()
        self.assertIsInstance(h, ByteHash)
        # Pinned rather than merely type-checked: `False` while no BLAKE3
        # emitter exists is the substantive claim the module docstring makes,
        # and a marker landing without this flag moving is silent.
        self.assertFalse(h.has_dedicated_fusion)

    def test_digest_size_matches_what_digest_returns(self) -> None:
        # Two rows, so the batch axis reaching the digest unchanged is pinned
        # here rather than in a case of its own — `blake3_test` already checks
        # rows do not interact, against the reference oracle and through the
        # tree, which is more than a delegation can break.
        msg = _rows(official_input(200), official_input(200))
        for size in (32, 131):
            with self.subTest(output_size=size):
                out = np.asarray(Blake3(size).digest(msg))
                self.assertEqual(out.shape, (2, Blake3(size).digest_size))

    def test_value_identity_keeps_the_seam_re_trace_safe(self) -> None:
        # Param-free, so equality is by type. Identity equality here would make
        # every freshly built instance a new jit cache key and silently re-trace
        # a consumer's enclosing zone.
        self.assertEqual(Blake3(), Blake3())
        self.assertEqual(hash(Blake3()), hash(Blake3()))
        self.assertNotEqual(Blake3(), object())

    def test_digest_accepts_a_tracer(self) -> None:
        # The return type is what says a consumer may hash inside its own jit,
        # and `has_dedicated_fusion` is False here — so this is the property
        # that has to be asserted rather than read off the flag. The body is
        # compiled here, so the length is kept to what the property needs: two
        # blocks, the smallest input that pads and still chains.
        #
        # Note the form. Value equality is a property of the *instance*, and a
        # bound method does not inherit it — CPython compares `__self__` by
        # identity, so `Blake3().digest != Blake3().digest` even though the two
        # instances are equal. A consumer holds the jitted wrapper, or passes
        # the instance itself where equality is what is read.
        rows = _rows(official_input(65))
        np.testing.assert_array_equal(
            np.asarray(frx.jit(Blake3().digest)(rows)),
            np.asarray(Blake3().digest(rows)),
        )


if __name__ == "__main__":
    absltest.main()

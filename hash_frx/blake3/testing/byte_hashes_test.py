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

**Every row runs every case.** The three modes share a body and differ by which
parameters they carry, so what is worth testing per row is exactly what a row
adds: that its parameter reaches the hash, and that it reaches `__eq__`. A row
that forgot the second returns right bytes forever and hands one key's trace to
another key's caller.
"""

from __future__ import annotations

from collections.abc import Callable

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.blake3.byte_hashes import Blake3, Blake3DeriveKey, Blake3Keyed
from hash_frx.blake3.testing.vectors import (
    ALL_LENGTHS,
    CONTEXT,
    DERIVE_KEY,
    KEY,
    KEYED,
    PER_LAYER_LENGTHS,
    official_input,
    rows,
)
from hash_frx.byte_hash import ByteHash


def _rows(*messages: bytes) -> frx.Array:
    return fnp.asarray(np.array([list(m) for m in messages], dtype=np.uint8))


# Each row with the arguments that name its mode, and the published column it
# reproduces. Everything below is parameterized over this, because the three are
# one body with different parameters and a case that ran for only one of them
# would be a case that stops covering the family the moment a fourth row lands.
_ROWS = (
    ("hash", Blake3, (), ALL_LENGTHS),
    ("keyed", Blake3Keyed, (KEY,), KEYED),
    ("derive_key", Blake3DeriveKey, (CONTEXT,), DERIVE_KEY),
)

_PER_LAYER = tuple(
    (f"{name}_{length}", cls, args, length, expected)
    for name, cls, args, table in _ROWS
    for length, expected in rows(table, PER_LAYER_LENGTHS)
)

_EVERY_ROW = tuple((name, cls, args) for name, cls, args, _ in _ROWS)


class ForwardedVectorTest(parameterized.TestCase):
    @parameterized.named_parameters(*_PER_LAYER)
    def test_official_vector_through_the_seam(
        self,
        cls: Callable[..., ByteHash],
        args: tuple[object, ...],
        length: int,
        expected: str,
    ) -> None:
        # One per layer, so each layer the groups name is on the delegated path:
        # a single compression (64 B), a chunk chain (129 B), a parent-node tree
        # (1025 B).
        got = np.asarray(cls(*args).digest(_rows(official_input(length))))
        self.assertEqual(bytes(got[0]).hex(), expected)


class SeamConformanceTest(parameterized.TestCase):
    @parameterized.named_parameters(*_EVERY_ROW)
    def test_the_implementation_satisfies_the_byte_hash_protocol(
        self, cls: Callable[..., ByteHash], args: tuple[object, ...]
    ) -> None:
        h = cls(*args)
        self.assertIsInstance(h, ByteHash)
        # Pinned rather than merely type-checked: `False` while no BLAKE3
        # emitter exists is the substantive claim the module docstring makes,
        # and a marker landing without this flag moving is silent.
        self.assertFalse(h.has_dedicated_fusion)

    @parameterized.named_parameters(*_EVERY_ROW)
    def test_digest_size_matches_what_digest_returns(
        self, cls: Callable[..., ByteHash], args: tuple[object, ...]
    ) -> None:
        # Two rows, so the batch axis reaching the digest unchanged is pinned
        # here rather than in a case of its own — `blake3_test` already checks
        # rows do not interact, against the reference oracle and through the
        # tree, which is more than a delegation can break.
        msg = _rows(official_input(200), official_input(200))
        for size in (32, 131):
            with self.subTest(output_size=size):
                out = np.asarray(cls(*args, size).digest(msg))
                self.assertEqual(out.shape, (2, cls(*args, size).digest_size))

    @parameterized.named_parameters(*_EVERY_ROW)
    def test_the_standards_own_output_length_is_the_default(
        self, cls: Callable[..., ByteHash], args: tuple[object, ...]
    ) -> None:
        # A default is permitted here and refused for `Shake256` because the
        # standard names one for each of BLAKE3's three modes; taking it is not
        # the same as taking 32 by accident, so it is asserted.
        self.assertEqual(cls(*args).digest_size, 32)

    @parameterized.named_parameters(*_EVERY_ROW)
    def test_rejects_an_empty_output(
        self, cls: Callable[..., ByteHash], args: tuple[object, ...]
    ) -> None:
        # A zero-length digest is a `digest_size` a consumer would size a buffer
        # from. It is refused at construction rather than at the call, so the
        # caller learns before it holds an unusable hash.
        with self.assertRaises(ValueError):
            cls(*args, 0)

    @parameterized.named_parameters(*_EVERY_ROW)
    def test_the_output_length_is_part_of_the_value(
        self, cls: Callable[..., ByteHash], args: tuple[object, ...]
    ) -> None:
        # Two lengths are two hashes rather than one asked twice. Identity
        # equality would make every fresh instance a new jit cache key and
        # silently re-trace a consumer's zone; by-type equality would do the
        # reverse and hand a consumer that asked for 64 bytes the trace built
        # for 32. Neither errors, so both have to be pinned here.
        self.assertEqual(cls(*args), cls(*args))
        self.assertEqual(hash(cls(*args)), hash(cls(*args)))
        self.assertEqual(cls(*args, 64), cls(*args, 64))
        self.assertEqual(hash(cls(*args, 64)), hash(cls(*args, 64)))
        self.assertNotEqual(cls(*args, 32), cls(*args, 64))
        self.assertNotEqual(cls(*args), object())

    @parameterized.named_parameters(*_EVERY_ROW)
    def test_digest_accepts_a_tracer(
        self, cls: Callable[..., ByteHash], args: tuple[object, ...]
    ) -> None:
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
        message = _rows(official_input(65))
        np.testing.assert_array_equal(
            np.asarray(frx.jit(cls(*args).digest)(message)),
            np.asarray(cls(*args).digest(message)),
        )


class ModeIdentityTest(absltest.TestCase):
    """What each row adds to the value surface, and that the rows are distinct."""

    def test_the_key_is_part_of_the_value(self) -> None:
        # The one thing a keyed row must not inherit unchanged. Comparing on
        # `digest_size` alone makes two keys one hash: as pytree aux that serves
        # one key's trace to the other key's caller, silently and forever.
        other = bytes(len(KEY))
        self.assertNotEqual(Blake3Keyed(KEY), Blake3Keyed(other))
        self.assertNotEqual(Blake3Keyed(KEY, 64), Blake3Keyed(other, 64))

    def test_the_context_is_part_of_the_value(self) -> None:
        self.assertNotEqual(Blake3DeriveKey("one"), Blake3DeriveKey("two"))

    def test_a_str_context_and_its_utf8_bytes_are_one_hash(self) -> None:
        # They derive identical bytes, so treating them as two would make one of
        # them a second jit cache key for no computation.
        self.assertEqual(Blake3DeriveKey(CONTEXT), Blake3DeriveKey(CONTEXT.encode()))
        self.assertEqual(
            hash(Blake3DeriveKey(CONTEXT)), hash(Blake3DeriveKey(CONTEXT.encode()))
        )

    def test_the_rows_are_three_hashes_and_not_one(self) -> None:
        # They share a base and a `digest_size`, which is exactly the shape in
        # which a value comparison over the base's parameters alone would call
        # them equal.
        self.assertNotEqual(Blake3(), Blake3Keyed(KEY))
        self.assertNotEqual(Blake3(), Blake3DeriveKey(CONTEXT))
        self.assertNotEqual(Blake3Keyed(KEY), Blake3DeriveKey(CONTEXT))

    def test_rejects_a_key_that_is_not_32_bytes(self) -> None:
        # Refused at construction, where the caller can still act on it, rather
        # than at the first `digest`.
        for name, size in (("short", 31), ("long", 33), ("empty", 0)):
            with self.subTest(case=name), self.assertRaises(ValueError):
                Blake3Keyed(bytes(size))


if __name__ == "__main__":
    absltest.main()

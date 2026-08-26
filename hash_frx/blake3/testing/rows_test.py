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

**And every case runs on every mode.** A case that named one mode would stop
covering the family, since the three are one body with different parameters.

Two cases run against the `blake3` binding rather than the rows, and neither
duplicates the above:

- **The whole published table**, in all three modes, at the digest length and the
  extended one. The device rows get one vector per layer here because each length
  is a compile; a native hash is free, so there is no reason to pin the host rows
  against three lengths when thirty-five cost nothing.
- **The differential sweep**, device against host at lengths the table does not
  publish. That is what a third-party implementation buys and a second in-tree
  transcription would not: the published vectors cover 35 lengths, and a
  device-only bug at an unlucky remainder between them survives every other test
  in this package. `docs/reference/conventions.md` asks for exactly this
  independence when it refuses a pin against another implementation in this tree.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import blake3 as blake3_py
import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.blake3.modes import BLOCK_LEN
from hash_frx.blake3.rows import (
    Blake3,
    Blake3DeriveKey,
    Blake3Keyed,
)
from hash_frx.blake3.testing.emitter import HAS_BLAKE3_EMITTER
from hash_frx.blake3.testing.vectors import (
    ALL_LENGTHS,
    CONTEXT,
    DERIVE_KEY,
    EXTENDED,
    EXTENDED_DERIVE_KEY,
    EXTENDED_KEYED,
    KEY,
    KEYED,
    PER_LAYER_LENGTHS,
    official_input,
    rows,
)
from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.testing.oracle import oracle_digest

# The published extended column is 131 bytes and its head is the digest, so one
# table anchors both output lengths (`vectors._digests` takes the head rather
# than transcribing it).
_EXTENDED_LEN = 131


def _rows(*messages: bytes) -> frx.Array:
    return fnp.asarray(np.array([list(m) for m in messages], dtype=np.uint8))


@cache
def _official_bytes(length: int) -> bytes:
    """The vectors' input at `length`.

    Cached because the sweeps below run every published length in three modes,
    and the vectors build their input a byte at a time — at 102400 bytes that
    dominates the hash it is feeding.
    """
    return official_input(length)


def _official_batch(length: int) -> np.ndarray:
    """The vectors' input at `length`, as a one-message batch."""
    return np.frombuffer(_official_bytes(length), dtype=np.uint8).reshape(1, length)


# Each mode as its two rows, the arguments that name it, and the two published
# columns it reproduces. Everything below is parameterized over this, because the
# modes are one body with different parameters and a case that ran for only one of
# them would stop covering the family the moment a fourth row lands.
# `oracle` is the BLAKE3 team's own binding for that mode, taking a message and
# an output length. It is what the deleted host rows wrapped, and keeping it
# here is what keeps the differential sweep below a claim about the
# implementation rather than about this tree agreeing with itself.
_ROWS = (
    (
        "hash",
        Blake3,
        (),
        ALL_LENGTHS,
        EXTENDED,
        lambda m, n: blake3_py.blake3(m).digest(n),
    ),
    (
        "keyed",
        Blake3Keyed,
        (KEY,),
        KEYED,
        EXTENDED_KEYED,
        lambda m, n: blake3_py.blake3(m, key=KEY).digest(n),
    ),
    (
        "derive_key",
        Blake3DeriveKey,
        (CONTEXT,),
        DERIVE_KEY,
        EXTENDED_DERIVE_KEY,
        lambda m, n: blake3_py.blake3(m, derive_key_context=CONTEXT).digest(n),
    ),
)

# Every row — what the seam sweeps run over.
_EVERY_ROW = tuple((name, cls, args) for name, cls, args, _, _, _ in _ROWS)

# One mode's row beside its oracle, for the cases that compare them.
_PAIRS = tuple((name, cls, args, oracle) for name, cls, args, _, _, oracle in _ROWS)

# How far the device rows reach, which is a question about compile cost rather
# than about the seam. Where the BLAKE3 emitter is present a marked digest is one
# custom fusion and its body is never codegen'd, so the device rows run what the
# host rows run: the published lengths through `CHUNK_LEN + 1`, the tree layer.
#
# Where it is absent the marker inlines and a device digest compiles its whole
# unrolled body, at a cost super-linear in the compression count — ~0.3s at one
# block, ~2s at three, and minutes at the two chunks `CHUNK_LEN + 1` reaches — so
# the rows stop below the tree layer. What that gives up is byte-equality over a
# tree on this leg; `blake3_test` covers the tree against the decomposition the
# marker inlines to, which is the same code without the compile.
_DEVICE_MAX_LEN = 1200 if HAS_BLAKE3_EMITTER else 2 * BLOCK_LEN + 1
_DEVICE_LAYER_LENGTHS = tuple(
    length for length in PER_LAYER_LENGTHS if length <= _DEVICE_MAX_LEN
)

_PER_LAYER = tuple(
    (f"{name}_{length}", cls, args, length, expected)
    for name, cls, args, table, _, _ in _ROWS
    for length, expected in rows(table, _DEVICE_LAYER_LENGTHS)
)

# Every published length in every mode, for the oracle alone — affordable
# because a native hash costs no compile.
_ORACLE_TABLE = tuple(
    (f"{name}_{length}", oracle, length, expected)
    for name, _, _, _, extended, oracle in _ROWS
    for length, expected in extended
)

# Lengths the published table does not carry, which is the whole point of the
# differential sweep. Drawn rather than chosen so they are not lengths anybody
# reasoned about, and seeded so a failure reproduces.
_PUBLISHED_LENGTHS = frozenset(length for length, _ in ALL_LENGTHS)


def _unpublished_lengths(count: int, high: int, seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    drawn: list[int] = []
    while len(drawn) < count:
        length = int(rng.integers(1, high))
        if length not in _PUBLISHED_LENGTHS and length not in drawn:
            drawn.append(length)
    return tuple(drawn)


# Kept to six on purpose: every length is a separate device compile, and the
# property under test does not scale with the tree. The ceiling rides
# `_DEVICE_MAX_LEN` for the reason stated there — two chunks where the emitter
# carries the compile, two blocks where each length would pay for its own
# unrolled body.
_RANDOM_LENGTHS = _unpublished_lengths(count=6, high=_DEVICE_MAX_LEN, seed=0)


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


class OracleVectorTest(parameterized.TestCase):
    """The oracle against the whole published table, at both output lengths.

    Affordable here and not on the device rows: a native hash costs no compile,
    so this gets thirty-five lengths instead of three. It is what makes the
    differential sweep below meaningful — an oracle anchored to the standard is
    a partner worth disagreeing with.
    """

    @parameterized.named_parameters(*_ORACLE_TABLE)
    def test_official_vector_at_both_output_lengths(
        self,
        oracle: Callable[[bytes, int], bytes],
        length: int,
        expected: str,
    ) -> None:
        msg = _official_bytes(length)
        with self.subTest(output_size=32):
            self.assertEqual(oracle(msg, 32).hex(), expected[: 2 * 32])
        with self.subTest(output_size=_EXTENDED_LEN):
            # 131 bytes is three output blocks and a 3-byte remainder, so the
            # stream running past one block and the partial tail are both here.
            self.assertEqual(oracle(msg, _EXTENDED_LEN).hex(), expected)


class DifferentialTest(parameterized.TestCase):
    """Device against host, on random messages at lengths nobody published.

    The one case in this package whose partner is not this tree: the `blake3`
    binding is the BLAKE3 team's own, wrapping the reference the vectors are
    generated from. Agreement is therefore evidence about the implementation
    rather than about one reading of the spec being applied twice.
    """

    @parameterized.named_parameters(*_PAIRS)
    def test_the_row_agrees_with_the_oracle_at_unpublished_lengths(
        self,
        cls: Callable[..., ByteHash],
        args: tuple[object, ...],
        oracle: Callable[[bytes, int], bytes],
    ) -> None:
        rng = np.random.default_rng(7)
        for length in _RANDOM_LENGTHS:
            # Two rows, so a length that packed the batch axis wrongly fails
            # here rather than passing on a batch of one.
            msg = rng.integers(0, 256, size=(2, length), dtype=np.uint8)
            with self.subTest(length=length):
                np.testing.assert_array_equal(
                    np.asarray(cls(*args).digest(msg)),
                    oracle_digest(lambda b: oracle(b, 32), 32, msg),
                )


class SeamConformanceTest(parameterized.TestCase):
    @parameterized.named_parameters(*_EVERY_ROW)
    def test_the_implementation_satisfies_the_byte_hash_protocol(
        self, cls: Callable[..., ByteHash], args: tuple[object, ...]
    ) -> None:
        self.assertIsInstance(cls(*args), ByteHash)

    @parameterized.named_parameters(*_EVERY_ROW)
    def test_a_device_row_derives_its_path_from_the_backend(
        self, cls: Callable[..., ByteHash], args: tuple[object, ...]
    ) -> None:
        # Pinned against the shipped switch rather than a backend tuple of its
        # own: re-hardcode detection lives with the central matrix
        # (`testing.fusion_path_test.MatrixFactsTest`), so a third spelling of
        # the tuple here would only add an edit site.
        expected = FusionPath.from_routing(HAS_BLAKE3_EMITTER)
        h = cls(*args)
        self.assertIs(h.fusion_path, expected)

    @parameterized.named_parameters(*_EVERY_ROW)
    def test_digest_size_matches_what_digest_returns(
        self, cls: Callable[..., ByteHash], args: tuple[object, ...]
    ) -> None:
        # Two rows, so the batch axis reaching the digest unchanged is pinned
        # here rather than in a case of its own — `blake3_test` already checks
        # rows do not interact, against the reference oracle and through the
        # tree, which is more than a delegation can break.
        #
        # The batch is a device array for both substrates, which also pins that
        # a host row takes one: a consumer holding device bytes should not have
        # to convert them to reach the host path.
        #
        # One block, because the claim is the *shape* of the result and a longer
        # message proves it no harder — while costing a device row four
        # compression bodies to compile instead of one, twice over for the two
        # output lengths (`_DEVICE_LAYER_LENGTHS` above measures the curve).
        msg = _rows(official_input(BLOCK_LEN), official_input(BLOCK_LEN))
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
        # The return type is what says a consumer may hash inside its own jit —
        # the return type is the authority, so the property is
        # asserted rather than read off the attribute. The body is compiled
        # here, so the length is kept to what the property needs: two blocks,
        # the smallest input that pads and still chains.
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


class ModeIdentityTest(parameterized.TestCase):
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


class DeriveKeyContextTest(absltest.TestCase):
    """The row takes a context the oracle binding cannot."""

    def test_the_row_takes_a_context_the_binding_refuses(self) -> None:
        # `blake3.blake3(..., derive_key_context=...)` takes a `str`, so the
        # binding cannot express a non-UTF-8 context while the row hashes it as
        # bytes. Pinned from both sides, because the narrowing is the binding's
        # and not the standard's — and it is why the differential sweep above
        # runs on UTF-8 contexts only.
        with self.assertRaises((UnicodeDecodeError, TypeError, ValueError)):
            blake3_py.blake3(b"", derive_key_context=b"\xff\xfe not utf-8")  # type: ignore[arg-type]
        self.assertEqual(Blake3DeriveKey(b"\xff\xfe").digest_size, 32)


class EmptyBatchTest(absltest.TestCase):
    """A zero-row batch is a valid batch (#211): every row returns
    uint8 [0, digest_size] instead of failing in a block-count reshape."""

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        got = np.asarray(Blake3().digest(fnp.zeros((0, 64), dtype=fnp.uint8)))
        self.assertEqual(got.shape, (0, 32))
        self.assertEqual(got.dtype, np.uint8)


if __name__ == "__main__":
    absltest.main()

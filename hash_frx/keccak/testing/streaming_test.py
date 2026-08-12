# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Incremental SHAKE — byte-exact against `hashlib`, and pytree-threadable.

`hashlib.shake_*` is the agnostic golden, the same one `byte_hashes_test` uses:
it implements FIPS 202 without sharing a line with this tree, so agreement is
agreement with the standard.

Two properties matter here and neither is a value check in the ordinary sense.
The first is that *where the message is split cannot matter* — the schedule
carries a pending buffer whose length is traced, and a split that lands
anywhere but a rate boundary is what exercises it. The second is that the state
threads a `@jit` boundary and a `lax.scan` carry, which is the whole reason it is
a fixed-shape pytree rather than a Python object; a rejection-sampling loop is a
scan whose carry is a squeeze state, so that shape is tested directly rather
than by proxy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import tree_util

from hash_frx.keccak.byte_hashes import (
    SHA3_256_RATE,
    SHA3_SUFFIX,
    SHAKE128_RATE,
    SHAKE256_RATE,
)
from hash_frx.keccak.streaming import (
    ShakeAbsorb,
    ShakeSqueeze,
    shake128_init,
    shake256_init,
    shake_init,
)

# Long enough to span several blocks at either rate.
_MESSAGE = bytes((i * 7 + 3) & 0xFF for i in range(400))

_Init = Callable[[], ShakeAbsorb]
# `hashlib.shake_*` returns a private `_Hash`-alike; what is used is `.digest(n)`.
_Reference = Callable[[bytes], Any]

# (name, init, rate, hashlib factory)
_CASES = (
    ("shake128", shake128_init, SHAKE128_RATE, hashlib.shake_128),
    ("shake256", shake256_init, SHAKE256_RATE, hashlib.shake_256),
)


def _u8(data: bytes) -> frx.Array:
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _absorb_all(init: _Init, chunks: list[bytes]) -> ShakeAbsorb:
    state = init()
    for chunk in chunks:
        state = state.absorb(_u8(chunk))
    return state


def _squeeze_once(init: _Init, chunks: list[bytes], nbytes: int) -> bytes:
    _, out = _absorb_all(init, chunks).finalize().squeeze(nbytes)
    return bytes(np.asarray(out))


class ShakeStreamTest(parameterized.TestCase):
    @parameterized.parameters(*_CASES)
    def test_single_absorb_matches_hashlib(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # Lengths straddling the rate: empty, one short of a block (so the domain
        # suffix and the pad's closing bit share a byte), exactly a block (so the
        # padding takes a whole extra one), and multi-block.
        for length in (0, 1, rate - 1, rate, rate + 1, 2 * rate, 2 * rate + 1, 400):
            with self.subTest(length=length):
                msg = _MESSAGE[:length]
                self.assertEqual(
                    _squeeze_once(init, [msg], 32), reference(msg).digest(32)
                )

    @parameterized.parameters(*_CASES)
    def test_every_split_point_gives_the_same_digest(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # The point of the pending buffer: the cut lands on every residue mod the
        # rate exactly once, plus two past the wrap, so no pending length goes
        # unexercised. Each cut is a distinct static length and therefore its own
        # trace, so sweeping the residues twice doubles the cost for no coverage.
        msg = _MESSAGE[: rate + 2]
        want = reference(msg).digest(32)
        for cut in range(len(msg) + 1):
            with self.subTest(cut=cut):
                self.assertEqual(_squeeze_once(init, [msg[:cut], msg[cut:]], 32), want)

    @parameterized.parameters(*_CASES)
    def test_three_way_split_gives_the_same_digest(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # Two pending carries in a row, so the second absorb starts from a
        # non-zero pending length rather than from a fresh state.
        msg = _MESSAGE[: 2 * rate + 8]
        want = reference(msg).digest(32)
        for a in (0, 1, rate - 1, rate, rate + 1):
            for b in (0, 1, rate // 2, rate):
                with self.subTest(first=a, second=b):
                    self.assertEqual(
                        _squeeze_once(
                            init, [msg[:a], msg[a : a + b], msg[a + b :]], 32
                        ),
                        want,
                    )

    @parameterized.parameters(*_CASES)
    def test_repeated_squeezes_equal_one_squeeze_of_the_total(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # The rejection-sampling access pattern: take a little at a time and get
        # the same stream as one long take. Splits chosen to leave the offset
        # mid-block, on a boundary, and past several blocks.
        for parts in (
            [32, 32],
            [1, 31, 32],
            [rate, rate],
            [rate - 1, 1, 1],
            [200, 200],
        ):
            with self.subTest(parts=tuple(parts)):
                squeezer = _absorb_all(init, [_MESSAGE]).finalize()
                got = b""
                for part in parts:
                    squeezer, out = squeezer.squeeze(part)
                    got += bytes(np.asarray(out))
                self.assertEqual(got, reference(_MESSAGE).digest(sum(parts)))

    def test_the_suffix_is_the_domain_and_not_a_constant(self) -> None:
        # What this streams is a sponge; SHAKE is the domain the two `init`
        # helpers pick. `shake_init` takes the FIPS 202 domain byte, and at
        # SHAKE256's rate the SHA-3 byte gives SHA3-256 — the standard's own
        # construction, and the reason the suffix is a parameter rather than a
        # constant. Nothing else reaches this argument.
        state = shake_init(SHA3_256_RATE, SHA3_SUFFIX).absorb(_u8(_MESSAGE))
        _, out = state.finalize().squeeze(32)
        self.assertEqual(bytes(np.asarray(out)), hashlib.sha3_256(_MESSAGE).digest())


class PytreeThreadingTest(parameterized.TestCase):
    """The reason this is a registered dataclass rather than a Python object."""

    @parameterized.parameters(*_CASES)
    def test_treedef_is_stable_across_absorbs(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # An unstable treedef re-traces the enclosing zone on every hand-off,
        # which does not error — it just makes every call slow.
        fresh = tree_util.tree_structure(init())
        used = tree_util.tree_structure(init().absorb(_u8(_MESSAGE[:100])))
        self.assertEqual(fresh, used)

    def test_the_two_rates_are_distinct_treedefs(self) -> None:
        # `rate` rides as static aux, so a SHAKE128 state cannot be substituted
        # into a zone traced for SHAKE256.
        self.assertNotEqual(
            tree_util.tree_structure(shake128_init()),
            tree_util.tree_structure(shake256_init()),
        )

    @parameterized.parameters(*_CASES)
    def test_the_whole_pipeline_traces(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        @frx.jit
        def run(data: frx.Array) -> frx.Array:
            _, out = init().absorb(data).finalize().squeeze(64)
            return out

        got = bytes(np.asarray(run(_u8(_MESSAGE))))
        self.assertEqual(got, reference(_MESSAGE).digest(64))

    @parameterized.parameters(*_CASES)
    def test_absorb_threads_a_scan_carry(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        chunks = np.frombuffer(_MESSAGE, dtype=np.uint8).reshape(8, 50)

        def body(carry: ShakeAbsorb, chunk: frx.Array) -> tuple[ShakeAbsorb, None]:
            return carry.absorb(chunk), None

        final, _ = frx.lax.scan(body, init(), fnp.asarray(chunks))
        _, out = final.finalize().squeeze(32)
        self.assertEqual(bytes(np.asarray(out)), reference(_MESSAGE).digest(32))

    @parameterized.parameters(*_CASES)
    def test_squeeze_threads_a_scan_carry(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # This is the shape ML-DSA's sampling loop has: squeeze a block, test it,
        # squeeze again — with the sponge as the carry.
        squeezer = _absorb_all(init, [_MESSAGE]).finalize()

        def body(carry: ShakeSqueeze, _: None) -> tuple[ShakeSqueeze, frx.Array]:
            carry, out = carry.squeeze(32)
            return carry, out

        _, outs = frx.lax.scan(body, squeezer, None, length=5)
        got = b"".join(bytes(row) for row in np.asarray(outs))
        self.assertEqual(got, reference(_MESSAGE).digest(160))


class ShakeStreamValidationTest(absltest.TestCase):
    def test_absorb_rejects_a_batched_message(self) -> None:
        # The streaming state is one sponge; a batch axis would silently absorb
        # the rows concatenated.
        with self.assertRaises(ValueError):
            shake256_init().absorb(fnp.zeros((2, 64), dtype=fnp.uint8))

    def test_absorb_rejects_a_message_that_is_not_bytes(self) -> None:
        # Coerced instead, a value above 255 is truncated and the sponge returns
        # a well-formed digest of a message the caller never passed.
        with self.assertRaises(TypeError):
            shake256_init().absorb(fnp.asarray(np.array([300, 65], dtype=np.int32)))

    def test_squeeze_rejects_an_empty_request(self) -> None:
        with self.assertRaises(ValueError):
            shake256_init().finalize().squeeze(0)


if __name__ == "__main__":
    absltest.main()

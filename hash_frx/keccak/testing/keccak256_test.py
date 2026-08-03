# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Keccak-256 — the published Ethereum vectors, and the padding byte pinned.

Separate from `fips202_test` because Keccak-256 is not a FIPS 202 function and
its golden is not `hashlib`: the standard library implements the *changed*
padding only. The vectors below are the published Ethereum ones, and
`HostKeccak256` gives a second, independent computation of the same bytes.

**The divergence test is why this module exists.** Keccak-256 and SHA3-256 share
a rate, a capacity, a permutation, an absorb, a padding rule, and an output size;
they differ in one byte of one constant. Every positive vector here would still
pass if that byte were wired to `0x06`, because then the implementation would be
a correct SHA3-256 — just not the hash anyone asked for. Only an assertion that
the two *disagree* catches it, which is why it is a test rather than a comment,
and it is the most-copied bug in Keccak implementations.
"""

from __future__ import annotations

import hashlib

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.byte_hash import ByteHash
from hash_frx.keccak.fips202 import (
    KECCAK256_RATE,
    KECCAK256_SUFFIX,
    SHA3_256_RATE,
    SHA3_256_SUFFIX,
    Keccak256,
    Sha3_256,
)
from hash_frx.keccak.testing.host_keccak256 import HostKeccak256

# Published Ethereum Keccak-256 vectors. The empty string is the one #41 names,
# and is the value every EVM implementation agrees on for `KECCAK256("")`.
_VECTORS = (
    (b"", "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
    (b"abc", "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
    (b"hello", "1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8"),
    (
        b"The quick brown fox jumps over the lazy dog",
        "4d741b6f1eb29cb2a9b9911c82f56fa8d73b04959d3d9d222895df6c0b28aa15",
    ),
)

# The same absorb boundaries `fips202_test` sweeps: the padding rule is shared,
# so the block-boundary cases that can break it are shared too.
_LENGTHS = (0, 1, 135, 136, 137, 300)


def _rows(msg: bytes) -> np.ndarray:
    return np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))


def _message(length: int) -> np.ndarray:
    return (np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)).reshape(1, length)


class PaddingDivergenceTest(parameterized.TestCase):
    """SHA3-256 and Keccak-256 must never agree. This is the issue's whole point."""

    @parameterized.parameters(*_LENGTHS)
    def test_sha3_and_keccak_disagree_on_the_same_input(self, length: int) -> None:
        msg = _message(length)
        sha3 = bytes(np.asarray(Sha3_256().digest(msg))[0])
        keccak = bytes(np.asarray(Keccak256().digest(msg))[0])
        self.assertNotEqual(
            keccak,
            sha3,
            "Keccak-256 produced the SHA3-256 digest — the domain byte is wired "
            "to 0x06 rather than 0x01",
        )

    def test_they_differ_only_in_the_domain_byte(self) -> None:
        # The positive half of the same claim: every other parameter is shared,
        # so a divergence test alone could pass on two hashes that differ for the
        # wrong reason (a mistyped rate, say).
        self.assertEqual(KECCAK256_RATE, SHA3_256_RATE)
        self.assertNotEqual(KECCAK256_SUFFIX, SHA3_256_SUFFIX)

    def test_keccak_is_not_accidentally_hashlib_sha3(self) -> None:
        # Guards the direction the divergence test cannot: that `Keccak256` is
        # wrong in some *other* way that happens to differ from our own SHA3-256.
        msg = _message(64)
        self.assertNotEqual(
            bytes(np.asarray(Keccak256().digest(msg))[0]),
            hashlib.sha3_256(bytes(msg[0])).digest(),
        )


class Keccak256VectorTest(parameterized.TestCase):
    @parameterized.parameters(*_VECTORS)
    def test_matches_the_published_ethereum_vectors(
        self, msg: bytes, expected: str
    ) -> None:
        got = np.asarray(Keccak256().digest(_rows(msg)))[0]
        self.assertEqual(bytes(got).hex(), expected)

    @parameterized.parameters(*_LENGTHS)
    def test_host_sibling_agrees_across_absorb_boundaries(self, length: int) -> None:
        msg = _message(length)
        np.testing.assert_array_equal(
            np.asarray(Keccak256().digest(msg)),
            np.asarray(HostKeccak256().digest(msg)),
        )

    def test_batched_equals_per_row(self) -> None:
        rng = np.random.default_rng(41)
        batch = rng.integers(0, 256, size=(5, 200), dtype=np.uint8)
        got = np.asarray(Keccak256().digest(batch))
        host = HostKeccak256()
        for i in range(batch.shape[0]):
            with self.subTest(row=i):
                self.assertEqual(
                    bytes(got[i]), bytes(np.asarray(host.digest(batch[i : i + 1]))[0])
                )

    def test_digest_accepts_a_tracer(self) -> None:
        # The seam property a consumer hashing inside its own `@jit` depends on —
        # sig-frx's Ethereum ECDSA variant is the caller waiting on it.
        msg = _message(64)
        hasher = Keccak256()
        np.testing.assert_array_equal(
            np.asarray(hasher.digest(msg)),
            np.asarray(frx.jit(hasher.digest)(fnp.asarray(msg))),
        )


class SeamConformanceTest(absltest.TestCase):
    def test_both_implementations_satisfy_the_byte_hash_protocol(self) -> None:
        for h in (Keccak256(), HostKeccak256()):
            with self.subTest(hash=type(h).__name__):
                self.assertIsInstance(h, ByteHash)
                self.assertEqual(h.digest_size, 32)
                self.assertFalse(h.has_dedicated_fusion)

    def test_value_identity_keeps_the_seam_re_trace_safe(self) -> None:
        self.assertEqual(Keccak256(), Keccak256())
        self.assertEqual(hash(Keccak256()), hash(Keccak256()))
        # Same rate, same output, different standard: a consumer holding one in
        # pytree aux must not confuse them.
        self.assertNotEqual(Keccak256(), Sha3_256())
        self.assertNotEqual(Keccak256(), HostKeccak256())


if __name__ == "__main__":
    absltest.main()

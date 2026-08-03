# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Keccak-256 — the published vectors, and the padding byte pinned separately.

Separate from `fips202_test` because Keccak-256 is not a FIPS 202 function and
its golden cannot be `hashlib`: the standard library implements the *changed*
padding only. The vectors below are the published Ethereum ones, and
`HostKeccak256` recomputes them through an independent implementation.

**The vectors are the gate; the divergence assertions are diagnostics.** Wiring
the domain byte to SHA-3's `0x06` turns this into a correct SHA3-256, which fails
every vector here — `keccak256("")` becomes `a7ffc6f8…`. So the vectors do catch
the most-copied bug in Keccak implementations, and the claim that only a
divergence test can is false. What the divergence assertions add is a failure
that names the cause in one line instead of reporting that 32 bytes differ.

`test_they_differ_only_in_the_domain_byte` is the exception, and the reason this
module keeps a constants check at all: it fails on a *mistyped rate*, which would
also make the two hashes disagree, and which a divergence assertion would
therefore report as success.
"""

from __future__ import annotations

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

# Published Keccak-256 vectors — the values every EVM implementation agrees on.
_VECTORS = (
    (b"", "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
    (b"abc", "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
    (b"hello", "1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8"),
    (
        b"The quick brown fox jumps over the lazy dog",
        "4d741b6f1eb29cb2a9b9911c82f56fa8d73b04959d3d9d222895df6c0b28aa15",
    ),
)

# The 136-byte-rate subset of `fips202_test`'s boundaries: empty, tiny, one short
# of a block (the single-byte pad), exactly a block (a whole extra padding
# block), one past it, and multi-block. Its 167/168/169 are SHAKE128's rate and
# say nothing here.
_LENGTHS = (0, 1, 135, 136, 137, 300)


def _message(length: int) -> np.ndarray:
    return (np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)).reshape(1, length)


class Keccak256Test(parameterized.TestCase):
    @parameterized.parameters(*_VECTORS)
    def test_matches_the_published_vectors(self, msg: bytes, expected: str) -> None:
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = np.asarray(Keccak256().digest(rows))[0]
        self.assertEqual(bytes(got).hex(), expected)

    @parameterized.parameters(*_LENGTHS)
    def test_host_sibling_agrees_across_absorb_boundaries(self, length: int) -> None:
        msg = _message(length)
        np.testing.assert_array_equal(
            np.asarray(Keccak256().digest(msg)),
            np.asarray(HostKeccak256().digest(msg)),
        )

    def test_sha3_and_keccak_disagree(self) -> None:
        # Diagnostic: names the cause when the domain byte is wrong, where the
        # vector tests report only that the bytes differ. One length is enough —
        # the absorb boundaries are swept above, and the padding rule is shared.
        msg = _message(137)
        self.assertNotEqual(
            bytes(np.asarray(Keccak256().digest(msg))[0]),
            bytes(np.asarray(Sha3_256().digest(msg))[0]),
            "Keccak-256 produced the SHA3-256 digest — the domain byte is wired "
            "to 0x06 rather than 0x01",
        )

    def test_they_differ_only_in_the_domain_byte(self) -> None:
        # The one check the vectors cannot make: they fail identically whether
        # the suffix or the rate is wrong, and a divergence assertion passes on a
        # mistyped rate. This fails at import-time constants instead.
        self.assertEqual(KECCAK256_RATE, SHA3_256_RATE)
        self.assertNotEqual(KECCAK256_SUFFIX, SHA3_256_SUFFIX)

    def test_digest_accepts_a_tracer(self) -> None:
        # The seam property a consumer hashing inside its own `@jit` depends on.
        # Sub-rate on purpose: `_permute_body` inlines 24 unrolled rounds per
        # absorb block, so compile time scales with block count while the
        # property under test does not.
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

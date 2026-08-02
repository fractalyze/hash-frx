# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHA3-256, SHAKE128 and SHAKE256 — byte-match against `hashlib`.

Agnostic golden, the same one `sha256_test` uses: `hashlib` implements FIPS 202
without sharing a line with this tree, so agreement is agreement with the
standard rather than with a second copy of one reading of it. The Keccak-f[1600]
oracle under `reference_test` is anchored the same way and for the same reason.

The lengths and output sizes are chosen for the boundaries the sponge can get
wrong rather than for coverage: a message that ends exactly on a rate boundary
(so padding takes a whole extra block), one that ends one byte short of it (so
the domain suffix and the `10*1` closing bit land on the same byte), and outputs
that cross a rate boundary (so the squeeze has to permute again).
"""

from __future__ import annotations

import hashlib

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.byte_hash import ByteHash
from hash_frx.keccak.fips202 import (
    SHA3_256_RATE,
    SHAKE128_RATE,
    SHAKE256_RATE,
    HostSha3_256,
    HostShake128,
    HostShake256,
    Sha3_256,
    Shake128,
    Shake256,
)
from hash_frx.keccak.sponge import KeccakSponge

# Absorb boundaries for a 136-byte rate: empty, tiny, one short of a block (the
# single-byte pad), exactly a block (a whole extra padding block), one past it,
# and multi-block. 167/168 do the same for SHAKE128's 168-byte rate.
_LENGTHS = (0, 1, 135, 136, 137, 167, 168, 169, 300)

# Output sizes for the XOFs: under a rate, exactly a rate, and over it (which
# forces a second squeeze block), plus a length that is not a lane multiple.
_SHAKE_OUTPUTS = (1, 32, 131, 136, 137, 168, 200, 400)


def _message(length: int) -> np.ndarray:
    return (np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)).reshape(1, length)


class Sha3_256Test(parameterized.TestCase):
    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        msg = _message(length)
        got = bytes(np.asarray(Sha3_256().digest(msg))[0])
        self.assertEqual(got, hashlib.sha3_256(bytes(msg[0])).digest())

    def test_batched_equals_per_row(self) -> None:
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(5, 200), dtype=np.uint8)
        got = np.asarray(Sha3_256().digest(batch))
        for i in range(batch.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.sha3_256(bytes(batch[i])).digest())

    def test_host_sibling_agrees_with_the_device_one(self) -> None:
        msg = _message(300)
        np.testing.assert_array_equal(
            np.asarray(Sha3_256().digest(msg)), HostSha3_256().digest(msg)
        )


class ShakeTest(parameterized.TestCase):
    @parameterized.parameters(*_SHAKE_OUTPUTS)
    def test_shake128_matches_hashlib(self, out: int) -> None:
        msg = _message(200)
        got = bytes(np.asarray(Shake128(out).digest(msg))[0])
        self.assertEqual(got, hashlib.shake_128(bytes(msg[0])).digest(out))

    @parameterized.parameters(*_SHAKE_OUTPUTS)
    def test_shake256_matches_hashlib(self, out: int) -> None:
        msg = _message(200)
        got = bytes(np.asarray(Shake256(out).digest(msg))[0])
        self.assertEqual(got, hashlib.shake_256(bytes(msg[0])).digest(out))

    @parameterized.parameters(*_LENGTHS)
    def test_shake256_matches_hashlib_across_absorb_boundaries(
        self, length: int
    ) -> None:
        msg = _message(length)
        got = bytes(np.asarray(Shake256(64).digest(msg))[0])
        self.assertEqual(got, hashlib.shake_256(bytes(msg[0])).digest(64))

    def test_the_two_shakes_differ_at_the_same_output_length(self) -> None:
        # Same suffix and output, different rate — so a rate mix-up cannot hide.
        msg = _message(200)
        self.assertNotEqual(
            bytes(np.asarray(Shake128(32).digest(msg))[0]),
            bytes(np.asarray(Shake256(32).digest(msg))[0]),
        )

    def test_host_siblings_agree_with_the_device_ones(self) -> None:
        msg = _message(300)
        np.testing.assert_array_equal(
            np.asarray(Shake128(200).digest(msg)), HostShake128(200).digest(msg)
        )
        np.testing.assert_array_equal(
            np.asarray(Shake256(200).digest(msg)), HostShake256(200).digest(msg)
        )


class TracedDigestTest(absltest.TestCase):
    """The property the seam actually promises for a device hash: a traced message.

    This is what a scheme reaching the hash through `ByteHash` needs in order to
    hash inside its own `@jit` — `fractalyze/sig-frx#15` is the caller waiting on
    it — and nothing about matching `hashlib` eagerly implies it.
    """

    def test_digest_accepts_a_tracer(self) -> None:
        msg = _message(200)
        eager = np.asarray(Shake256(64).digest(msg))
        traced = np.asarray(frx.jit(Shake256(64).digest)(fnp.asarray(msg)))
        np.testing.assert_array_equal(eager, traced)
        self.assertEqual(bytes(traced[0]), hashlib.shake_256(bytes(msg[0])).digest(64))

    def test_sha3_digest_accepts_a_tracer(self) -> None:
        msg = _message(137)
        traced = np.asarray(frx.jit(Sha3_256().digest)(fnp.asarray(msg)))
        self.assertEqual(bytes(traced[0]), hashlib.sha3_256(bytes(msg[0])).digest())


class SeamConformanceTest(absltest.TestCase):
    def test_every_implementation_satisfies_the_byte_hash_protocol(self) -> None:
        for h in (
            Sha3_256(),
            Shake128(32),
            Shake256(32),
            HostSha3_256(),
            HostShake128(32),
            HostShake256(32),
        ):
            with self.subTest(hash=type(h).__name__):
                self.assertIsInstance(h, ByteHash)
                self.assertIsInstance(h.has_dedicated_fusion, bool)

    def test_digest_size_matches_what_digest_returns(self) -> None:
        msg = _message(64)
        for h in (Sha3_256(), Shake128(48), Shake256(48), HostShake256(48)):
            with self.subTest(hash=type(h).__name__):
                out = np.asarray(h.digest(msg))
                self.assertEqual(out.shape, (1, h.digest_size))

    def test_value_identity_keeps_the_seam_re_trace_safe(self) -> None:
        # Param-free compares by type; an XOF's output length is part of its
        # value, so two lengths are two hashes rather than one asked twice.
        self.assertEqual(Sha3_256(), Sha3_256())
        self.assertEqual(hash(Sha3_256()), hash(Sha3_256()))
        self.assertEqual(Shake256(32), Shake256(32))
        self.assertEqual(hash(Shake256(32)), hash(Shake256(32)))
        self.assertNotEqual(Shake256(32), Shake256(64))
        # Same rate and output, different function: distinct types must not
        # compare equal, or a consumer's pytree aux would confuse them.
        self.assertNotEqual(Shake256(32), Shake128(32))
        self.assertNotEqual(Shake256(32), HostShake256(32))


class KeccakSpongeTest(absltest.TestCase):
    def test_rejects_a_rate_off_the_element_boundary(self) -> None:
        with self.assertRaises(ValueError):
            KeccakSponge(rate=137, suffix=0x06, output_size=32)

    def test_rejects_a_rate_leaving_no_capacity(self) -> None:
        with self.assertRaises(ValueError):
            KeccakSponge(rate=200, suffix=0x06, output_size=32)

    def test_rejects_a_non_byte_suffix(self) -> None:
        with self.assertRaises(ValueError):
            KeccakSponge(rate=136, suffix=0x100, output_size=32)

    def test_rejects_an_empty_output(self) -> None:
        with self.assertRaises(ValueError):
            KeccakSponge(rate=136, suffix=0x06, output_size=0)

    def test_rejects_a_message_that_is_not_a_batch(self) -> None:
        sponge = KeccakSponge(rate=136, suffix=0x06, output_size=32)
        with self.assertRaises(ValueError):
            sponge.hash(np.zeros(64, dtype=np.uint8))

    def test_value_equality_over_the_three_parameters(self) -> None:
        a = KeccakSponge(rate=136, suffix=0x06, output_size=32)
        self.assertEqual(a, KeccakSponge(rate=136, suffix=0x06, output_size=32))
        self.assertEqual(hash(a), hash(KeccakSponge(136, 0x06, 32)))
        self.assertNotEqual(a, KeccakSponge(rate=136, suffix=0x1F, output_size=32))

    def test_the_published_rates_are_the_standard_capacities(self) -> None:
        # FIPS 202 section 6: rate = (1600 - capacity) / 8 bytes. Stated as an
        # assertion rather than a comment so a mistyped constant fails here.
        self.assertEqual(SHA3_256_RATE, (1600 - 512) // 8)
        self.assertEqual(SHAKE128_RATE, (1600 - 256) // 8)
        self.assertEqual(SHAKE256_RATE, (1600 - 512) // 8)


if __name__ == "__main__":
    absltest.main()

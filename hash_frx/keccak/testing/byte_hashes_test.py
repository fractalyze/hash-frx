# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHA3-256, SHA3-512, SHAKE128 and SHAKE256 — byte-match against `hashlib`.

Agnostic golden, the same one `sha256_test` uses: `hashlib` implements FIPS 202
without sharing a line with this tree, so agreement is agreement with the
standard rather than with a second copy of one reading of it. The Keccak-f[1600]
oracle under `reference_test` is anchored the same way and for the same reason.

The four functions ride one case table so that every sweep runs on all of them.
Written as a test per function instead, a sweep ends up covering whichever one it
was written against — the lengths straddling 168 were added for SHAKE128's rate
and would otherwise never have run on SHAKE128.

The lengths and output sizes are chosen for the boundaries the sponge can get
wrong rather than for coverage: a message that ends exactly on a rate boundary
(so padding takes a whole extra block), one that ends one byte short of it (so
the domain suffix and the `10*1` closing bit land on the same byte), and outputs
that cross a rate boundary (so the squeeze has to permute again).

Every rate in the family needs its own three lengths, because a boundary is a
property of the rate and not of the function. SHA3-512's 72 is the narrowest,
and the only rate under which the 136- and 168-byte lengths land mid-block.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.byte_hash import ByteHash
from hash_frx.keccak.byte_hashes import (
    SHA3_256_RATE,
    SHA3_512_RATE,
    SHAKE128_RATE,
    SHAKE256_RATE,
    HostSha3_256,
    HostSha3_512,
    HostShake128,
    HostShake256,
    Keccak256,
    Sha3_256,
    Sha3_512,
    Shake128,
    Shake256,
)
from hash_frx.keccak.sponge import KeccakSponge

# Absorb boundaries for a 136-byte rate: empty, tiny, one short of a block (the
# single-byte pad), exactly a block (a whole extra padding block), one past it,
# and multi-block. 167/168/169 do the same for SHAKE128's 168-byte rate, and
# 71/72/73 for SHA3-512's 72-byte one.
_LENGTHS = (0, 1, 71, 72, 73, 135, 136, 137, 167, 168, 169, 300)

# Output sizes for the XOFs: under a rate, exactly a rate, and over it (which
# forces a second squeeze block), plus a length that is not a lane multiple.
_SHAKE_OUTPUTS = (1, 32, 131, 136, 137, 168, 200, 400)

# One row per FIPS 202 function: (name, device factory, host factory, hashlib
# reference). Both factories take the output length, so a fixed-output function
# ignores it and every sweep can drive all four the same way.
_SHA3_CASES = (
    (
        "sha3_256",
        lambda _out: Sha3_256(),
        lambda _out: HostSha3_256(),
        lambda msg, _out: hashlib.sha3_256(msg).digest(),
    ),
    (
        "sha3_512",
        lambda _out: Sha3_512(),
        lambda _out: HostSha3_512(),
        lambda msg, _out: hashlib.sha3_512(msg).digest(),
    ),
)
# The XOFs alone, for the sweep that varies output length — SHA-3's is fixed.
_XOF_CASES = (
    (
        "shake128",
        Shake128,
        HostShake128,
        lambda msg, out: hashlib.shake_128(msg).digest(out),
    ),
    (
        "shake256",
        Shake256,
        HostShake256,
        lambda msg, out: hashlib.shake_256(msg).digest(out),
    ),
)
# Split by whether the output length is the caller's rather than sliced out of
# one table: a slice silently picks the wrong rows the moment one is inserted.
_CASES = _SHA3_CASES + _XOF_CASES

_Case = tuple[str, Callable[[int], ByteHash], Callable[[int], ByteHash], Callable]


def _message(length: int) -> np.ndarray:
    return (np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)).reshape(1, length)


class Fips202Test(parameterized.TestCase):
    @parameterized.product(case=_CASES, length=_LENGTHS)
    def test_matches_hashlib_across_absorb_boundaries(
        self, case: _Case, length: int
    ) -> None:
        name, device, _host, reference = case
        msg = _message(length)
        with self.subTest(hash=name):
            got = bytes(np.asarray(device(64).digest(msg))[0])
            self.assertEqual(got, reference(bytes(msg[0]), 64))

    @parameterized.product(case=_XOF_CASES, out=_SHAKE_OUTPUTS)
    def test_output_length_matches_hashlib(self, case: _Case, out: int) -> None:
        name, device, _host, reference = case
        msg = _message(200)
        with self.subTest(hash=name):
            got = bytes(np.asarray(device(out).digest(msg))[0])
            self.assertEqual(got, reference(bytes(msg[0]), out))

    @parameterized.parameters(*_CASES)
    def test_host_sibling_agrees_with_the_device_one(
        self,
        name: str,
        device: Callable[[int], ByteHash],
        host: Callable[[int], ByteHash],
        _reference: Callable,
    ) -> None:
        msg = _message(300)
        np.testing.assert_array_equal(
            np.asarray(device(200).digest(msg)), np.asarray(host(200).digest(msg))
        )

    def test_batched_equals_per_row(self) -> None:
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(5, 200), dtype=np.uint8)
        got = np.asarray(Sha3_256().digest(batch))
        for i in range(batch.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.sha3_256(bytes(batch[i])).digest())

    def test_the_two_shakes_differ_at_the_same_output_length(self) -> None:
        # Same suffix and output, different rate — so a rate mix-up cannot hide.
        msg = _message(200)
        self.assertNotEqual(
            bytes(np.asarray(Shake128(32).digest(msg))[0]),
            bytes(np.asarray(Shake256(32).digest(msg))[0]),
        )

    def test_the_two_sha3s_differ_where_they_overlap(self) -> None:
        # The same pair for SHA-3, where the suffix is shared and the rate is the
        # only difference: truncating SHA3-512 to 32 bytes must not give
        # SHA3-256. Compared on the prefix because the digests differ in length,
        # which is what makes a rate mix-up otherwise invisible here.
        msg = _message(200)
        self.assertNotEqual(
            bytes(np.asarray(Sha3_256().digest(msg))[0]),
            bytes(np.asarray(Sha3_512().digest(msg))[0][:32]),
        )


class TracedDigestTest(parameterized.TestCase):
    """The property the seam actually promises for a device hash: a traced message.

    This is what a scheme reaching the hash through `ByteHash` needs in order to
    hash inside its own `@jit` — `fractalyze/sig-frx#15` is the caller waiting on
    it — and nothing about matching `hashlib` eagerly implies it.

    Sub-rate messages on purpose: `_permute_body` inlines 24 unrolled rounds per
    absorb block, so compile time scales with block count while the property under
    test does not. Block boundaries are swept eagerly and cheaply above.
    """

    @parameterized.parameters(*_CASES)
    def test_digest_accepts_a_tracer(
        self,
        name: str,
        device: Callable[[int], ByteHash],
        _host: Callable[[int], ByteHash],
        reference: Callable,
    ) -> None:
        msg = _message(64)
        hasher = device(32)
        eager = np.asarray(hasher.digest(msg))
        traced = np.asarray(frx.jit(hasher.digest)(fnp.asarray(msg)))
        np.testing.assert_array_equal(eager, traced)
        self.assertEqual(bytes(traced[0]), reference(bytes(msg[0]), 32))


class SeamConformanceTest(absltest.TestCase):
    def test_every_implementation_satisfies_the_byte_hash_protocol(self) -> None:
        for h in (
            Sha3_256(),
            Sha3_512(),
            Shake128(32),
            Shake256(32),
            Keccak256(),
            HostSha3_256(),
            HostSha3_512(),
            HostShake128(32),
            HostShake256(32),
        ):
            with self.subTest(hash=type(h).__name__):
                self.assertIsInstance(h, ByteHash)
                # Pinned rather than merely type-checked: `False` on the device
                # implementations too is the substantive claim this module's
                # docstring makes, and only an assertion keeps it honest.
                self.assertFalse(h.has_dedicated_fusion)

    def test_digest_size_matches_what_digest_returns(self) -> None:
        msg = _message(64)
        for h in (
            Sha3_256(),
            Sha3_512(),
            Shake128(48),
            Shake256(48),
            HostSha3_512(),
            HostShake256(48),
        ):
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
        # The two SHA-3 rows share a suffix and differ only in rate and length,
        # so they are the pair most likely to be conflated by a `__eq__` written
        # over `digest_size` without the type.
        self.assertEqual(Sha3_512(), Sha3_512())
        self.assertEqual(hash(Sha3_512()), hash(Sha3_512()))
        self.assertNotEqual(Sha3_512(), Sha3_256())
        self.assertNotEqual(Sha3_512(), Shake256(64))


class KeccakSpongeTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("rate_off_the_element_boundary", {"rate": 137, "suffix": 0x06, "out": 32}),
        ("rate_leaving_no_capacity", {"rate": 200, "suffix": 0x06, "out": 32}),
        ("non_byte_suffix", {"rate": 136, "suffix": 0x100, "out": 32}),
        ("empty_output", {"rate": 136, "suffix": 0x06, "out": 0}),
    )
    def test_rejects_an_impossible_parameterization(self, kwargs: dict) -> None:
        with self.assertRaises(ValueError):
            KeccakSponge(
                rate=kwargs["rate"], suffix=kwargs["suffix"], output_size=kwargs["out"]
            )

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
        self.assertEqual(SHA3_512_RATE, (1600 - 1024) // 8)
        self.assertEqual(SHAKE128_RATE, (1600 - 256) // 8)
        self.assertEqual(SHAKE256_RATE, (1600 - 512) // 8)


if __name__ == "__main__":
    absltest.main()

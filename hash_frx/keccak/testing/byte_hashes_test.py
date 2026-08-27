# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The four SHA-3 rows and both SHAKEs — byte-match against `hashlib`, and the
published CAVP vectors under them.

Agnostic golden, the same one `sha256_test` uses: `hashlib` implements FIPS 202
without sharing a line with this tree, so agreement is agreement with the
standard rather than with a second copy of one reading of it. The Keccak-f[1600]
oracle under `reference_test` is anchored the same way and for the same reason.

The six functions ride one case table so that every sweep runs on all of them,
and the seam sweeps take their rows from it too. Written as a test per function
instead, a sweep ends up covering whichever one it was written against — the
lengths straddling 168 were added for SHAKE128's rate and would otherwise never
have run on SHAKE128.

The lengths and output sizes are chosen for the boundaries the sponge can get
wrong rather than for coverage: a message that ends exactly on a rate boundary
(so padding takes a whole extra block), one that ends one byte short of it (so
the domain suffix and the `10*1` closing bit land on the same byte), and outputs
that cross a rate boundary (so the squeeze has to permute again).

Every rate in the family needs its own three lengths, because a boundary is a
property of the rate and not of the function — so `_LENGTHS` is derived from
`_RATES` rather than listed, and a row arrives with its boundaries or not at
all. SHA3-512's 72 is the narrowest; the four rates between it and SHAKE128's
168 each own three lengths that land mid-block under every other one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.keccak import permutation as permutation_mod
from hash_frx.keccak.byte_hashes import (
    SHA3_224_RATE,
    SHA3_256_RATE,
    SHA3_384_RATE,
    SHA3_512_RATE,
    SHAKE128_RATE,
    SHAKE256_RATE,
    Keccak256,
    Sha3_224,
    Sha3_256,
    Sha3_384,
    Sha3_512,
    Shake128,
    Shake256,
)
from hash_frx.keccak.sponge import KeccakSponge

# Every rate in the family, so the boundaries below can be DERIVED from them
# rather than listed. The module docstring states the rule — a boundary is a
# property of the rate, so every rate needs its own three lengths — and a
# hand-kept list leaves that rule to whoever adds the next row: SHA3-224's 144
# and SHA3-384's 104 are shared with no other row, so a row arriving without
# its three lengths is a rate nothing lands on.
_RATES = (
    SHA3_224_RATE,
    SHA3_256_RATE,
    SHA3_384_RATE,
    SHA3_512_RATE,
    SHAKE128_RATE,
    SHAKE256_RATE,
)

# One short of a block (the domain suffix and the closing bit share a byte),
# exactly a block (the padding takes a whole extra one), and one past it — for
# each rate — plus the three that belong to no rate: empty, tiny, multi-block.
_LENGTHS = tuple(sorted({0, 1, 300} | {r + d for r in _RATES for d in (-1, 0, 1)}))

# Output sizes for the XOFs: under a rate, exactly a rate, and over it (which
# forces a second squeeze block), plus a length that is not a lane multiple.
_SHAKE_OUTPUTS = (1, 32, 131, 136, 137, 168, 200, 400)

# One row per FIPS 202 function: (name, device factory, host factory, hashlib
# reference). Both factories take the output length, so a fixed-output function
# ignores it and every sweep can drive all four the same way.
_SHA3_CASES = (
    (
        "sha3_224",
        lambda _out: Sha3_224(),
        lambda msg, _out: hashlib.sha3_224(msg).digest(),
    ),
    (
        "sha3_256",
        lambda _out: Sha3_256(),
        lambda msg, _out: hashlib.sha3_256(msg).digest(),
    ),
    (
        "sha3_384",
        lambda _out: Sha3_384(),
        lambda msg, _out: hashlib.sha3_384(msg).digest(),
    ),
    (
        "sha3_512",
        lambda _out: Sha3_512(),
        lambda msg, _out: hashlib.sha3_512(msg).digest(),
    ),
)
# The XOFs alone, for the sweep that varies output length — SHA-3's is fixed.
_XOF_CASES = (
    (
        "shake128",
        Shake128,
        lambda msg, out: hashlib.shake_128(msg).digest(out),
    ),
    (
        "shake256",
        Shake256,
        lambda msg, out: hashlib.shake_256(msg).digest(out),
    ),
)
# Split by whether the output length is the caller's rather than sliced out of
# one table: a slice silently picks the wrong rows the moment one is inserted.
_CASES = _SHA3_CASES + _XOF_CASES

_Case = tuple[str, Callable[[int], ByteHash], Callable]


def _rows(output_size: int) -> list[ByteHash]:
    """Every row this module ships, at `output_size` where the row takes one.

    Derived from `_CASES` for the reason that table exists: a hand-written list
    covers whichever rows it was written against, and these three sweeps had
    fallen a family behind it once already.
    """
    return [device(output_size) for _name, device, _ref in _CASES] + [Keccak256()]


def _message(length: int) -> np.ndarray:
    return (np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)).reshape(1, length)


class Fips202Test(parameterized.TestCase):
    @parameterized.product(case=_CASES, length=_LENGTHS)
    def test_matches_hashlib_across_absorb_boundaries(
        self, case: _Case, length: int
    ) -> None:
        name, device, reference = case
        msg = _message(length)
        with self.subTest(hash=name):
            got = bytes(np.asarray(device(64).digest(msg))[0])
            self.assertEqual(got, reference(bytes(msg[0]), 64))

    @parameterized.product(case=_XOF_CASES, out=_SHAKE_OUTPUTS)
    def test_output_length_matches_hashlib(self, case: _Case, out: int) -> None:
        name, device, reference = case
        msg = _message(200)
        with self.subTest(hash=name):
            got = bytes(np.asarray(device(out).digest(msg))[0])
            self.assertEqual(got, reference(bytes(msg[0]), out))

    def test_batched_equals_per_row(self) -> None:
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(5, 200), dtype=np.uint8)
        got = np.asarray(Sha3_256().digest(batch))
        for i in range(batch.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.sha3_256(bytes(batch[i])).digest())

    @parameterized.product(case=_CASES, outer=(1, 3))
    def test_vmapped_equals_hashlib_per_row(self, case: _Case, outer: int) -> None:
        # `digest` documents `[B, L]`, and a consumer that already batches
        # reaches it under its own `frx.vmap` rather than by widening `B` —
        # sig-frx's ML-DSA verification is `frx.vmap(_verify_one)` over the whole
        # path. Inside the vmap the logical shape is still `[B, L]`, so the
        # `ndim != 2` guard cannot see the outer axis and the marked region is
        # emitted one rank deeper than its ABI once admitted.
        #
        # `outer=1` is not redundant with 3: a leading axis of one is the case a
        # fix that squeezes rather than collapses would pass.
        name, device, reference = case
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(outer, 5, 200), dtype=np.uint8)
        with self.subTest(hash=name):
            # 200 crosses every rate in the family, so the XOFs squeeze twice
            # here; the fixed-output rows ignore it and keep their own size.
            hash_ = device(200)
            got = np.asarray(frx.vmap(hash_.digest)(batch))
            self.assertEqual(got.shape, (outer, 5, hash_.digest_size))
            for v in range(outer):
                for i in range(batch.shape[1]):
                    self.assertEqual(
                        bytes(got[v, i]),
                        reference(bytes(batch[v, i]), hash_.digest_size),
                    )

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
        for h in _rows(32):
            with self.subTest(hash=type(h).__name__):
                self.assertIsInstance(h, ByteHash)

    def test_the_rows_report_a_dedicated_fusion(self) -> None:
        # Pinned rather than left to the docstring: a row lowers the whole
        # padded absorb and squeeze to one `hash_frx.digest.keccak_sponge`
        # kernel wherever that emitter can be reached.
        #
        # The device side is compared against the shipped condition rather than
        # against True so the case follows the pin and the backend rather than
        # restating them — it was written when the answer was False on the CPU
        # leg. What that costs is the case's old ability to
        # catch a pin that dropped below the `frx>=` floor — right bytes, no
        # kernel, nothing else in the suite noticing. That half now lives in
        # `permutation_test.EmitterGateTest`, which asserts the pin as its
        # premise and then holds the backend to it.
        expected = permutation_mod._routes_to_dedicated_emitter()
        for device in _rows(32):
            with self.subTest(device=type(device).__name__):
                self.assertIs(
                    device.fusion_path,
                    FusionPath.DEDICATED if expected else FusionPath.GENERIC,
                )

    def test_digest_size_matches_what_digest_returns(self) -> None:
        msg = _message(64)
        for h in _rows(48):
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
        self.assertEqual(SHA3_224_RATE, (1600 - 448) // 8)
        self.assertEqual(SHA3_256_RATE, (1600 - 512) // 8)
        self.assertEqual(SHA3_384_RATE, (1600 - 768) // 8)
        self.assertEqual(SHA3_512_RATE, (1600 - 1024) // 8)
        self.assertEqual(SHAKE128_RATE, (1600 - 256) // 8)
        self.assertEqual(SHAKE256_RATE, (1600 - 512) // 8)


class EmptyBatchTest(absltest.TestCase):
    """A zero-row batch is a valid batch (#211): every row returns
    uint8 [0, digest_size] instead of failing in a block-count reshape."""

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        rows: list[tuple[ByteHash, int]] = [
            (Sha3_256(), 32),
            (Shake128(32), 32),
            (Keccak256(), 32),
        ]
        for hasher, size in rows:
            got = np.asarray(hasher.digest(fnp.zeros((0, 64), dtype=fnp.uint8)))
            self.assertEqual(got.shape, (0, size))
            self.assertEqual(got.dtype, np.uint8)


class XofSizeTest(absltest.TestCase):
    """An XOF refuses a zero-length output (#215)."""

    def test_zero_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_size"):
            Shake128(0)


# ---------------------------------------------------------------------------
# NIST CAVP SHA-3 byte-test vectors for the two rows this file gained, from
# csrc.nist.gov's `sha-3bytetestvectors.zip` (`SHA3_224ShortMsg.rsp`,
# `SHA3_224LongMsg.rsp` and the SHA3-384 pair). Extracted from those files,
# never typed.
#
# The `hashlib` sweeps above already cover every length these do and more, so
# what these add is independence from `hashlib` itself: the sweeps say this
# tree agrees with CPython's FIPS 202, and these say both agree with NIST.
# While extracting, all 450 rows of the four files were checked against
# `hashlib` — 245 for SHA3-224 and 205 for SHA3-384 — so the five kept per
# function are a readable subset of a full agreement rather than the extent of
# what was verified.
#
# The lengths are chosen where a rate can be got wrong, and CAVP's ShortMsg
# files happen to stop exactly at each function's rate — 144 bytes for SHA3-224
# and 104 for SHA3-384 — so the "message ends on a block boundary, padding
# takes a whole further one" case has published evidence here rather than only
# a differential. One past the rate does not, and stays the sweeps'.
_CAVP_VECTORS: tuple[tuple[str, type[ByteHash], str, str], ...] = (
    # empty.
    (
        "sha3_224_0b",
        Sha3_224,
        "",
        "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7",
    ),
    # one byte.
    (
        "sha3_224_1b",
        Sha3_224,
        "01",
        "488286d9d32716e5881ea1ee51f36d3660d70f0db03b3f612ce9eda4",
    ),
    # one short of the rate.
    (
        "sha3_224_143b",
        Sha3_224,
        "0eef947f1e4f01cdb5481ca6eaa25f2caca4c401612888fecef52e283748c8df"
        "c7b47259322c1f4f985f98f6ad44c13117f51e0517c0974d6c7b78af7419bcce"
        "957b8bc1db8801c5e280312ef78d6aa47a9cb98b866aaec3d5e26392dda6bbde"
        "3fece8a0628b30955b55f03711a8e1eb9e409a7cf84f56c8d0d0f8b9ba184c77"
        "8fae90dc0f5c3329cb86dcf743bbae",
        "98ec52c21cb988b1434b1653dd4ac806d118de6af1bb471c16577c34",
    ),
    # exactly the rate.
    (
        "sha3_224_144b",
        Sha3_224,
        "e65de91fdcb7606f14dbcfc94c9c94a57240a6b2c31ed410346c4dc011526559"
        "e44296fc988cc589de2dc713d0e82492d4991bd8c4c5e6c74c753fc09345225e"
        "1db8d565f0ce26f5f5d9f404a28cf00bd655a5fe04edb682942d675b86235f23"
        "5965ad422ba5081a21865b8209ae81763e1c4c0cccbccdaad539cf773413a50f"
        "5ff1267b9238f5602adc06764f775d3c",
        "26ec9df54d9afe11710772bfbeccc83d9d0439d3530777c81b8ae6a3",
    ),
    # multi-block.
    (
        "sha3_224_289b",
        Sha3_224,
        "31c82d71785b7ca6b651cb6c8c9ad5e2aceb0b0633c088d33aa247ada7a594ff"
        "4936c023251319820a9b19fc6c48de8a6f7ada214176ccdaadaeef51ed43714a"
        "c0c8269bbd497e46e78bb5e58196494b2471b1680e2d4c6dbd249831bd83a4d3"
        "be06c8a2e903933974aa05ee748bfe6ef359f7a143edf0d4918da916bd6f15e2"
        "6a790cff514b40a5da7f72e1ed2fe63a05b8149587bea05653718cc8980eadbf"
        "eca85b7c9c286dd040936585938be7f98219700c83a9443c2856a80ff46852b2"
        "6d1b1edf72a30203cf6c44a10fa6eaf1920173cedfb5c4cf3ac665b37a86ed02"
        "155bbbf17dc2e786af9478fe0889d86c5bfa85a242eb0854b1482b7bd16f67f8"
        "0bef9c7a628f05a107936a64273a97b0088b0e515451f916b5656230a12ba6dc"
        "78",
        "aab23c9e7fb9d7dacefdfd0b1ae85ab1374abff7c4e3f7556ecae412",
    ),
    # empty.
    (
        "sha3_384_0b",
        Sha3_384,
        "",
        "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2a"
        "c3713831264adb47fb6bd1e058d5f004",
    ),
    # one byte.
    (
        "sha3_384_1b",
        Sha3_384,
        "80",
        "7541384852e10ff10d5fb6a7213a4a6c15ccc86d8bc1068ac04f69277142944f"
        "4ee50d91fdc56553db06b2f5039c8ab7",
    ),
    # one short of the rate.
    (
        "sha3_384_103b",
        Sha3_384,
        "6c36147652e71b560becbca1e7656c81b4f70bece26321d5e55e67a3db9d89e2"
        "6f2f2a38fd0f289bf7fa22c2877e38d9755412794cef24d7b855303c332e0cb5"
        "e01aa50bb74844f5e345108d6811d5010978038b699ffaa370de8473f0cda38b"
        "89a28ed6cabaf6",
        "b1319192df11faa00d3c4b068becc8f1ba3b00e0d1ff1f93c11a3663522fdb92"
        "ab3cca389634687c632e0a4b5a26ce92",
    ),
    # exactly the rate.
    (
        "sha3_384_104b",
        Sha3_384,
        "92c41d34bd249c182ad4e18e3b856770766f1757209675020d4c1cf7b6f7686c"
        "8c1472678c7c412514e63eb9f5aee9f5c9d5cb8d8748ab7a5465059d9cbbb8a5"
        "6211ff32d4aaa23a23c86ead916fe254cc6b2bff7a9553df1551b531f95bb41c"
        "bbc4acddbd372921",
        "71307eec1355f73e5b726ed9efa1129086af81364e30a291f684dfade693cc4b"
        "c3d6ffcb7f3b4012a21976ff9edcab61",
    ),
    # multi-block.
    (
        "sha3_384_209b",
        Sha3_384,
        "5fe35923b4e0af7dd24971812a58425519850a506dfa9b0d254795be785786c3"
        "19a2567cbaa5e35bcf8fe83d943e23fa5169b73adc1fcf8b607084b15e6a013d"
        "f147e46256e4e803ab75c110f77848136be7d806e8b2f868c16c3a90c1446340"
        "7038cb7d9285079ef162c6a45cedf9c9f066375c969b5fcbcda37f02aacff4f3"
        "1cded3767570885426bebd9eca877e44674e9ae2f0c24cdd0e7e1aaf1ff2fe7f"
        "80a1c4f5078eb34cd4f06fa94a2d1eab5806ca43fd0f06c60b63d5402b95c70c"
        "21ea65a151c5cfaf8262a46be3c722264b",
        "3054d249f916a6039b2a9c3ebec1418791a0608a170e6d36486035e5f92635ea"
        "ba98072a85373cb54e2ae3f982ce132b",
    ),
)


class Sha3CavpVectorTest(parameterized.TestCase):
    """The published vectors, under the `hashlib` sweeps rather than instead of
    them."""

    @parameterized.named_parameters(*_CAVP_VECTORS)
    def test_matches_the_cavp_vector(
        self, row: type[ByteHash], msg_hex: str, md_hex: str
    ) -> None:
        msg = np.frombuffer(bytes.fromhex(msg_hex), dtype=np.uint8)[None, :]
        got = np.asarray(row().digest(msg))[0]
        self.assertEqual(bytes(got).hex(), md_hex)


if __name__ == "__main__":
    absltest.main()

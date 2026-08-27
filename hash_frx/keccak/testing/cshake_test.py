# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""cSHAKE128 / cSHAKE256 against SP 800-185 section 3.

Three layers of anchor, because no single one covers this construction:

**NIST's published sample vectors**, all four, at both rates and both output
lengths. `hashlib` has no cSHAKE, so unlike the FIPS 202 rows there is no
independent implementation of the *whole* function to diff against — the
published values are what pin it.

**`hashlib`'s SHAKE for the fallback.** Section 3.3's empty-`N`-and-`S` case is
plain SHAKE, and that half of the construction does have an agnostic golden, so
it gets one across every absorb boundary.

**An in-tree reference for everything between.** `reference.sponge` is a
plain-Python FIPS 202 sponge that shares no line with the device path, driven
here through the same encodings. That is a weaker claim than the FIPS 202 rows
enjoy — the encodings under it are this tree's — and it is made honestly: the
encodings carry their own published anchor in `encodings_test.py`, against
NIST's KMAC intermediates, so what this layer adds is that the device sponge
agrees with a separate sponge over the same prefix, at lengths no published
vector reaches.

**The fallback is asserted to be a branch.** Section 3.3 makes empty `N` and `S`
*be* SHAKE rather than be cSHAKE over empty strings, and those are different
hashes — so the test asserts both that the fallback matches SHAKE and that the
path not taken would have disagreed. Without the second half, an implementation
that never branched would pass the first by accident only if it were also wrong,
which is precisely the confusion worth ruling out.

**Sweeps are unions, not cross products.** Absorb and squeeze are sequential in
a sponge, so a message length and an output length have no interaction for a
cross product to find — and every distinct `(rate, output_size, block count)`
re-traces the whole fused composite at roughly 1.5 s a time. So lengths are
swept at one output size and output sizes at one length, which is
`byte_hashes_test`'s structure and for this reason.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.keccak.byte_hashes import (
    SHAKE128_RATE,
    SHAKE256_RATE,
    SHAKE_SUFFIX,
    Shake128,
    Shake256,
)
from hash_frx.keccak.cshake import (
    CSHAKE_SUFFIX,
    CShake128,
    CShake256,
    _CShake,
    _prefix_block,
)
from hash_frx.keccak.encodings import bytepad, encode_string
from hash_frx.keccak.testing.reference import sponge

# `hashlib.shake_128` / `shake_256`: bytes -> an XOF whose `.digest(n)` takes a
# length. `hashlib`'s own XOF type is not public, so the result is `Any`.
_ShakeFactory = Callable[[bytes], Any]

# One case per cSHAKE row: the rate it inherits from the SHAKE it customizes
# (and which its `bytepad` fills), that SHAKE row, and `hashlib`'s. Rates are
# imported rather than spelled, so a row arrives with its rate or not at all —
# `byte_hashes_test`'s rule, and the constants it pins against the FIPS 202
# capacity formula are these.
_CASES = (
    ("cshake128", CShake128, SHAKE128_RATE, Shake128, hashlib.shake_128),
    ("cshake256", CShake256, SHAKE256_RATE, Shake256, hashlib.shake_256),
)


def _boundary_lengths(rate: int) -> tuple[int, ...]:
    """Lengths that land where a sponge can get the padding wrong: exactly on a
    rate boundary, one short of it (suffix and closing bit share a byte), and
    one past it."""
    return (0, 1, rate - 1, rate, rate + 1, 2 * rate)


def _batch(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).reshape(1, -1)


def _digest(row: _CShake, data: bytes) -> bytes:
    return bytes(np.asarray(row.digest(_batch(data)))[0])


def _reference_cshake(
    data: bytes, out_bytes: int, name: bytes, customization: bytes, rate: int
) -> bytes:
    """Section 3.3 over the plain-Python sponge, written as the standard states
    it — the branch spelled out rather than folded into a constant, so it does
    not reproduce `cshake.py`'s structure."""
    if not name and not customization:
        return sponge(data, rate, SHAKE_SUFFIX, out_bytes)
    prefix = bytepad(encode_string(name) + encode_string(customization), rate)
    return sponge(prefix + data, rate, CSHAKE_SUFFIX, out_bytes)


# The two `(N, S)` shapes the reference sweeps run: a bare customization, and a
# non-empty function name beside one. The second is what KMAC will use, and
# without it the `N` limb of `encode_string(N) ‖ encode_string(S)` is never
# compared against anything but itself.
_NAMED = ((b"", b"S"), (b"KMAC", b"Email Signature"))


@dataclass(frozen=True)
class _Vector:
    """One published sample. Parsed out of NIST's cSHAKE_samples.pdf rather than
    typed: a transcribed vector and a wrong implementation fail identically."""

    sample: int
    row: type[_CShake]
    name: bytes
    customization: bytes
    data: bytes
    outval: bytes


_VECTORS = (
    _Vector(
        sample=1,
        row=CShake128,
        name=b"",
        customization=b"Email Signature",
        data=bytes.fromhex("00010203"),
        outval=bytes.fromhex(
            "C1C36925B6409A04F1B504FCBCA9D82B4017277CB5ED2B2065FC1D3814D5AAF5"
        ),
    ),
    _Vector(
        sample=2,
        row=CShake128,
        name=b"",
        customization=b"Email Signature",
        data=bytes.fromhex(
            "000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F"
            "202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F"
            "404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F"
            "606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F"
            "808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9F"
            "A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF"
            "C0C1C2C3C4C5C6C7"
        ),
        outval=bytes.fromhex(
            "C5221D50E4F822D96A2E8881A961420F294B7B24FE3D2094BAED2C6524CC166B"
        ),
    ),
    _Vector(
        sample=3,
        row=CShake256,
        name=b"",
        customization=b"Email Signature",
        data=bytes.fromhex("00010203"),
        outval=bytes.fromhex(
            "D008828E2B80AC9D2218FFEE1D070C48B8E4C87BFF32C9699D5B6896EEE0EDD1"
            "64020E2BE0560858D9C00C037E34A96937C561A74C412BB4C746469527281C8C"
        ),
    ),
    _Vector(
        sample=4,
        row=CShake256,
        name=b"",
        customization=b"Email Signature",
        data=bytes.fromhex(
            "000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F"
            "202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F"
            "404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F"
            "606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F"
            "808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9F"
            "A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF"
            "C0C1C2C3C4C5C6C7"
        ),
        outval=bytes.fromhex(
            "07DC27B11E51FBAC75BC7B3C1D983E8B4B85FB1DEFAF218912AC864302730917"
            "27F42B17ED1DF63E8EC118F04B23633C1DFB1574C8FB55CB45DA8E25AFB092BB"
        ),
    ),
)


class PublishedVectorTest(parameterized.TestCase):
    """All four of NIST's cSHAKE samples."""

    @parameterized.named_parameters(*((f"sample_{v.sample}", v) for v in _VECTORS))
    def test_matches_the_published_sample(self, vector: _Vector) -> None:
        row = vector.row(
            customization=vector.customization,
            name=vector.name,
            output_size=len(vector.outval),
        )
        self.assertEqual(_digest(row, vector.data), vector.outval)

    def test_the_vectors_were_parsed_at_the_published_lengths(self) -> None:
        # The samples state their own lengths in bits; a vector whose declared
        # and actual lengths disagree was mis-parsed, and that is a different
        # failure from a wrong implementation.
        self.assertEqual({len(v.data) for v in _VECTORS}, {4, 200})
        self.assertEqual({len(v.outval) for v in _VECTORS}, {32, 64})


class FallbackTest(parameterized.TestCase):
    """Section 3.3's empty-`N`-and-`S` case, from both sides."""

    @parameterized.named_parameters(*_CASES)
    def test_empty_name_and_customization_is_plain_shake(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # Against `hashlib`, which implements FIPS 202 without sharing a line
        # with this tree — so this is agreement with the standard rather than
        # with a second copy of one reading of it.
        for length in _boundary_lengths(rate):
            data = bytes(i % 256 for i in range(length))
            self.assertEqual(
                _digest(row(output_size=32), data),
                reference(data).digest(32),
                f"length={length}",
            )
        data = bytes(i % 256 for i in range(200))
        for out in (16, 64, rate + 1):
            self.assertEqual(
                _digest(row(output_size=out), data),
                reference(data).digest(out),
                f"out={out}",
            )

    @parameterized.named_parameters(*_CASES)
    def test_the_fallback_equals_the_plain_shake_row(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # And equals this tree's own SHAKE row, which is the substitution a
        # consumer actually makes.
        data = bytes(range(200))
        for out in (16, 32, 64):
            self.assertEqual(
                _digest(row(output_size=out), data), _digest(shake(out), data)
            )

    @parameterized.named_parameters(*_CASES)
    def test_the_fallback_is_a_branch_not_an_identity(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # The heart of it. Had the prefix path been taken at empty N and S, the
        # digest would have been well-defined and NOT SHAKE's — so an
        # implementation that forgot to branch produces a self-consistent wrong
        # hash. This asserts the road not taken really does diverge.
        data = b"abc"
        unbranched = sponge(
            bytepad(encode_string(b"") + encode_string(b""), rate) + data,
            rate,
            CSHAKE_SUFFIX,
            32,
        )
        self.assertNotEqual(unbranched, reference(data).digest(32))
        self.assertEqual(_digest(row(output_size=32), data), reference(data).digest(32))

    @parameterized.named_parameters(*_CASES)
    def test_only_both_empty_takes_the_fallback(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # Asserted through `digest` rather than through the prefix, so it holds
        # whatever the construction does internally: a customization with an
        # empty name still customizes, and vice versa.
        data = b"abc"
        plain = reference(data).digest(32)
        self.assertEqual(_digest(row(output_size=32), data), plain)
        self.assertNotEqual(
            _digest(row(customization=b"S", output_size=32), data), plain
        )
        self.assertNotEqual(_digest(row(name=b"N", output_size=32), data), plain)


class DomainSeparationTest(parameterized.TestCase):
    """That the customization actually separates — the construction's purpose."""

    @parameterized.named_parameters(*_CASES)
    def test_different_customizations_disagree(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        data = b"the same message"
        digests = {
            _digest(row(customization=s, output_size=32), data)
            for s in (b"", b"a", b"b", b"Email Signature")
        }
        self.assertLen(digests, 4)

    @parameterized.named_parameters(*_CASES)
    def test_name_and_customization_are_not_interchangeable(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # `encode_string(N) || encode_string(S)` is unambiguous, so swapping the
        # two is a different prefix. A `bytepad` over the raw concatenation
        # would make these collide.
        data = b"msg"
        self.assertNotEqual(
            _digest(row(customization=b"beta", name=b"alpha", output_size=32), data),
            _digest(row(customization=b"alpha", name=b"beta", output_size=32), data),
        )

    @parameterized.named_parameters(*_CASES)
    def test_a_split_customization_does_not_collide(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # The unambiguous-parsing property: ("ab", "") and ("a", "b") share a
        # byte concatenation and must not share a digest.
        data = b"msg"
        self.assertNotEqual(
            _digest(row(customization=b"ab", name=b"", output_size=32), data),
            _digest(row(customization=b"b", name=b"a", output_size=32), data),
        )


class ReferenceSweepTest(parameterized.TestCase):
    """The device path against the plain-Python sponge, where no published
    vector reaches."""

    @parameterized.named_parameters(*_CASES)
    def test_matches_the_reference_across_absorb_boundaries(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        for length in _boundary_lengths(rate):
            data = bytes(i % 256 for i in range(length))
            for name, custom in _NAMED:
                self.assertEqual(
                    _digest(row(customization=custom, name=name, output_size=32), data),
                    _reference_cshake(data, 32, name, custom, rate),
                    f"length={length} name={name!r} custom={custom!r}",
                )

    @parameterized.named_parameters(*_CASES)
    def test_matches_the_reference_across_squeeze_boundaries(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # Outputs that cross a rate boundary force the squeeze to permute again.
        data = b"squeeze"
        for out in (1, rate - 1, rate, rate + 1, 2 * rate):
            self.assertEqual(
                _digest(row(customization=b"S", output_size=out), data),
                _reference_cshake(data, out, b"", b"S", rate),
                f"out={out}",
            )

    @parameterized.named_parameters(*_CASES)
    def test_a_customization_spanning_blocks_still_agrees(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # A customization long enough to push `bytepad` past one block: the
        # prefix is still rate-aligned, so the message still starts fresh.
        for size in (rate, rate + 1, 3 * rate):
            custom = bytes(i % 256 for i in range(size))
            self.assertEqual(len(_prefix_block(b"", custom, rate)) % rate, 0)
            self.assertEqual(
                _digest(row(customization=custom, output_size=32), b"msg"),
                _reference_cshake(b"msg", 32, b"", custom, rate),
                f"customization={size}B",
            )


class SeamTest(parameterized.TestCase):
    """What these rows owe the seam beyond what the shipped-row registry says.

    `testing/rows.py` registers both, so `row_conformance_test` already runs the
    Protocol check and the equality matrix on them. The equality contract is
    kept here as well, deliberately: it is the jit cache key, its failure mode is
    silent, and this module is where a reader looks for what `N` and `S` do.
    """

    @parameterized.named_parameters(*_CASES)
    def test_digest_size_matches_what_digest_returns(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        for out in (1, 32):
            instance = row(customization=b"S", output_size=out)
            self.assertLen(_digest(instance, b"msg"), instance.digest_size)

    @parameterized.named_parameters(*_CASES)
    def test_reports_the_same_fusion_path_as_the_plain_row(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # cSHAKE lowers through the same sponge as the SHAKE it customizes, so
        # it routes the same way; a different answer would mean the prefix had
        # changed the lowering, which it must not.
        self.assertEqual(
            row(customization=b"S", output_size=32).fusion_path, shake(32).fusion_path
        )

    @parameterized.named_parameters(*_CASES)
    def test_batched_equals_per_row(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        instance = row(customization=b"S", output_size=32)
        messages = [bytes([i] * 40) for i in range(4)]
        batched = np.asarray(
            instance.digest(np.array([list(m) for m in messages], dtype=np.uint8))
        )
        for i, message in enumerate(messages):
            self.assertEqual(bytes(batched[i]), _digest(instance, message))

    @parameterized.named_parameters(*_CASES)
    def test_digest_accepts_a_tracer(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # The prefix is a host constant and every loop bound is static, so
        # nothing here reads a message byte.
        instance = row(customization=b"S", output_size=32)
        data = bytes(range(80))
        traced = frx.jit(instance.digest)(fnp.asarray(_batch(data)))
        self.assertEqual(bytes(np.asarray(traced)[0]), _digest(instance, data))

    @parameterized.named_parameters(*_CASES)
    def test_value_identity_covers_every_parameter(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # Each parameter varied alone, because varying two at once passes while
        # either is ignored.
        base = row(customization=b"a", name=b"n", output_size=32)
        self.assertEqual(base, row(customization=b"a", name=b"n", output_size=32))
        self.assertEqual(
            hash(base), hash(row(customization=b"a", name=b"n", output_size=32))
        )
        for other in (
            row(customization=b"b", name=b"n", output_size=32),
            row(customization=b"a", name=b"m", output_size=32),
            row(customization=b"a", name=b"n", output_size=64),
        ):
            self.assertNotEqual(base, other)

    @parameterized.named_parameters(*_CASES)
    def test_the_output_length_is_required(
        self, row: type[_CShake], rate: int, shake: type, reference: _ShakeFactory
    ) -> None:
        # An XOF names no output length, so the family refuses a default
        # (`docs/reference/conventions.md`). Keyword-only also stops `row(24)` —
        # the shape every other variable-output row takes — from reading 24 as a
        # customization of twenty-four NUL bytes.
        with self.assertRaises(TypeError):
            row()  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            row(24)  # type: ignore[call-arg,arg-type]

    def test_the_two_rates_are_different_hashes(self) -> None:
        self.assertNotEqual(
            _digest(CShake128(customization=b"S", output_size=32), b"msg"),
            _digest(CShake256(customization=b"S", output_size=32), b"msg"),
        )


if __name__ == "__main__":
    absltest.main()

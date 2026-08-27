# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""KMAC128 / KMAC256 and their XOF forms against SP 800-185 section 4.

All twelve of NIST's published samples — six KMAC, six KMACXOF — run through
both surfaces: the bound row, whose key is `bytes`, and the free function, whose
key is a device operand. A construction that agreed with the standard on one and
not the other would be two implementations, which is exactly what the shared
`_kmac` body exists to prevent.

**The two sample sets share their inputs**, which is what makes the
`right_encode(L)` claim testable rather than asserted: the same key, message,
customization and output length appear in both files with different answers, so
`test_kmac_and_kmacxof_disagree_at_the_same_length` is comparing published
values and not just two of this tree's own outputs.

**The key is only ever a length to the encodings.** `test_one_program_serves
_every_key` pins that a traced key leaves a single trace-cache entry across
different key values — the property that makes a per-call-key MAC compile once,
and the reason `encodings.py` can be host-only at all.

Unlike cSHAKE there is no in-tree differential sweep here. KMAC's whole surface
is `bytepad(encode_string(K), rate) ‖ X ‖ right_encode(L)` handed to a cSHAKE
that `cshake_test` already pins against its own published vectors, so a sweep
against a reference re-assembled from the same encodings would be checking this
file's arithmetic against itself. What is not covered by the published samples
is covered structurally instead: the batch/per-row equality, the two key shapes
agreeing, and the XOF distinction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.keccak.byte_hashes import (
    SHAKE128_RATE,
    SHAKE256_RATE,
    SHAKE_SUFFIX,
)
from hash_frx.keccak.cshake import CSHAKE_SUFFIX
from hash_frx.keccak.encodings import bytepad, encode_string, right_encode
from hash_frx.keccak.kmac import (
    Kmac128,
    Kmac256,
    KmacXof128,
    KmacXof256,
    _Kmac,
    kmac128,
    kmac256,
    kmac_xof128,
    kmac_xof256,
)
from hash_frx.keccak.testing.reference import sponge

# The four rows and the free function each is bound to, with the rate they run
# at. Rates are imported rather than spelled, `byte_hashes_test`'s rule.
_CASES = (
    ("kmac128", Kmac128, kmac128, SHAKE128_RATE, False),
    ("kmac256", Kmac256, kmac256, SHAKE256_RATE, False),
    ("kmac_xof128", KmacXof128, kmac_xof128, SHAKE128_RATE, True),
    ("kmac_xof256", KmacXof256, kmac_xof256, SHAKE256_RATE, True),
)

# The three published inputs every sample draws from, transcribed once rather
# than twelve times. `encodings_test` pins this key independently against the
# `Encoded K` bytes the same document prints.
_KEY = bytes.fromhex("404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F")
_DATA_4 = bytes.fromhex("00010203")
_DATA_200 = bytes.fromhex(
    "000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F"
    "202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F"
    "404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F"
    "606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F"
    "808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9F"
    "A0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF"
    "C0C1C2C3C4C5C6C7"
)


def _batch(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).reshape(1, -1)


def _digest(row: _Kmac, data: bytes) -> bytes:
    return bytes(np.asarray(row.digest(_batch(data)))[0])


@dataclass(frozen=True)
class _Vector:
    """One published sample, parsed out of NIST's KMAC_samples.pdf /
    KMACXOF_samples.pdf rather than typed: a transcribed vector and a wrong
    implementation fail identically."""

    name: str
    row: type[_Kmac]
    free: Callable[..., object]
    key: bytes
    customization: bytes
    data: bytes
    outval: bytes


_VECTORS = (
    _Vector(
        name="KMAC1",
        row=Kmac128,
        free=kmac128,
        key=_KEY,
        customization=b"",
        data=_DATA_4,
        outval=bytes.fromhex(
            "E5780B0D3EA6F7D3A429C5706AA43A00FADBD7D49628839E3187243F456EE14E"
        ),
    ),
    _Vector(
        name="KMAC2",
        row=Kmac128,
        free=kmac128,
        key=_KEY,
        customization=b"My Tagged Application",
        data=_DATA_4,
        outval=bytes.fromhex(
            "3B1FBA963CD8B0B59E8C1A6D71888B7143651AF8BA0A7070C0979E2811324AA5"
        ),
    ),
    _Vector(
        name="KMAC3",
        row=Kmac128,
        free=kmac128,
        key=_KEY,
        customization=b"My Tagged Application",
        data=_DATA_200,
        outval=bytes.fromhex(
            "1F5B4E6CCA02209E0DCB5CA635B89A15E271ECC760071DFD805FAA38F9729230"
        ),
    ),
    _Vector(
        name="KMAC4",
        row=Kmac256,
        free=kmac256,
        key=_KEY,
        customization=b"My Tagged Application",
        data=_DATA_4,
        outval=bytes.fromhex(
            "20C570C31346F703C9AC36C61C03CB64C3970D0CFC787E9B79599D273A68D2F7"
            "F69D4CC3DE9D104A351689F27CF6F5951F0103F33F4F24871024D9C27773A8DD"
        ),
    ),
    _Vector(
        name="KMAC5",
        row=Kmac256,
        free=kmac256,
        key=_KEY,
        customization=b"",
        data=_DATA_200,
        outval=bytes.fromhex(
            "75358CF39E41494E949707927CEE0AF20A3FF553904C86B08F21CC414BCFD691"
            "589D27CF5E15369CBBFF8B9A4C2EB17800855D0235FF635DA82533EC6B759B69"
        ),
    ),
    _Vector(
        name="KMAC6",
        row=Kmac256,
        free=kmac256,
        key=_KEY,
        customization=b"My Tagged Application",
        data=_DATA_200,
        outval=bytes.fromhex(
            "B58618F71F92E1D56C1B8C55DDD7CD188B97B4CA4D99831EB2699A837DA2E4D9"
            "70FBACFDE50033AEA585F1A2708510C32D07880801BD182898FE476876FC8965"
        ),
    ),
    _Vector(
        name="KMACXOF1",
        row=KmacXof128,
        free=kmac_xof128,
        key=_KEY,
        customization=b"",
        data=_DATA_4,
        outval=bytes.fromhex(
            "CD83740BBD92CCC8CF032B1481A0F4460E7CA9DD12B08A0C4031178BACD6EC35"
        ),
    ),
    _Vector(
        name="KMACXOF2",
        row=KmacXof128,
        free=kmac_xof128,
        key=_KEY,
        customization=b"My Tagged Application",
        data=_DATA_4,
        outval=bytes.fromhex(
            "31A44527B4ED9F5C6101D11DE6D26F0620AA5C341DEF41299657FE9DF1A3B16C"
        ),
    ),
    _Vector(
        name="KMACXOF3",
        row=KmacXof128,
        free=kmac_xof128,
        key=_KEY,
        customization=b"My Tagged Application",
        data=_DATA_200,
        outval=bytes.fromhex(
            "47026C7CD793084AA0283C253EF658490C0DB61438B8326FE9BDDF281B83AE0F"
        ),
    ),
    _Vector(
        name="KMACXOF4",
        row=KmacXof256,
        free=kmac_xof256,
        key=_KEY,
        customization=b"My Tagged Application",
        data=_DATA_4,
        outval=bytes.fromhex(
            "1755133F1534752AAD0748F2C706FB5C784512CAB835CD15676B16C0C6647FA9"
            "6FAA7AF634A0BF8FF6DF39374FA00FAD9A39E322A7C92065A64EB1FB0801EB2B"
        ),
    ),
    _Vector(
        name="KMACXOF5",
        row=KmacXof256,
        free=kmac_xof256,
        key=_KEY,
        customization=b"",
        data=_DATA_200,
        outval=bytes.fromhex(
            "FF7B171F1E8A2B24683EED37830EE797538BA8DC563F6DA1E667391A75EDC02C"
            "A633079F81CE12A25F45615EC89972031D18337331D24CEB8F8CA8E6A19FD98B"
        ),
    ),
    _Vector(
        name="KMACXOF6",
        row=KmacXof256,
        free=kmac_xof256,
        key=_KEY,
        customization=b"My Tagged Application",
        data=_DATA_200,
        outval=bytes.fromhex(
            "D5BE731C954ED7732846BB59DBE3A8E30F83E77A4BFF4459F2F1C2B4ECEBB8CE"
            "67BA01C62E8AB8578D2D499BD1BB276768781190020A306A97DE281DCC30305D"
        ),
    ),
)


class PublishedVectorTest(parameterized.TestCase):
    """All twelve samples, through both surfaces."""

    @parameterized.named_parameters(*((v.name, v) for v in _VECTORS))
    def test_the_bound_row_matches(self, vector: _Vector) -> None:
        row = vector.row(
            vector.key, vector.customization, output_size=len(vector.outval)
        )
        self.assertEqual(_digest(row, vector.data), vector.outval)

    @parameterized.named_parameters(*((v.name, v) for v in _VECTORS))
    def test_the_operand_key_matches(self, vector: _Vector) -> None:
        # The same vector with the key as a device operand rather than baked
        # into the row. Two surfaces, one `_kmac` body, one published answer.
        got = vector.free(
            fnp.asarray(np.frombuffer(vector.key, dtype=np.uint8)),
            _batch(vector.data),
            len(vector.outval),
            vector.customization,
        )
        self.assertEqual(bytes(np.asarray(got)[0]), vector.outval)

    def test_the_vectors_were_parsed_at_the_published_lengths(self) -> None:
        # A mis-parse is a different failure from a wrong implementation.
        self.assertEqual({len(v.outval) for v in _VECTORS}, {32, 64})
        self.assertEqual((len(_KEY), len(_DATA_4), len(_DATA_200)), (32, 4, 200))
        self.assertLen(_VECTORS, 12)


class XofDistinctionTest(parameterized.TestCase):
    """Section 4.3.1: the encoded output length, not the read length."""

    @parameterized.named_parameters(
        ("bits128", Kmac128, KmacXof128), ("bits256", Kmac256, KmacXof256)
    )
    def test_kmac_and_kmacxof_disagree_at_the_same_length(
        self, plain: type[_Kmac], xof: type[_Kmac]
    ) -> None:
        # Not one function truncated two ways. `right_encode(8 * L)` against
        # `right_encode(0)` changes the final absorbed bytes, so the streams
        # diverge from the first byte out.
        self.assertNotEqual(
            _digest(plain(_KEY, output_size=32), b"msg"),
            _digest(xof(_KEY, output_size=32), b"msg"),
        )

    def test_the_published_pairs_share_inputs_and_differ(self) -> None:
        # The strongest form of the claim: NIST publishes both functions over
        # identical (key, data, S, L), so this compares published values.
        pairs = 0
        for plain in (v for v in _VECTORS if not v.name.startswith("KMACXOF")):
            twin = next(
                v
                for v in _VECTORS
                if v.name == "KMACXOF" + plain.name.removeprefix("KMAC")
            )
            self.assertEqual(
                (plain.key, plain.data, plain.customization, len(plain.outval)),
                (twin.key, twin.data, twin.customization, len(twin.outval)),
            )
            self.assertNotEqual(plain.outval, twin.outval)
            pairs += 1
        self.assertEqual(pairs, 6)


class ConstructionTest(parameterized.TestCase):
    """Section 4.3's assembly, checked where a digest alone would not say why."""

    @parameterized.named_parameters(*_CASES)
    def test_the_trailing_length_is_encoded_in_bits(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        # `right_encode(8 * output_size)`, not `right_encode(output_size)`.
        # Both are legal encodings of a plausible number, so the wrong one is a
        # self-consistent different hash. Asserted by rebuilding section 4.3's
        # `newX` both ways over the sponge and requiring the bit form to be the
        # one the row computes -- and the byte form not to be.
        out, data = 32, b"msg"
        key_block = bytepad(encode_string(_KEY), rate)
        prefix = bytepad(encode_string(b"KMAC") + encode_string(b""), rate)

        def absorbed(tail: bytes) -> bytes:
            return sponge(prefix + key_block + data + tail, rate, CSHAKE_SUFFIX, out)

        digest = _digest(row(_KEY, output_size=out), data)
        self.assertEqual(digest, absorbed(right_encode(0 if xof else 8 * out)))
        if not xof:
            # The XOF form encodes 0 either way, so only the plain rows can
            # tell the two units apart -- which is why this half is guarded.
            self.assertNotEqual(digest, absorbed(right_encode(out)))

    @parameterized.named_parameters(*_CASES)
    def test_the_domain_byte_is_cshakes_whatever_the_customization(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        # KMAC's `N` is the literal "KMAC" (SP 800-185 section 4.3), so section
        # 3.3's both-empty fallback cannot fire and the domain byte is 0x04 even
        # with no customization.
        #
        # Asserted through `digest` against the oracle, NOT through the row's
        # `_sponge`: an assertion on that attribute passes even when the hashing
        # path uses a different suffix, which is exactly the coverage illusion
        # this test previously had.
        data, out = b"msg", 32
        tail = right_encode(0) if xof else right_encode(8 * out)
        for customization in (b"", b"S"):
            stream = (
                bytepad(encode_string(b"KMAC") + encode_string(customization), rate)
                + bytepad(encode_string(_KEY), rate)
                + data
                + tail
            )
            digest = _digest(row(_KEY, customization, output_size=out), data)
            self.assertEqual(digest, sponge(stream, rate, CSHAKE_SUFFIX, out))
            self.assertNotEqual(digest, sponge(stream, rate, SHAKE_SUFFIX, out))

    @parameterized.named_parameters(*_CASES)
    def test_an_empty_customization_is_not_a_missing_one(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        # Different customizations separate, and the empty one is just another
        # value rather than a way to opt out.
        digests = {
            _digest(row(_KEY, s, output_size=32), b"msg")
            for s in (b"", b"a", b"My Tagged Application")
        }
        self.assertLen(digests, 3)


class KeyTest(parameterized.TestCase):
    """The two key surfaces, and the property that lets the second exist."""

    @parameterized.named_parameters(*_CASES)
    def test_the_two_surfaces_agree(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        # `_DATA_4`, not a fresh length: at both rates this shares the fused
        # sponge's trace key with the four-byte published vectors above, and a
        # 200-byte body here would restate what KMAC3/5/6 already run through
        # BOTH surfaces at a key of its own.
        data = _DATA_4
        bound = _digest(row(_KEY, b"S", output_size=32), data)
        operand = free(
            fnp.asarray(np.frombuffer(_KEY, dtype=np.uint8)), _batch(data), 32, b"S"
        )
        self.assertEqual(bytes(np.asarray(operand)[0]), bound)

    @parameterized.named_parameters(*_CASES)
    def test_one_program_serves_every_key(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        # The point of the operand form, and the evidence that only the key's
        # LENGTH reaches the encodings: two different key values, one trace.
        jitted = frx.jit(lambda k, m: free(k, m, 32))
        message = fnp.asarray(_batch(b"msg"))
        first = np.asarray(
            jitted(fnp.asarray(np.frombuffer(_KEY, dtype=np.uint8)), message)
        )
        second = np.asarray(
            jitted(fnp.asarray(np.frombuffer(_KEY[::-1], dtype=np.uint8)), message)
        )
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(jitted._cache_size(), 1)

    @parameterized.named_parameters(*_CASES)
    def test_a_per_message_key_matches_one_message_at_a_time(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        # `[B, K]` keys, the shape `Hmac.mac` also takes.
        # Four rows rather than two, so this shares a trace key with
        # `test_batched_equals_per_row` -- the batch size is part of the fused
        # sponge's cache key, and a fresh one costs seconds.
        keys = np.stack(
            [np.frombuffer(_KEY, dtype=np.uint8)] * 2
            + [np.frombuffer(_KEY[::-1], dtype=np.uint8)] * 2
        )
        messages = np.stack([np.arange(40, dtype=np.uint8)] * 4)
        batched = np.asarray(free(keys, messages, 32))
        for i in range(4):
            single = free(keys[i], messages[i : i + 1], 32)
            self.assertEqual(bytes(batched[i]), bytes(np.asarray(single)[0]))

    @parameterized.named_parameters(*_CASES)
    def test_a_different_key_is_a_different_tag(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        self.assertNotEqual(
            _digest(row(_KEY, output_size=32), b"msg"),
            _digest(row(_KEY[::-1], output_size=32), b"msg"),
        )

    @parameterized.named_parameters(*_CASES)
    def test_a_malformed_operand_key_is_rejected(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        with self.assertRaisesRegex(ValueError, r"key must be \[K\] or \[B, K\]"):
            free(np.zeros((2, 2, 2), dtype=np.uint8), _batch(b"msg"), 32)


class SeamTest(parameterized.TestCase):
    """What these rows owe the seam beyond the shipped-row registry."""

    @parameterized.named_parameters(*_CASES)
    def test_digest_size_matches_what_digest_returns(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        instance = row(_KEY, output_size=32)
        self.assertLen(_digest(instance, b"msg"), instance.digest_size)

    @parameterized.named_parameters(*_CASES)
    def test_batched_equals_per_row(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        instance = row(_KEY, b"S", output_size=32)
        messages = [bytes([i] * 40) for i in range(4)]
        batched = np.asarray(
            instance.digest(np.array([list(m) for m in messages], dtype=np.uint8))
        )
        for i, message in enumerate(messages):
            self.assertEqual(bytes(batched[i]), _digest(instance, message))

    @parameterized.named_parameters(*_CASES)
    def test_digest_accepts_a_tracer(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        instance = row(_KEY, b"S", output_size=32)
        data = bytes(range(80))
        traced = frx.jit(instance.digest)(fnp.asarray(_batch(data)))
        self.assertEqual(bytes(np.asarray(traced)[0]), _digest(instance, data))

    @parameterized.named_parameters(*_CASES)
    def test_value_identity_covers_every_parameter(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        # Each parameter varied alone. The key especially: two rows that compare
        # equal share a compiled program, and for a MAC that is the key leaking
        # across a cache hit.
        base = row(_KEY, b"s", output_size=32)
        self.assertEqual(base, row(_KEY, b"s", output_size=32))
        self.assertEqual(hash(base), hash(row(_KEY, b"s", output_size=32)))
        for other in (
            row(_KEY[::-1], b"s", output_size=32),
            row(_KEY, b"t", output_size=32),
            row(_KEY, b"s", output_size=64),
        ):
            self.assertNotEqual(base, other)

    def test_the_four_rows_are_four_different_hashes(self) -> None:
        digests = {
            _digest(r(_KEY, b"S", output_size=32), b"msg") for _, r, _, _, _ in _CASES
        }
        self.assertLen(digests, 4)

    @parameterized.named_parameters(*_CASES)
    def test_the_output_length_is_required(
        self, row: type[_Kmac], free: Callable[..., object], rate: int, xof: bool
    ) -> None:
        # An XOF names no output length, so the family refuses a default, and
        # keyword-only stops a positional length being read as a customization.
        with self.assertRaises(TypeError):
            row(_KEY)  # type: ignore[call-arg]


if __name__ == "__main__":
    absltest.main()

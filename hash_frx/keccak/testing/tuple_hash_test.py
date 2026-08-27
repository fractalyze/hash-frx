# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""TupleHash against SP 800-185 section 5.

All twelve published samples — six TupleHash, six TupleHashXOF, at both rates
and both output lengths — through both surfaces: the bound row and the free
function.

**The headline claim is tested directly.** TupleHash exists because
`("ab", "c")`, `("a", "bc")` and `("abc")` are one byte string once concatenated
and three different inputs otherwise. `AmbiguityTest` asserts they produce three
different digests, which is the property a consumer reaches for this
construction to get and the one no published vector isolates.

**The two sample sets share their tuples**, so the `right_encode(L)` claim is
comparing published values rather than two of this tree's own outputs — the same
arrangement `kmac_test` relies on.

Sweeps are unions rather than cross products, and output sizes are held fixed
where the claim does not depend on them: every distinct `(rate, output_size,
total length)` re-traces the fused sponge, which is what the time in this
directory is made of.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.byte_hash import ByteHash
from hash_frx.keccak.byte_hashes import SHAKE128_RATE, SHAKE256_RATE, Shake128
from hash_frx.keccak.cshake import CSHAKE_SUFFIX
from hash_frx.keccak.encodings import bytepad, encode_string, right_encode
from hash_frx.keccak.testing.reference import sponge
from hash_frx.keccak.tuple_hash import (
    TupleHash128,
    TupleHash256,
    TupleHashXof128,
    TupleHashXof256,
    _TupleHash,
    tuple_hash128,
    tuple_hash256,
    tuple_hash_xof128,
    tuple_hash_xof256,
)

_CASES = (
    # No free-function slot: the free functions are covered through `_VECTORS`,
    # and a parameter no body reads is weight every reader has to check.
    ("tuplehash128", TupleHash128, SHAKE128_RATE, False),
    ("tuplehash256", TupleHash256, SHAKE256_RATE, False),
    ("tuplehash_xof128", TupleHashXof128, SHAKE128_RATE, True),
    ("tuplehash_xof256", TupleHashXof256, SHAKE256_RATE, True),
)


def _batch(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).reshape(1, -1)


def _hash(row: _TupleHash, strings: Sequence[bytes]) -> bytes:
    return bytes(np.asarray(row.hash([_batch(s) for s in strings]))[0])


# The published tuple elements: every sample draws its strings from these three,
# so they are transcribed once rather than per vector.
_S1 = bytes.fromhex("000102")
_S2 = bytes.fromhex("101112131415")
_S3 = bytes.fromhex("202122232425262728")
_PAIR = (_S1, _S2)
_TRIPLE = (_S1, _S2, _S3)


@dataclass(frozen=True)
class _Vector:
    """One published sample, parsed out of NIST's TupleHash_samples.pdf /
    TupleHashXOF_samples.pdf rather than typed."""

    name: str
    row: type[_TupleHash]
    free: Callable[..., object]
    strings: tuple[bytes, ...]
    customization: bytes
    outval: bytes


_VECTORS = (
    _Vector(
        name="TupleHash1",
        row=TupleHash128,
        free=tuple_hash128,
        strings=_PAIR,
        customization=b"",
        outval=bytes.fromhex(
            "C5D8786C1AFB9B82111AB34B65B2C0048FA64E6D48E263264CE1707D3FFC8ED1"
        ),
    ),
    _Vector(
        name="TupleHash2",
        row=TupleHash128,
        free=tuple_hash128,
        strings=_PAIR,
        customization=b"My Tuple App",
        outval=bytes.fromhex(
            "75CDB20FF4DB1154E841D758E24160C54BAE86EB8C13E7F5F40EB35588E96DFB"
        ),
    ),
    _Vector(
        name="TupleHash3",
        row=TupleHash128,
        free=tuple_hash128,
        strings=_TRIPLE,
        customization=b"My Tuple App",
        outval=bytes.fromhex(
            "E60F202C89A2631EDA8D4C588CA5FD07F39E5151998DECCF973ADB3804BB6E84"
        ),
    ),
    _Vector(
        name="TupleHash4",
        row=TupleHash256,
        free=tuple_hash256,
        strings=_PAIR,
        customization=b"",
        outval=bytes.fromhex(
            "CFB7058CACA5E668F81A12A20A2195CE97A925F1DBA3E7449A56F82201EC6073"
            "11AC2696B1AB5EA2352DF1423BDE7BD4BB78C9AED1A853C78672F9EB23BBE194"
        ),
    ),
    _Vector(
        name="TupleHash5",
        row=TupleHash256,
        free=tuple_hash256,
        strings=_PAIR,
        customization=b"My Tuple App",
        outval=bytes.fromhex(
            "147C2191D5ED7EFD98DBD96D7AB5A11692576F5FE2A5065F3E33DE6BBA9F3AA1"
            "C4E9A068A289C61C95AAB30AEE1E410B0B607DE3620E24A4E3BF9852A1D4367E"
        ),
    ),
    _Vector(
        name="TupleHash6",
        row=TupleHash256,
        free=tuple_hash256,
        strings=_TRIPLE,
        customization=b"My Tuple App",
        outval=bytes.fromhex(
            "45000BE63F9B6BFD89F54717670F69A9BC763591A4F05C50D68891A744BCC6E7"
            "D6D5B5E82C018DA999ED35B0BB49C9678E526ABD8E85C13ED254021DB9E790CE"
        ),
    ),
    _Vector(
        name="TupleHashXOF1",
        row=TupleHashXof128,
        free=tuple_hash_xof128,
        strings=_PAIR,
        customization=b"",
        outval=bytes.fromhex(
            "2F103CD7C32320353495C68DE1A8129245C6325F6F2A3D608D92179C96E68488"
        ),
    ),
    _Vector(
        name="TupleHashXOF2",
        row=TupleHashXof128,
        free=tuple_hash_xof128,
        strings=_PAIR,
        customization=b"My Tuple App",
        outval=bytes.fromhex(
            "3FC8AD69453128292859A18B6C67D7AD85F01B32815E22CE839C49EC374E9B9A"
        ),
    ),
    _Vector(
        name="TupleHashXOF3",
        row=TupleHashXof128,
        free=tuple_hash_xof128,
        strings=_TRIPLE,
        customization=b"My Tuple App",
        outval=bytes.fromhex(
            "900FE16CAD098D28E74D632ED852F99DAAB7F7DF4D99E775657885B4BF76D6F8"
        ),
    ),
    _Vector(
        name="TupleHashXOF4",
        row=TupleHashXof256,
        free=tuple_hash_xof256,
        strings=_PAIR,
        customization=b"",
        outval=bytes.fromhex(
            "03DED4610ED6450A1E3F8BC44951D14FBC384AB0EFE57B000DF6B6DF5AAE7CD5"
            "68E77377DAF13F37EC75CF5FC598B6841D51DD207C991CD45D210BA60AC52EB9"
        ),
    ),
    _Vector(
        name="TupleHashXOF5",
        row=TupleHashXof256,
        free=tuple_hash_xof256,
        strings=_PAIR,
        customization=b"My Tuple App",
        outval=bytes.fromhex(
            "6483CB3C9952EB20E830AF4785851FC597EE3BF93BB7602C0EF6A65D741AECA7"
            "E63C3B128981AA05C6D27438C79D2754BB1B7191F125D6620FCA12CE658B2442"
        ),
    ),
    _Vector(
        name="TupleHashXOF6",
        row=TupleHashXof256,
        free=tuple_hash_xof256,
        strings=_TRIPLE,
        customization=b"My Tuple App",
        outval=bytes.fromhex(
            "0C59B11464F2336C34663ED51B2B950BEC743610856F36C28D1D088D8A244628"
            "4DD09830A6A178DC752376199FAE935D86CFDEE5913D4922DFD369B66A53C897"
        ),
    ),
)


class PublishedVectorTest(parameterized.TestCase):
    @parameterized.named_parameters(*((v.name, v) for v in _VECTORS))
    def test_the_bound_row_matches(self, vector: _Vector) -> None:
        row = vector.row(vector.customization, output_size=len(vector.outval))
        self.assertEqual(_hash(row, vector.strings), vector.outval)

    @parameterized.named_parameters(*((v.name, v) for v in _VECTORS))
    def test_the_free_function_matches(self, vector: _Vector) -> None:
        got = vector.free(
            [_batch(s) for s in vector.strings],
            len(vector.outval),
            vector.customization,
        )
        self.assertEqual(bytes(np.asarray(got)[0]), vector.outval)

    def test_the_vectors_were_parsed_at_the_published_shapes(self) -> None:
        self.assertLen(_VECTORS, 12)
        self.assertEqual({len(v.outval) for v in _VECTORS}, {32, 64})
        # The samples use two- and three-element tuples of 3/6/9 bytes.
        self.assertEqual(
            {tuple(len(s) for s in v.strings) for v in _VECTORS}, {(3, 6), (3, 6, 9)}
        )


class AmbiguityTest(absltest.TestCase):
    """The property the construction exists to provide."""

    def test_a_split_is_part_of_what_is_hashed(self) -> None:
        # Concatenated, these three are the same five bytes. That collision is
        # what a consumer hashing a tuple by joining it would inherit.
        digests = {
            _hash(TupleHash128(output_size=32), t)
            for t in ((b"ab", b"c"), (b"a", b"bc"), (b"abc",))
        }
        self.assertLen(digests, 3)

    def test_an_empty_element_still_counts(self) -> None:
        # ("a", "") and ("a",) differ: the empty element contributes its own
        # `encode_string`, which is `01 00`.
        self.assertNotEqual(
            _hash(TupleHash128(output_size=32), (b"a", b"")),
            _hash(TupleHash128(output_size=32), (b"a",)),
        )

    def test_arity_changes_the_digest(self) -> None:
        row = TupleHash128(output_size=32)
        digests = {_hash(row, tuple(b"x" for _ in range(n))) for n in (1, 2, 3, 4)}
        self.assertLen(digests, 4)


class XofDistinctionTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("bits128", TupleHash128, TupleHashXof128),
        ("bits256", TupleHash256, TupleHashXof256),
    )
    def test_the_two_forms_disagree_at_the_same_length(
        self, plain: type[_TupleHash], xof: type[_TupleHash]
    ) -> None:
        self.assertNotEqual(
            _hash(plain(output_size=32), (b"a", b"bc")),
            _hash(xof(output_size=32), (b"a", b"bc")),
        )

    def test_the_published_pairs_share_inputs_and_differ(self) -> None:
        pairs = 0
        for plain in (v for v in _VECTORS if not v.name.startswith("TupleHashXOF")):
            twin = next(
                v
                for v in _VECTORS
                if v.name == "TupleHashXOF" + plain.name.removeprefix("TupleHash")
            )
            self.assertEqual(
                (plain.strings, plain.customization, len(plain.outval)),
                (twin.strings, twin.customization, len(twin.outval)),
            )
            self.assertNotEqual(plain.outval, twin.outval)
            pairs += 1
        self.assertEqual(pairs, 6)


class ConstructionTest(parameterized.TestCase):
    @parameterized.named_parameters(*_CASES)
    def test_it_is_cshake_over_the_encoded_tuple(
        self, row: type[_TupleHash], rate: int, xof: bool
    ) -> None:
        # Section 5.3 rebuilt over the plain-Python sponge, spelled as the
        # standard states it. The domain byte is cSHAKE's 0x04 -- `N` is the
        # constant "TupleHash", so section 3.3's both-empty fallback cannot
        # fire -- and the negative half pins that rather than assuming it.
        strings, out = (b"abc", b"defghi"), 32
        z = b"".join(encode_string(s) for s in strings)
        stream = (
            bytepad(encode_string(b"TupleHash") + encode_string(b""), rate)
            + z
            + (right_encode(0) if xof else right_encode(8 * out))
        )
        digest = _hash(row(output_size=out), strings)
        self.assertEqual(digest, sponge(stream, rate, CSHAKE_SUFFIX, out))
        self.assertNotEqual(digest, sponge(stream, rate, 0x1F, out))

    @parameterized.named_parameters(*_CASES)
    def test_different_customizations_disagree(
        self, row: type[_TupleHash], rate: int, xof: bool
    ) -> None:
        digests = {
            _hash(row(s, output_size=32), (b"a", b"bc"))
            for s in (b"", b"x", b"My Tuple App")
        }
        self.assertLen(digests, 3)

    @parameterized.named_parameters(*_CASES)
    def test_elements_may_have_different_lengths(
        self, row: type[_TupleHash], rate: int, xof: bool
    ) -> None:
        # Each element's `encode_string` prefix is read off its own shape, so a
        # ragged tuple is the ordinary case rather than a special one. Thirty-two
        # bytes is the SHORTEST element whose `left_encode(8 * len)` needs two
        # value bytes, so it covers the multi-byte prefix while the padded total
        # still lands on the trace key the rest of this file already pays for --
        # a longer element buys a second encoding of nothing and a fresh trace.
        instance = row(output_size=32)
        self.assertLen(_hash(instance, (b"", b"a", b"b" * 32, b"ccc")), 32)


class RejectionTest(absltest.TestCase):
    def test_an_empty_sequence_is_rejected(self) -> None:
        # Not swept: the check is in `_tuple_message`, reached identically by all
        # four rows before any device work -- the arrangement the two rejection
        # tests below already use.
        with self.assertRaisesRegex(ValueError, "non-empty"):
            TupleHash128(output_size=32).hash([])

    def test_mismatched_batch_sizes_are_rejected(self) -> None:
        # A tuple is hashed per row, so every element must carry the same batch.
        with self.assertRaisesRegex(ValueError, "same batch size"):
            TupleHash128(output_size=32).hash(
                [np.zeros((2, 3), np.uint8), np.zeros((3, 3), np.uint8)]
            )

    def test_a_non_batch_element_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D uint8"):
            TupleHash128(output_size=32).hash([np.zeros(3, np.uint8)])


class SeamTest(parameterized.TestCase):
    """What these rows do and do not promise."""

    @parameterized.named_parameters(*_CASES)
    def test_it_is_deliberately_not_a_byte_hash(
        self, row: type[_TupleHash], rate: int, xof: bool
    ) -> None:
        # The module's central claim, asserted rather than left to the
        # docstring: a `ByteHash` takes one flat message, and flattening a
        # tuple is exactly the ambiguity this construction removes.
        instance = row(output_size=32)
        self.assertNotIsInstance(instance, ByteHash)
        self.assertFalse(hasattr(instance, "digest"))

    @parameterized.named_parameters(*_CASES)
    def test_it_reports_a_fusion_path(
        self, row: type[_TupleHash], rate: int, xof: bool
    ) -> None:
        # It lowers through the same sponge as every other Keccak row, so it
        # answers the routing question even though it is not a `ByteHash`.
        #
        # Compared against a plain SHAKE row rather than asserted non-None:
        # `FusionPath` is an Enum, so `assertIsNotNone` holds whatever the
        # derivation does. This row re-implements that derivation, so the
        # comparison is the only thing pinning it to the sponge it runs on.
        self.assertEqual(row(output_size=32).fusion_path, Shake128(32).fusion_path)

    @parameterized.named_parameters(*_CASES)
    def test_batched_equals_per_row(
        self, row: type[_TupleHash], rate: int, xof: bool
    ) -> None:
        instance = row(b"S", output_size=32)
        first = np.array([[i] * 4 for i in range(4)], dtype=np.uint8)
        second = np.array([[i + 9] * 6 for i in range(4)], dtype=np.uint8)
        batched = np.asarray(instance.hash([first, second]))
        for i in range(4):
            single = instance.hash([first[i : i + 1], second[i : i + 1]])
            self.assertEqual(bytes(batched[i]), bytes(np.asarray(single)[0]))

    @parameterized.named_parameters(*_CASES)
    def test_hash_accepts_tracers(
        self, row: type[_TupleHash], rate: int, xof: bool
    ) -> None:
        # Every length is a shape, so nothing here reads an element byte.
        instance = row(b"S", output_size=32)
        parts = [_batch(b"abc"), _batch(b"defghi")]
        traced = frx.jit(lambda a, b: instance.hash([a, b]))(
            fnp.asarray(parts[0]), fnp.asarray(parts[1])
        )
        self.assertEqual(
            bytes(np.asarray(traced)[0]), _hash(instance, (b"abc", b"defghi"))
        )

    @parameterized.named_parameters(*_CASES)
    def test_value_identity_covers_every_parameter(
        self, row: type[_TupleHash], rate: int, xof: bool
    ) -> None:
        base = row(b"s", output_size=32)
        self.assertEqual(base, row(b"s", output_size=32))
        self.assertEqual(hash(base), hash(row(b"s", output_size=32)))
        for other in (row(b"t", output_size=32), row(b"s", output_size=64)):
            self.assertNotEqual(base, other)

    def test_the_four_rows_are_four_different_hashes(self) -> None:
        digests = {
            _hash(r(b"S", output_size=32), (b"a", b"bc")) for _, r, _, _ in _CASES
        }
        self.assertLen(digests, 4)

    @parameterized.named_parameters(*_CASES)
    def test_the_output_length_is_required(
        self, row: type[_TupleHash], rate: int, xof: bool
    ) -> None:
        with self.assertRaises(TypeError):
            row()  # type: ignore[call-arg]


if __name__ == "__main__":
    absltest.main()

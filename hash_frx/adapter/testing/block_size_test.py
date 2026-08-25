# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The block-size table, and the two families deliberately absent from it.

Every entry is a number from a standard, so the useful cases are the ones a
typo would survive: that the table agrees with the rate constants the sponge
families already export, that host and device rows of one family answer alike,
and that the absences are absences on purpose rather than gaps nobody noticed.

No device work: the lookup is host arithmetic over a row's type.
"""

from __future__ import annotations

from absl.testing import absltest

from hash_frx.adapter.block_size import _BLOCK_SIZES, block_size
from hash_frx.ascon.ascon import AsconHash256
from hash_frx.blake2b import blake2b
from hash_frx.blake2s import blake2s
from hash_frx.blake3.rows import Blake3, Blake3Keyed, HostBlake3
from hash_frx.grostl import grostl
from hash_frx.keccak.byte_hashes import (
    KECCAK256_RATE,
    SHA3_256_RATE,
    SHA3_512_RATE,
    SHAKE128_RATE,
    SHAKE256_RATE,
    Keccak256,
    Sha3_256,
    Sha3_512,
    Shake128,
    Shake256,
)
from hash_frx.ripemd160 import ripemd160
from hash_frx.sha256 import sha256 as sha256_mod
from hash_frx.sha256.sha256 import HostSha256, Sha256
from hash_frx.sha512 import sha512 as sha512_mod
from hash_frx.sha512.sha512 import Sha384, Sha512, Sha512_256
from hash_frx.sm3 import sm3
from hash_frx.testing.rows import BYTE_HASH_ROWS


class KnownWidthsTest(absltest.TestCase):
    def test_the_merkle_damgard_families_agree_with_their_own_pad_rule(
        self,
    ) -> None:
        # Each family already states its block once, as the width of the
        # `PadRule` it pads with. Held against that rather than against a
        # literal here, for the reason the sponge case below gives: a
        # transposed digit in this table ships well-formed bytes under the
        # wrong key schedule, and nothing downstream errors.
        for row, module in (
            (Sha256(), sha256_mod),
            (Sha512(), sha512_mod),
            (Sha384(), sha512_mod),
            (Sha512_256(), sha512_mod),
            (sm3.Sm3(), sm3),
            (ripemd160.Ripemd160(), ripemd160),
            (blake2s.Blake2s(), blake2s),
            (blake2b.Blake2b(), blake2b),
        ):
            with self.subTest(row=type(row).__name__):
                self.assertEqual(block_size(row), module._PAD.block_size)

    def test_grostl_agrees_with_its_own_block_constant(self) -> None:
        # Grostl states its block as `_BLOCK` rather than through a `PadRule`
        # width, so it is read from there.
        self.assertEqual(block_size(grostl.Grostl256()), grostl._BLOCK)

    def test_the_sponge_rows_agree_with_their_own_rate_constants(self) -> None:
        # The entry a typo would survive: 136 and 72 are easy to transpose, and
        # nothing downstream errors on a wrong B. Held against the constants
        # the family already exports rather than against literals here.
        self.assertEqual(block_size(Sha3_256()), SHA3_256_RATE)
        self.assertEqual(block_size(Sha3_512()), SHA3_512_RATE)
        self.assertEqual(block_size(Keccak256()), KECCAK256_RATE)
        self.assertEqual(block_size(Shake128(32)), SHAKE128_RATE)
        self.assertEqual(block_size(Shake256(32)), SHAKE256_RATE)

    def test_host_and_device_rows_of_one_family_agree(self) -> None:
        # They are the same hash; the host row differs in where it runs, not in
        # what it computes, so a block-keyed construction must read one width.
        self.assertEqual(block_size(HostSha256()), block_size(Sha256()))

    def test_output_length_does_not_change_the_block(self) -> None:
        # The width is the family's, not the instance's — which is why the
        # table is keyed by type rather than by row value.
        self.assertEqual(block_size(Shake256(32)), block_size(Shake256(64)))


class DeliberateAbsencesTest(absltest.TestCase):
    """The absences carry the argument, so they are pinned like the entries."""

    def test_blake3_has_no_block_size(self) -> None:
        # Not an oversight: BLAKE3 keys natively (spec section 2.3), so HMAC is
        # the wrong construction over it rather than an unsupported one. Its
        # 64-byte compression block is not an HMAC `B` just because the number
        # exists — `hmac.py` states the rule this pins.
        for row in (Blake3(), Blake3Keyed(bytes(32)), HostBlake3()):
            with self.subTest(row=type(row).__name__):
                # The message also names the file to edit, because a caller
                # hitting this needs to know it is a registry gap rather than a
                # bug in their code.
                with self.assertRaisesRegex(LookupError, "keyed mode is native"):
                    block_size(row)
                with self.assertRaisesRegex(LookupError, r"adapter/block_size\.py"):
                    block_size(row)

    def test_ascon_has_no_block_size(self) -> None:
        # An 8-byte rate against a 32-byte digest, so FIPS 198-1 §4's "replace a
        # longer-than-block key by its digest" has nowhere to put the result.
        # An entry would only move the error into `Hmac.__init__`.
        with self.assertRaises(LookupError):
            block_size(AsconHash256())


class TableNamesRealRowsTest(absltest.TestCase):
    """Every key must name a row that still ships under that name.

    The table is keyed by class name, which is what keeps its bazel target free
    of every row module — and what makes a rename drop an entry silently. This
    is the check that turns that trade into a caught error rather than a wrong
    key schedule. `testing/rows.py` is the registry the package already keeps
    complete (`row_conformance_test` pins it), so the two cannot drift apart.
    """

    def test_every_key_names_a_shipped_row(self) -> None:
        shipped = {type(case.make()).__name__ for case in BYTE_HASH_ROWS}
        shipped |= {n.removeprefix("Host") for n in shipped}
        for name in _BLOCK_SIZES:
            with self.subTest(hash=name):
                self.assertIn(
                    name,
                    shipped,
                    f"{name} has a block size but no such row ships — a rename "
                    "drops the entry silently, which is a wrong key schedule "
                    "rather than an error",
                )


class TableShapeTest(absltest.TestCase):
    def test_every_width_is_a_positive_multiple_of_eight(self) -> None:
        # Every standard in the table defines its block in whole 64-bit words.
        for name, width in _BLOCK_SIZES.items():
            with self.subTest(hash=name):
                self.assertGreater(width, 0)
                self.assertEqual(width % 8, 0)

    def test_no_entry_is_keyed_by_a_host_row_name(self) -> None:
        # `block_size` strips a `Host` prefix before looking up, so an entry
        # spelled `HostSha256` would be dead and its device row would miss.
        for name in _BLOCK_SIZES:
            with self.subTest(hash=name):
                self.assertFalse(name.startswith("Host"))


if __name__ == "__main__":
    absltest.main()

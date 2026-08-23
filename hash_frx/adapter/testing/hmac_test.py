# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""HMAC over the ByteHash seam — pinned to the published RFC 4231 vectors and,
across shapes, byte-matched against the universal reference (the stdlib
`hmac`/`hashlib` pair, named by no consumer).

The RFC rows cover both arms of the FIPS 198-1 key processing: keys shorter
than, equal to, and longer than the block (the long-key rows hash down first),
plus the RFC's truncated-output case. The shape sweep covers the batch axis the
seam promises: shared `[K]` keys broadcasting over a batch and per-message
`[B, K]` keys.
"""

from __future__ import annotations

import hashlib
import hmac as stdlib_hmac

import frx
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.adapter.hmac import Hmac
from hash_frx.sha256 import HostSha256, Sha256

# RFC 4231 §4 HMAC-SHA-256 rows. `trunc` is the RFC's own output truncation
# (case 5 keeps 128 bits); None keeps the full digest.
_RFC4231 = (
    (
        "case1",
        b"\x0b" * 20,
        b"Hi There",
        None,
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
    ),
    (
        "case2",
        b"Jefe",
        b"what do ya want for nothing?",
        None,
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
    ),
    (
        "case3",
        b"\xaa" * 20,
        b"\xdd" * 50,
        None,
        "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
    ),
    (
        "case4",
        bytes(range(0x01, 0x1A)),
        b"\xcd" * 50,
        None,
        "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b",
    ),
    (
        "case5",
        b"\x0c" * 20,
        b"Test With Truncation",
        16,
        "a3b6167473100ee06e0c796c2955552b",
    ),
    (
        "case6",
        b"\xaa" * 131,
        b"Test Using Larger Than Block-Size Key - Hash Key First",
        None,
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
    ),
    (
        "case7",
        b"\xaa" * 131,
        b"This is a test using a larger than block-size key and a larger than "
        b"block-size data. The key needs to be hashed before being used by the "
        b"HMAC algorithm.",
        None,
        "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2",
    ),
)

# Both seam substrates must produce identical bytes; the host row is also what
# keeps the eager path honest.
_ROWS = (("device", Sha256), ("host", HostSha256))


def _hmac_sha256(row: type) -> Hmac:
    return Hmac(row(), block_size=64)


class HmacTest(parameterized.TestCase):
    @parameterized.parameters(
        *(
            (f"{r}_{c}", row, key, data, trunc, hexmac)
            for (r, row) in _ROWS
            for (c, key, data, trunc, hexmac) in _RFC4231
        )
    )
    def test_rfc4231_vectors(
        self,
        _name: str,
        row: type,
        key: bytes,
        data: bytes,
        trunc: int | None,
        hexmac: str,
    ) -> None:
        mac = _hmac_sha256(row)
        got = bytes(
            np.asarray(
                mac.mac(
                    np.frombuffer(key, dtype=np.uint8),
                    np.frombuffer(data, dtype=np.uint8)[None, :],
                )
            )[0]
        )
        if trunc is not None:
            got = got[:trunc]
        self.assertEqual(got.hex(), hexmac)

    @parameterized.parameters(0, 1, 32, 63, 64, 65, 131)
    def test_matches_stdlib_across_key_lengths(self, key_len: int) -> None:
        # Sweeps the key across both block-size boundaries (63/64/65) plus the
        # empty key; the message length crosses its own block boundary too.
        rng = np.random.default_rng(key_len)
        key = rng.integers(0, 256, size=key_len, dtype=np.uint8)
        msg = rng.integers(0, 256, size=(1, 120), dtype=np.uint8)
        got = bytes(np.asarray(_hmac_sha256(Sha256).mac(key, msg))[0])
        want = stdlib_hmac.new(bytes(key), bytes(msg[0]), hashlib.sha256).digest()
        self.assertEqual(got, want)

    def test_shared_key_broadcasts_over_batch(self) -> None:
        rng = np.random.default_rng(0)
        key = rng.integers(0, 256, size=32, dtype=np.uint8)
        msgs = rng.integers(0, 256, size=(5, 40), dtype=np.uint8)
        got = np.asarray(_hmac_sha256(Sha256).mac(key, msgs))
        for i in range(msgs.shape[0]):
            want = stdlib_hmac.new(bytes(key), bytes(msgs[i]), hashlib.sha256)
            self.assertEqual(bytes(got[i]), want.digest())

    def test_per_message_keys(self) -> None:
        rng = np.random.default_rng(1)
        keys = rng.integers(0, 256, size=(5, 48), dtype=np.uint8)
        msgs = rng.integers(0, 256, size=(5, 40), dtype=np.uint8)
        got = np.asarray(_hmac_sha256(Sha256).mac(keys, msgs))
        for i in range(5):
            want = stdlib_hmac.new(bytes(keys[i]), bytes(msgs[i]), hashlib.sha256)
            self.assertEqual(bytes(got[i]), want.digest())

    def test_empty_message(self) -> None:
        key = np.arange(16, dtype=np.uint8)
        got = bytes(
            np.asarray(_hmac_sha256(Sha256).mac(key, np.zeros((1, 0), np.uint8)))[0]
        )
        self.assertEqual(got, stdlib_hmac.new(bytes(key), b"", hashlib.sha256).digest())

    def test_traced_matches_eager(self) -> None:
        # The device row accepts tracers, so a consumer can MAC inside its own
        # `@jit`; the traced bytes must equal the eager ones.
        rng = np.random.default_rng(2)
        key = rng.integers(0, 256, size=(3, 32), dtype=np.uint8)
        msg = rng.integers(0, 256, size=(3, 40), dtype=np.uint8)
        mac = _hmac_sha256(Sha256)
        eager = np.asarray(mac.mac(key, msg))
        traced = np.asarray(frx.jit(mac.mac)(key, msg))
        np.testing.assert_array_equal(traced, eager)

    def test_lowers_to_its_digest_calls_alone(self) -> None:
        # The construction adds no marker of its own: a short-key mac is the
        # inner and outer digest markers (2 composites), and a long key adds
        # exactly the hash-down digest (3) — nothing else.
        mac = _hmac_sha256(Sha256)
        short = np.zeros((1, 32), np.uint8)
        long_ = np.zeros((1, 65), np.uint8)
        msg = np.zeros((1, 40), np.uint8)
        self.assertEqual(
            frx.jit(mac.mac).lower(short, msg).as_text().count("stablehlo.composite"),
            2,
        )
        self.assertEqual(
            frx.jit(mac.mac).lower(long_, msg).as_text().count("stablehlo.composite"),
            3,
        )

    def test_value_equality_over_full_parameter_surface(self) -> None:
        # Aux-safety: fresh same-parameter instances must compare (and hash)
        # equal, and any differing parameter must split them.
        self.assertEqual(Hmac(Sha256(), 64), Hmac(Sha256(), 64))
        self.assertEqual(hash(Hmac(Sha256(), 64)), hash(Hmac(Sha256(), 64)))
        self.assertNotEqual(Hmac(Sha256(), 64), Hmac(Sha256(), 128))
        self.assertNotEqual(Hmac(Sha256(), 64), Hmac(HostSha256(), 64))

    def test_the_block_size_defaults_through_the_table(self) -> None:
        # What makes `adapter/block_size.py` an adapter rather than a data
        # sheet: the caller does not have to know FIPS 198-1's B for a hash
        # that has one.
        self.assertEqual(Hmac(Sha256()).block_size, 64)
        self.assertEqual(Hmac(Sha256()), Hmac(Sha256(), 64))

    def test_a_hash_with_no_registered_block_is_refused(self) -> None:
        # The absence carries an argument, so it has to reach the caller rather
        # than being papered over with a guess.
        from hash_frx.blake3.rows import Blake3

        with self.assertRaisesRegex(LookupError, "keyed mode is native"):
            Hmac(Blake3())

    def test_block_size_below_digest_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Hmac(Sha256(), block_size=16)

    def test_unbatched_message_rejected(self) -> None:
        # Matched by regex against the SEAM's message, not a spelling of its
        # own: this front door routes through `byte_hash.device_message` like
        # the other nine, and a divergent string is how that silently stops
        # being true. `_require_batch_rank` states the invariant.
        with self.assertRaisesRegex(ValueError, r"msg must be 2-D uint8 \[B, L\]"):
            _hmac_sha256(Sha256).mac(np.zeros(16, np.uint8), np.zeros(40, np.uint8))

    def test_a_wrong_rank_is_rejected_before_any_conversion(self) -> None:
        # The reason the shared helper checks first: a wrong rank must not reach
        # a device. A 3-D message would broadcast-fail somewhere inside the
        # digest if it got that far, naming an intermediate the caller never
        # wrote.
        with self.assertRaisesRegex(ValueError, r"got ndim=3"):
            _hmac_sha256(Sha256).mac(
                np.zeros(16, np.uint8), np.zeros((2, 3, 40), np.uint8)
            )


if __name__ == "__main__":
    absltest.main()

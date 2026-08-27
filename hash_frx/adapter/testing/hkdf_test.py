# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""HKDF — pinned to the published RFC 5869 Appendix A vectors (the three
SHA-256 cases) on both seam substrates, with the batch axis checked against a
stdlib-built reference.

A.1 is the basic case, A.2 drives a multi-block expand with maximal-length
salt/info, A.3 exercises the RFC's defaults (empty salt and info — the zero
salt is the §2.2 stated default, checked equal to passing None).
"""

from __future__ import annotations

import hashlib
import hmac as stdlib_hmac

import frx
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

from hash_frx.adapter.hkdf import hkdf_expand, hkdf_extract
from hash_frx.adapter.hmac import Hmac
from hash_frx.sha256.sha256 import Sha256

# RFC 5869 Appendix A, the HMAC-SHA-256 cases: (ikm, salt, info, L, PRK, OKM).
_RFC5869 = (
    (
        "a1",
        b"\x0b" * 22,
        bytes(range(0x00, 0x0D)),
        bytes(range(0xF0, 0xFA)),
        42,
        "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5",
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865",
    ),
    (
        "a2",
        bytes(range(0x00, 0x50)),
        bytes(range(0x60, 0xB0)),
        bytes(range(0xB0, 0x100)),
        82,
        "06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244",
        "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c"
        "59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71"
        "cc30c58179ec3e87c14c01d5c1f3434f1d87",
    ),
    (
        "a3",
        b"\x0b" * 22,
        b"",
        b"",
        42,
        "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04",
        "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d"
        "9d201395faa4b61a96c8",
    ),
)


def _hmac_sha256() -> Hmac:
    return Hmac(Sha256(), block_size=64)


def _reference_okm(prk: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 §2.3 spelled with the stdlib pair, per batch entry."""
    okm, t, i = b"", b"", 1
    while len(okm) < length:
        t = stdlib_hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]


def _bytes_arr(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8)


class HkdfTest(parameterized.TestCase):
    @parameterized.parameters(*_RFC5869)
    def test_rfc5869_vectors(
        self,
        _name: str,
        ikm: bytes,
        salt: bytes,
        info: bytes,
        length: int,
        prk_hex: str,
        okm_hex: str,
    ) -> None:
        mac = _hmac_sha256()
        prk = hkdf_extract(mac, _bytes_arr(salt), _bytes_arr(ikm)[None, :])
        self.assertEqual(bytes(np.asarray(prk)[0]).hex(), prk_hex)
        okm = hkdf_expand(mac, prk, _bytes_arr(info), length)
        self.assertEqual(bytes(np.asarray(okm)[0]).hex(), okm_hex)

    def test_none_salt_is_the_zero_salt(self) -> None:
        mac = _hmac_sha256()
        ikm = np.arange(22, dtype=np.uint8)[None, :]
        explicit = np.asarray(hkdf_extract(mac, np.zeros(32, np.uint8), ikm))
        defaulted = np.asarray(hkdf_extract(mac, None, ikm))
        np.testing.assert_array_equal(defaulted, explicit)

    def test_batched_ikm_and_per_entry_info(self) -> None:
        # The batch axis rides every stage: per-entry IKM through extract, a
        # shared PRK-per-entry through a per-entry info expand.
        rng = np.random.default_rng(0)
        ikm = rng.integers(0, 256, size=(4, 22), dtype=np.uint8)
        salt = rng.integers(0, 256, size=13, dtype=np.uint8)
        info = rng.integers(0, 256, size=(4, 10), dtype=np.uint8)
        length = 42
        mac = _hmac_sha256()
        prk = hkdf_extract(mac, salt, ikm)
        okm = np.asarray(hkdf_expand(mac, prk, info, length))
        for i in range(4):
            want_prk = stdlib_hmac.new(
                bytes(salt), bytes(ikm[i]), hashlib.sha256
            ).digest()
            self.assertEqual(bytes(np.asarray(prk)[i]), want_prk)
            self.assertEqual(
                bytes(okm[i]), _reference_okm(want_prk, bytes(info[i]), length)
            )

    def test_shared_prk_broadcasts_over_batched_info(self) -> None:
        rng = np.random.default_rng(1)
        prk = rng.integers(0, 256, size=32, dtype=np.uint8)
        info = rng.integers(0, 256, size=(3, 7), dtype=np.uint8)
        okm = np.asarray(hkdf_expand(_hmac_sha256(), prk, info, 16))
        for i in range(3):
            self.assertEqual(
                bytes(okm[i]), _reference_okm(bytes(prk), bytes(info[i]), 16)
            )

    def test_single_block_length(self) -> None:
        prk = np.arange(32, dtype=np.uint8)
        okm = np.asarray(hkdf_expand(_hmac_sha256(), prk, None, 16))
        self.assertEqual(bytes(okm[0]), _reference_okm(bytes(prk), b"", 16))

    @parameterized.parameters(0, 255 * 32 + 1)
    def test_length_bounds_rejected(self, length: int) -> None:
        with self.assertRaises(ValueError):
            hkdf_expand(_hmac_sha256(), np.zeros(32, np.uint8), None, length)

    def test_extract_and_expand_return_device_arrays(self) -> None:
        # Both front doors inherit the seam's `Array` law through `Hmac.mac`,
        # and a PRK that came back host-side would be caught here rather than
        # wherever it was next concatenated. Neither is a byte hash, so
        # `row_conformance_test`'s registry sweep does not reach them.
        mac = _hmac_sha256()
        prk = hkdf_extract(mac, np.zeros(13, np.uint8), np.zeros((2, 22), np.uint8))
        self.assertIsInstance(prk, Array)
        self.assertIsInstance(hkdf_expand(mac, prk, None, 42), Array)

    def test_traced_matches_eager(self) -> None:
        rng = np.random.default_rng(2)
        ikm = rng.integers(0, 256, size=(3, 22), dtype=np.uint8)
        salt = rng.integers(0, 256, size=13, dtype=np.uint8)
        info = rng.integers(0, 256, size=10, dtype=np.uint8)
        mac = _hmac_sha256()

        def kdf(salt_: np.ndarray, ikm_: np.ndarray, info_: np.ndarray) -> object:
            return hkdf_expand(mac, hkdf_extract(mac, salt_, ikm_), info_, 42)

        eager = np.asarray(kdf(salt, ikm, info))
        traced = np.asarray(frx.jit(kdf)(salt, ikm, info))
        np.testing.assert_array_equal(traced, eager)


if __name__ == "__main__":
    absltest.main()

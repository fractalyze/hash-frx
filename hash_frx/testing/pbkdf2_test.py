# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""PBKDF2 over the seam's HMAC — the published vectors, the
`hashlib.pbkdf2_hmac` differential, the fast-path/generic-path agreement,
and the loop-outside-marker lowering pin.

Values are held to RFC 7914 §11's PBKDF2-HMAC-SHA256 vector and BIP-39's
published mnemonic→seed vector (the c = 2048 HMAC-SHA512 consumer this
construction was scheduled for), then differentially to `hashlib` across
iteration counts, derived-key lengths (truncation and multi-block T alike),
key-processing arms, and batch layouts. The midstate fast path and the
seam-generic body must agree byte for byte — the fast path is patched away
to prove it — and a hash with no profile (BLAKE2s) is held to a plain-Python
reference over `hmac.new`.

The lowering pin is the construction's one structural promise: the c-chain
is a traced loop, so the lowered module's composite count is a small
constant independent of c (nothing unrolls), with the loop visible as a
`stablehlo.while`.
"""

from __future__ import annotations

import hashlib
import hmac as py_hmac
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx import pbkdf2 as pbkdf2_mod
from hash_frx.blake2s import Blake2s
from hash_frx.hmac import Hmac
from hash_frx.pbkdf2 import pbkdf2
from hash_frx.sha256 import HostSha256, Sha256
from hash_frx.sha512 import Sha512

_SHA256 = Hmac(Sha256(), 64)
_SHA512 = Hmac(Sha512(), 128)


def _rows(*items: bytes) -> np.ndarray:
    """Equal-length byte strings as a uint8 [B, L] batch."""
    assert len({len(i) for i in items}) == 1
    return np.stack([np.frombuffer(i, dtype=np.uint8) for i in items])


class Pbkdf2VectorTest(absltest.TestCase):
    def test_rfc_7914_hmac_sha256_vector(self) -> None:
        # RFC 7914 §11's first PBKDF2-HMAC-SHA256 test vector (published for
        # scrypt's inner KDF; the standard record for the SHA-256 profile).
        got = np.asarray(pbkdf2(_SHA256, _rows(b"passwd"), _rows(b"salt"), 1, 64))
        self.assertEqual(
            bytes(got[0]).hex(),
            "55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc"
            "49ca9cccf179b645991664b39d77ef317c71b845b1e30bd509112041d3a19783",
        )

    def test_bip39_mnemonic_to_seed(self) -> None:
        # The consumer-shaped end-to-end vector: BIP-39's published TREZOR
        # row — seed = PBKDF2-HMAC-SHA512(mnemonic, "mnemonic" ‖ passphrase,
        # c = 2048, 64 bytes).
        mnemonic = " ".join(["abandon"] * 11 + ["about"]).encode()
        got = np.asarray(
            pbkdf2(_SHA512, _rows(mnemonic), _rows(b"mnemonic" + b"TREZOR"), 2048, 64)
        )
        self.assertEqual(
            bytes(got[0]).hex(),
            "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e5349553"
            "1f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04",
        )


class Pbkdf2DifferentialTest(parameterized.TestCase):
    @parameterized.product(
        profile=("sha256", "sha512"),
        iterations=(1, 2, 37),
        dk_len=(16, 64, 100),
    )
    def test_matches_hashlib(self, profile: str, iterations: int, dk_len: int) -> None:
        # The free oracle, batched: three independent per-row derivations in
        # lockstep must equal three scalar hashlib runs. dk_len sweeps
        # truncation (16), one exact SHA-512 block (64), and a multi-block T
        # chain (100).
        mac = {"sha256": _SHA256, "sha512": _SHA512}[profile]
        pws = _rows(b"password-a", b"password-b", b"password-c")
        salts = _rows(b"salt-0001", b"salt-0002", b"salt-0003")
        got = np.asarray(pbkdf2(mac, pws, salts, iterations, dk_len))
        self.assertEqual(got.shape, (3, dk_len))
        for i in range(3):
            self.assertEqual(
                bytes(got[i]),
                hashlib.pbkdf2_hmac(
                    profile, bytes(pws[i]), bytes(salts[i]), iterations, dk_len
                ),
            )

    def test_long_password_takes_the_hash_down_arm(self) -> None:
        # A password longer than the block runs FIPS 198-1 §4's K0 = H(K)
        # arm — on the fast path that hashed-down key feeds the midstates.
        pw = bytes(range(200))
        got = np.asarray(pbkdf2(_SHA256, _rows(pw), _rows(b"salt"), 3, 32))
        self.assertEqual(
            bytes(got[0]), hashlib.pbkdf2_hmac("sha256", pw, b"salt", 3, 32)
        )

    def test_shared_password_broadcasts_over_salts(self) -> None:
        # One [P] password against [B, S] salts — the hkdf broadcast rule.
        pw = b"one password"
        salts = _rows(b"salt-a", b"salt-b")
        got = np.asarray(
            pbkdf2(_SHA256, np.frombuffer(pw, dtype=np.uint8), salts, 5, 32)
        )
        for i in range(2):
            self.assertEqual(
                bytes(got[i]),
                hashlib.pbkdf2_hmac("sha256", pw, bytes(salts[i]), 5, 32),
            )

    def test_generic_body_matches_the_fast_path(self) -> None:
        # The two in-loop bodies must be one construction: patch the profile
        # table empty so the same derivation runs the seam-generic HMAC body,
        # and hold it to the fast path's bytes.
        args = (_rows(b"same-password"), _rows(b"same-salt"), 16, 48)
        fast = np.asarray(pbkdf2(_SHA256, *args))
        with mock.patch.dict(pbkdf2_mod._MIDSTATE_PROFILES, clear=True):
            generic = np.asarray(pbkdf2(_SHA256, *args))
        np.testing.assert_array_equal(fast, generic)

    def test_a_hash_without_a_profile_runs_generic(self) -> None:
        # BLAKE2s has no midstate profile, so the loop body is the seam
        # HMAC; held to a plain-Python PBKDF2 over `hmac.new` (hashlib's
        # pbkdf2_hmac only speaks openssl digest names).
        def py_pbkdf2(pw: bytes, salt: bytes, c: int, dk: int) -> bytes:
            def prf(m: bytes) -> bytes:
                return py_hmac.new(pw, m, hashlib.blake2s).digest()

            out = b""
            i = 1
            while len(out) < dk:
                u = prf(salt + i.to_bytes(4, "big"))
                t = u
                for _ in range(c - 1):
                    u = prf(u)
                    t = bytes(a ^ b for a, b in zip(t, u))
                out += t
                i += 1
            return out[:dk]

        mac = Hmac(Blake2s(), 64)
        got = np.asarray(pbkdf2(mac, _rows(b"pw"), _rows(b"salt"), 8, 40))
        self.assertEqual(bytes(got[0]), py_pbkdf2(b"pw", b"salt", 8, 40))


class Pbkdf2LoweringTest(absltest.TestCase):
    def test_the_chain_is_a_loop_not_an_unroll(self) -> None:
        # The #202 acceptance pin: the c-chain lowers as a while loop with
        # the markers OUTSIDE the trip count — the composite count is a small
        # constant, IDENTICAL at different c, and the loop is visible. An
        # unrolled chain would scale the count with c.
        pw, salt = _rows(b"pw"), _rows(b"salt")

        def lowered(c: int) -> str:
            fn = lambda p, s: pbkdf2(_SHA256, p, s, c, 32)  # noqa: E731
            return frx.jit(fn).lower(fnp.asarray(pw), fnp.asarray(salt)).as_text()

        low8, low64 = lowered(8), lowered(64)
        self.assertIn("stablehlo.while", low8)
        count8 = low8.count("stablehlo.composite")
        self.assertEqual(count8, low64.count("stablehlo.composite"))
        self.assertLessEqual(count8, 4)  # U1's HMAC digests, nothing c-scaled


class Pbkdf2SurfaceTest(absltest.TestCase):
    def test_host_rows_are_refused(self) -> None:
        # A host row's eager digest cannot thread the traced carry; the
        # error names the right tool instead.
        with self.assertRaisesRegex(ValueError, "hashlib.pbkdf2_hmac"):
            pbkdf2(Hmac(HostSha256(), 64), _rows(b"pw"), _rows(b"salt"), 2, 32)

    def test_out_of_range_parameters_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            pbkdf2(_SHA256, _rows(b"pw"), _rows(b"salt"), 0, 32)
        with self.assertRaises(ValueError):
            pbkdf2(_SHA256, _rows(b"pw"), _rows(b"salt"), 1, 0)

    def test_traced_matches_eager(self) -> None:
        # The construction holds the seam's traced-or-concrete property: a
        # consumer can derive under its own @jit.
        pw, salt = fnp.asarray(_rows(b"pw")), fnp.asarray(_rows(b"salt"))
        fn = lambda p, s: pbkdf2(_SHA256, p, s, 8, 32)  # noqa: E731
        np.testing.assert_array_equal(
            np.asarray(frx.jit(fn)(pw, salt)), np.asarray(fn(pw, salt))
        )


if __name__ == "__main__":
    absltest.main()

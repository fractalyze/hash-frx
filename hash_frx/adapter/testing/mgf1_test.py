# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""MGF1, held to `hashlib` block by block rather than to a second MGF1.

**RFC 8017 publishes no test vectors** — not for MGF1, not for any scheme in
it; Appendix B.2.1 is a definition only, and a search for a citable standalone
MGF1 vector set (Wycheproof, Botan, NIST CAVP) finds none, because MGF1 is
always exercised indirectly through OAEP/PSS. Recorded here so the next person
does not go looking.

So the oracle is the construction's own definition, decomposed. MGF1's output
is by definition

    T_i = H(seed ‖ I2OSP(i, 4))

concatenated and truncated, which means every block can be checked against
`hashlib.sha256(seed + i.to_bytes(4, "big")).digest()` — stdlib, and **no MGF1
implementation on the other side of the comparison**. That is what makes it an
independent check rather than a round-trip: a wrong counter width, a
one-indexed loop, a little-endian I2OSP or a bad truncation each fail it, and
none of them could fail a self-written host MGF1 that shared the same mistake.
`docs/reference/conventions.md` is explicit that two implementations of one
standard agreeing is worth nothing; this avoids having a second one at all.

Two structural properties ride along, because they catch the same bugs from a
direction the block check cannot: the mask at one length must be a **prefix**
of the mask at any greater length, and the lengths either side of a block
boundary are where an off-by-one lands.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.adapter.mgf1 import Mgf1, mgf1
from hash_frx.byte_hash import ByteHash
from hash_frx.keccak.byte_hashes import Shake256
from hash_frx.sha256.sha256 import HostSha256, Sha256
from hash_frx.sha512.sha512 import Sha512

_SEED = bytes(range(32))
_HLEN = 32


def _u8(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8)


def _seed() -> np.ndarray:
    """`_SEED` as the seam's `[1, S]` batch of one."""
    return _u8(_SEED)[None, :]


def _expected_block(
    index: int,
    hash_fn: Callable[[bytes], Any] = hashlib.sha256,
    h_len: int = _HLEN,
) -> bytes:
    """`T_i` straight from the spec, over `hashlib` — no MGF1 involved.

    `hash_fn` is the stdlib hash matching the row under test, so every case
    reads its expectation from the same place rather than re-spelling the
    counter per hash.
    """
    digest = hash_fn(_SEED + index.to_bytes(4, "big"))
    out = digest.digest(h_len) if hash_fn is hashlib.shake_256 else digest.digest()
    return out[:h_len]


class BlockDecompositionTest(parameterized.TestCase):
    """The load-bearing case: each block against `hashlib`, no second MGF1."""

    @parameterized.parameters(1, 2, 3, 8)
    def test_every_block_is_the_hash_of_seed_and_its_counter(self, blocks: int) -> None:
        got = bytes(np.asarray(mgf1(Sha256(), _seed(), blocks * _HLEN))[0])
        for i in range(blocks):
            with self.subTest(block=i):
                self.assertEqual(
                    got[i * _HLEN : (i + 1) * _HLEN],
                    _expected_block(i),
                    f"block {i} is not H(seed ‖ I2OSP({i}, 4))",
                )

    def test_a_counter_past_one_octet_still_indexes_correctly(self) -> None:
        # Index 256 is the first that needs more than the low octet, so a
        # two-octet counter or a little-endian I2OSP diverges here and nowhere
        # earlier. Reached through the public surface rather than by exporting
        # the counter helper — 257 blocks of SHA-256 is 8224 bytes of mask.
        got = bytes(np.asarray(mgf1(Sha256(), _seed(), 257 * _HLEN))[0])
        self.assertEqual(got[256 * _HLEN : 257 * _HLEN], _expected_block(256))


class LengthTest(parameterized.TestCase):
    @parameterized.parameters(1, 31, 32, 33, 63, 64, 65, 100)
    def test_a_shorter_mask_is_the_prefix_of_a_longer_one(self, length: int) -> None:
        # Falls out of "concatenate the blocks, then truncate", and breaks under
        # any counter or truncation bug — including one wrong only at
        # non-boundary lengths, which the per-block check cannot see. The width
        # rides along: a result of the wrong length cannot equal a `length`-byte
        # slice. Lengths straddle the 32-byte block boundary either side.
        short = bytes(np.asarray(mgf1(Sha256(), _seed(), length))[0])
        long = bytes(np.asarray(mgf1(Sha256(), _seed(), 128))[0])
        self.assertEqual(short, long[:length])
        self.assertLen(short, length)


class OtherHashesTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # SHA-512's blocks are 64 bytes, so the same request spans half as many
        # of them; `Shake256(48)` means hLen = 48 because that is the row's own
        # parameter, not anything MGF1 chooses.
        ("sha512", Sha512(), hashlib.sha512, 64),
        ("shake256_48", Shake256(48), hashlib.shake_256, 48),
    )
    def test_hlen_is_read_off_the_row(
        self, row: ByteHash, hash_fn: Callable[[bytes], Any], h_len: int
    ) -> None:
        got = np.asarray(mgf1(row, _seed(), 2 * h_len))
        self.assertEqual(got.shape, (1, 2 * h_len))
        for i in range(2):
            with self.subTest(block=i):
                self.assertEqual(
                    bytes(got[0][i * h_len : (i + 1) * h_len]),
                    _expected_block(i, hash_fn, h_len),
                )

    def test_a_host_row_agrees_with_its_device_sibling(self) -> None:
        # The host row returns `np.ndarray` and never traces; the bytes are
        # the same hash either way.
        device = np.asarray(mgf1(Sha256(), _seed(), 100))
        host = np.asarray(mgf1(HostSha256(), _seed(), 100))
        np.testing.assert_array_equal(device, host)


class BatchingTest(absltest.TestCase):
    def test_rows_are_independent(self) -> None:
        seeds = np.stack([_u8(bytes([i]) * 32) for i in range(4)])
        got = np.asarray(mgf1(Sha256(), seeds, 70))
        self.assertEqual(got.shape, (4, 70))
        for i in range(4):
            with self.subTest(row=i):
                alone = np.asarray(mgf1(Sha256(), seeds[i][None, :], 70))
                np.testing.assert_array_equal(got[i], alone[0])

    def test_a_bare_seed_is_refused_like_any_bare_message(self) -> None:
        # The seam's rule, which this row now goes through rather than around:
        # a single seed is `B = 1`, not a bare `[S]`. `byte_hash` records why
        # the promotion is not offered — it is the common miss.
        with self.assertRaisesRegex(ValueError, r"2-D uint8 \[B, L\]"):
            mgf1(Sha256(), _u8(_SEED), 48)


class RowContractTest(absltest.TestCase):
    """`Mgf1` is a `ByteHash`, so it owes the seam what every row owes."""

    def test_two_lengths_are_two_hashes(self) -> None:
        # `Row`'s equality is a jit cache key: comparing these equal would
        # serve one length's compiled executable for the other's.
        self.assertEqual(Mgf1(Sha256(), 32), Mgf1(Sha256(), 32))
        self.assertNotEqual(Mgf1(Sha256(), 32), Mgf1(Sha256(), 64))
        self.assertEqual(hash(Mgf1(Sha256(), 32)), hash(Mgf1(Sha256(), 32)))

    def test_two_underlying_hashes_are_two_hashes(self) -> None:
        # Same output width, different `H` — the parameter `_parameters` would
        # drop if it named only the length.
        self.assertNotEqual(Mgf1(Sha256(), 32), Mgf1(Sha512(), 32))

    def test_digest_size_is_the_output_size(self) -> None:
        self.assertEqual(Mgf1(Sha256(), 70).digest_size, 70)

    def test_fusion_path_delegates_to_the_underlying_hash(self) -> None:
        # The mask IS that hash's digests, so the path cannot honestly differ.
        self.assertEqual(Mgf1(Sha256(), 70).fusion_path, Sha256().fusion_path)
        self.assertEqual(Mgf1(HostSha256(), 70).fusion_path, HostSha256().fusion_path)

    def test_it_satisfies_the_ByteHash_protocol(self) -> None:
        # The point of being a row: an `Xof` slot or anything else taking a
        # `ByteHash` accepts it.
        self.assertIsInstance(Mgf1(Sha256(), 32), ByteHash)


class TracerTest(absltest.TestCase):
    def test_the_seed_may_be_a_tracer(self) -> None:
        # Nothing here reads a seed byte — the counter is a host constant and
        # every length is static — so a consumer can call this inside its own
        # jit. That is the property that makes it usable at all.
        def run(seed: frx.Array) -> frx.Array:
            return mgf1(Sha256(), seed, 70)

        got = np.asarray(frx.jit(run)(fnp.asarray(_seed())))
        eager = np.asarray(mgf1(Sha256(), _seed(), 70))
        np.testing.assert_array_equal(got, eager)


class ValidationTest(absltest.TestCase):
    def test_a_zero_length_mask_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, r"must be in \[1, 2\^32"):
            mgf1(Sha256(), _seed(), 0)

    def test_a_mask_past_the_four_octet_bound_is_refused(self) -> None:
        # The upper half of the same guard. Rejected at construction, before
        # anything is allocated — the bound is 2^32 blocks, not a shape.
        with self.assertRaisesRegex(ValueError, r"must be in \[1, 2\^32"):
            Mgf1(Sha256(), (1 << 32) * 32 + 1)

    def test_a_wrong_rank_seed_is_refused_by_the_seam(self) -> None:
        # A `[B, R, S]` seed reaches the seam's own rank check rather than
        # failing as a concatenate error against an intermediate.
        with self.assertRaisesRegex(ValueError, r"2-D uint8 \[B, L\]"):
            mgf1(Sha256(), np.zeros((2, 3, 32), np.uint8), 32)


if __name__ == "__main__":
    absltest.main()

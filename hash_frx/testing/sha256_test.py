# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHA-256 byte-hash — byte-match against the universal reference `hashlib.sha256`.

Agnostic golden: `hashlib` is the FIPS 180-4 reference, named by no consumer. The
lengths exercise every padding boundary — empty, sub-block, the 55/56 one-vs-two
block transition (where the 8-byte length field forces a second block), exact
block multiples, and a multi-block message.
"""

from __future__ import annotations

import functools
import hashlib

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx import sha256
from hash_frx.byte_hash import ByteHash

# Padding-boundary lengths: 0/1 (empty + tiny), 55/56 (the one-block/two-block
# cutoff), 63/64 (block edge), 119/120 (multi-block).
_LENGTHS = (0, 1, 55, 56, 63, 64, 119, 120)


class Sha256Test(parameterized.TestCase):
    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        got = bytes(np.asarray(sha256.digest(msg[None, :]))[0])
        self.assertEqual(got, hashlib.sha256(bytes(msg)).digest())

    def test_batched_equals_per_row(self) -> None:
        # One data-parallel call over a stack of equal-length messages must equal
        # the per-message hashlib digests, in order.
        length = 64
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(7, length), dtype=np.uint8)
        got = np.asarray(sha256.digest(batch))
        for i in range(batch.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.sha256(bytes(batch[i])).digest())

    @parameterized.parameters(*_LENGTHS)
    def test_marked_equals_inline(self, length: int) -> None:
        # The zorch.sha256 marker only tags the region; with no dedicated emitter
        # wired it inlines its decomposition, so the marked digest must byte-equal
        # the unmarked compression at every padding boundary.
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        blocks = fnp.asarray(sha256._pad(msg[None, :]))
        marked = np.asarray(sha256.sha256_merkle_damgard(sha256.INITIAL_STATE, blocks))
        state = fnp.broadcast_to(sha256.INITIAL_STATE, (1, 8))
        inline = np.asarray(sha256.serialize_digest(sha256.compress(state, blocks)))
        np.testing.assert_array_equal(marked, inline)

    def test_emits_single_composite_marker(self) -> None:
        # digest lowers to exactly one stablehlo.composite, name-routed to the
        # dedicated zorch.sha256 emitter (parallel to zorch.poseidon2).
        blocks = fnp.asarray(sha256._pad(np.arange(64, dtype=np.uint8)[None, :]))
        fn = functools.partial(sha256.sha256_merkle_damgard, sha256.INITIAL_STATE)
        txt = frx.jit(fn).lower(blocks).as_text()
        self.assertIn(sha256.SHA256_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_serialize_deserialize_roundtrip(self) -> None:
        # deserialize_digest inverts serialize_digest, so unpacking a digest
        # recovers the exact midstate a stream resumes from.
        rng = np.random.default_rng(0)
        state = fnp.asarray(rng.integers(0, 2**32, (3, 8), np.int64).astype(np.uint32))
        back = sha256.deserialize_digest(sha256.serialize_digest(state))
        np.testing.assert_array_equal(np.asarray(back), np.asarray(state))

    @parameterized.parameters(1, 2, 3)
    def test_merkle_damgard_resumes_from_midstate(self, split: int) -> None:
        # From a non-IV midstate the compression resumes: hashing a 4-block
        # message in two chained halves must equal one pass over all 4.
        blocks = fnp.asarray(
            np.random.default_rng(split)
            .integers(0, 2**32, (1, 4, 16), np.int64)
            .astype(np.uint32)
        )
        whole = sha256.sha256_merkle_damgard(sha256.INITIAL_STATE, blocks)
        mid = sha256.deserialize_digest(
            sha256.sha256_merkle_damgard(sha256.INITIAL_STATE, blocks[:, :split])
        )[0]
        resumed = sha256.sha256_merkle_damgard(mid, blocks[:, split:])
        np.testing.assert_array_equal(np.asarray(whole), np.asarray(resumed))

    def test_compress_explicit_k_matches_default(self) -> None:
        # Threading the round-constant table as an explicit `k` operand (what the
        # marked region does) matches the module-default `_Kd`.
        blocks = fnp.asarray(sha256._pad(np.arange(80, dtype=np.uint8)[None, :]))
        state = fnp.broadcast_to(sha256.INITIAL_STATE, (1, 8))
        default = sha256.compress(state, blocks)
        explicit = sha256.compress(state, blocks, sha256._Kd)
        np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))


class Sha256ByteHashTest(parameterized.TestCase):
    """The two `ByteHash` implementations, against the seam and against each other.

    `byte_hash_test.py` stays seam-only — a double, so it runs on a branch where
    no concrete hash exists — which leaves the real classes untested by it. This
    is the other half of that split: the assertions that need `Sha256` and
    `HostSha256` themselves and are tautologies against a double.
    """

    def test_impls_satisfy_the_seam(self) -> None:
        for h in (sha256.Sha256(), sha256.HostSha256()):
            with self.subTest(impl=type(h).__name__):
                self.assertIsInstance(h, ByteHash)
                self.assertEqual(h.digest_size, 32)
                self.assertIsInstance(h.has_dedicated_fusion, bool)

    def test_fusion_flag_pins_the_substrate(self) -> None:
        # The only assertion in the repo that says WHICH implementation is the
        # device one. A consumer reads this at construction time to branch — see
        # zorch's byte_transcript, whose grind window is a wide nonce sweep on a
        # fused hash and a single nonce otherwise — so a silent flip here would
        # change a proof-of-work strategy rather than fail a hash.
        self.assertTrue(sha256.Sha256().has_dedicated_fusion)
        self.assertFalse(sha256.HostSha256().has_dedicated_fusion)

    @parameterized.parameters(*_LENGTHS)
    def test_host_matches_hashlib(self, length: int) -> None:
        # HostSha256 is a separate implementation, not a wrapper over the marked
        # path: it loops `hashlib` per row. Nothing above covers it.
        rng = np.random.default_rng(length)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        got = np.asarray(sha256.HostSha256().digest(msgs))
        self.assertEqual(got.shape, (4, 32))
        for i in range(msgs.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.sha256(bytes(msgs[i])).digest())

    @parameterized.parameters(*_LENGTHS)
    def test_device_and_host_agree(self, length: int) -> None:
        # Two implementations of identical bytes are only safe while they stay
        # identical; this is the guard that keeps them from drifting, across every
        # padding boundary rather than one convenient length.
        rng = np.random.default_rng(length + 1)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(sha256.Sha256().digest(msgs)),
            np.asarray(sha256.HostSha256().digest(msgs)),
        )

    def test_value_identity_is_by_type(self) -> None:
        # Param-free, so every instance of a type is equal and hashes alike — what
        # keeps the seam re-trace-safe as pytree aux. The two are never equal, or
        # swapping substrate would not re-trace.
        for cls in (sha256.Sha256, sha256.HostSha256):
            with self.subTest(impl=cls.__name__):
                self.assertEqual(cls(), cls())
                self.assertEqual(hash(cls()), hash(cls()))
        self.assertNotEqual(sha256.Sha256(), sha256.HostSha256())


if __name__ == "__main__":
    absltest.main()

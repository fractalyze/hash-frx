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
from hash_frx.fusion import FusionPath
from hash_frx.testing.marker_recognized import assert_marker_recognized

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
        # The hash_frx.sha256 marker only tags the region; with no dedicated emitter
        # wired it inlines its decomposition, so the marked digest must byte-equal
        # the unmarked compression at every padding boundary.
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        blocks = sha256._padded_words(fnp.asarray(msg[None, :]))
        marked = np.asarray(sha256.sha256_merkle_damgard(sha256.INITIAL_STATE, blocks))
        state = fnp.broadcast_to(sha256.INITIAL_STATE, (1, 8))
        inline = np.asarray(sha256.serialize_digest(sha256.compress(state, blocks)))
        np.testing.assert_array_equal(marked, inline)

    def test_emits_single_composite_marker(self) -> None:
        # digest lowers to exactly one stablehlo.composite, name-routed to the
        # dedicated hash_frx.sha256 emitter (parallel to hash_frx.poseidon2).
        blocks = sha256._padded_words(
            fnp.asarray(np.arange(64, dtype=np.uint8))[None, :]
        )
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
        blocks = sha256._padded_words(
            fnp.asarray(np.arange(80, dtype=np.uint8))[None, :]
        )
        state = fnp.broadcast_to(sha256.INITIAL_STATE, (1, 8))
        default = sha256.compress(state, blocks)
        explicit = sha256.compress(state, blocks, sha256._Kd)
        np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))


class Sha256TracedTest(parameterized.TestCase):
    """`digest` inside a traced region, at every padding boundary.

    The point of the padding being a function of the length: a consumer can hash
    inside its own `@jit` without reaching past the seam for the compression, so
    a scheme built on `ByteHash` does not have to name SHA-256 to be traceable.
    Byte-equality with the eager path is the whole claim — a traced padding that
    is subtly different produces a self-consistent wrong hash.
    """

    @parameterized.parameters(*_LENGTHS)
    def test_jit_matches_eager(self, length: int) -> None:
        msg = (np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A))[None, :]
        np.testing.assert_array_equal(
            np.asarray(frx.jit(sha256.digest)(msg)), np.asarray(sha256.digest(msg))
        )

    @parameterized.parameters(*_LENGTHS)
    def test_jit_matches_hashlib(self, length: int) -> None:
        # Against the reference rather than against ourselves: the eager path
        # agreeing with a traced one that shares its bug would prove nothing.
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0xA5)
        got = np.asarray(frx.jit(sha256.digest)(msg[None, :]))[0]
        self.assertEqual(bytes(got), hashlib.sha256(bytes(msg)).digest())

    def test_vmap_matches_the_batch_axis(self) -> None:
        # `digest` already takes the batch, so mapping a one-row call over a stack
        # must reproduce it — which is what a consumer gets for free when its own
        # vmap encloses the hash.
        batch = np.random.default_rng(3).integers(0, 256, (5, 70), dtype=np.uint8)
        mapped = frx.vmap(lambda row: sha256.digest(row[None, :])[0])(batch)
        np.testing.assert_array_equal(
            np.asarray(mapped), np.asarray(sha256.digest(batch))
        )

    def test_the_seam_carries_it(self) -> None:
        # Through `ByteHash.digest` rather than the module function: the consumer
        # holds the seam, and that is the call that has to survive the tracer.
        hasher: ByteHash = sha256.Sha256()
        msg = np.random.default_rng(4).integers(0, 256, (3, 100), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(hasher.digest)(msg)), np.asarray(hasher.digest(msg))
        )

    def test_traced_digest_still_emits_one_marker(self) -> None:
        # The marker is what makes a digest one device unit; a traced caller must
        # not lose it by taking a different path into the compression.
        msg = np.zeros((1, 64), dtype=np.uint8)
        txt = frx.jit(sha256.digest).lower(msg).as_text()
        self.assertIn(sha256.SHA256_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)


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
                self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_flag_pins_the_substrate(self) -> None:
        # The only assertion in the repo that says WHICH implementation is the
        # device one. A consumer reads this at construction time to branch — see
        # zorch's byte_transcript, whose grind window is a wide nonce sweep on a
        # fused hash and a single nonce otherwise — so a silent flip here would
        # change a proof-of-work strategy rather than fail a hash.
        self.assertIs(sha256.Sha256().fusion_path.is_one_kernel, True)
        self.assertIs(sha256.HostSha256().fusion_path, FusionPath.HOST)

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

    def test_marker_is_recognized_by_the_pinned_toolchain(self) -> None:
        blocks = sha256._padded_words(
            fnp.asarray(np.arange(64, dtype=np.uint8))[None, :]
        )
        fn = functools.partial(sha256.sha256_merkle_damgard, sha256.INITIAL_STATE)
        assert_marker_recognized(self, "sha256", fn, blocks)


class Sha256BytesMarkerTest(parameterized.TestCase):
    """The raw-bytes whole-message marker (`sha256_bytes`).

    No pinned recognizer claims the name, so every execution here runs the
    inlined decomposition — which is exactly the contract under test: right
    bytes on the fallback path, one composite on the wire, and `digest`
    holding the blocks route while `_BYTES_EMITTER_AVAILABLE` says no
    recognizer exists.
    """

    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x3C)
        got = bytes(np.asarray(sha256.sha256_bytes(fnp.asarray(msg[None, :])))[0])
        self.assertEqual(got, hashlib.sha256(bytes(msg)).digest())

    @parameterized.parameters(*_LENGTHS)
    def test_matches_the_blocks_marker(self, length: int) -> None:
        # Two wire forms of one digest: the bytes marker must byte-equal the
        # blocks marker (and so, transitively, every consumer's goldens).
        msgs = np.random.default_rng(length).integers(
            0, 256, size=(4, length), dtype=np.uint8
        )
        via_blocks = sha256.sha256_merkle_damgard(
            sha256.INITIAL_STATE, sha256._padded_words(fnp.asarray(msgs))
        )
        np.testing.assert_array_equal(
            np.asarray(sha256.sha256_bytes(fnp.asarray(msgs))),
            np.asarray(via_blocks),
        )

    def test_emits_one_composite_with_the_bytes_name(self) -> None:
        msg = np.zeros((2, 100), dtype=np.uint8)
        txt = frx.jit(sha256.sha256_bytes).lower(fnp.asarray(msg)).as_text()
        self.assertIn(sha256.SHA256_BYTES_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_jit_matches_eager(self) -> None:
        msg = np.random.default_rng(9).integers(0, 256, (3, 120), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(sha256.sha256_bytes)(fnp.asarray(msg))),
            np.asarray(sha256.sha256_bytes(fnp.asarray(msg))),
        )

    def test_digest_routes_by_the_recognizer_flag(self) -> None:
        # `digest` must not emit a marker the pinned plugin cannot claim: with
        # `_BYTES_EMITTER_AVAILABLE` false the wire carries the blocks marker
        # only. Flipping the flag is what moves the wire — this is the test
        # that makes that flip deliberate.
        msg = np.zeros((1, 64), dtype=np.uint8)
        txt = frx.jit(sha256.digest).lower(msg).as_text()
        if sha256._routes_to_bytes_marker():
            self.assertIn(sha256.SHA256_BYTES_MARKER, txt)
        else:
            self.assertIn(sha256.SHA256_MARKER, txt)
            self.assertNotIn(sha256.SHA256_BYTES_MARKER, txt)


if __name__ == "__main__":
    absltest.main()

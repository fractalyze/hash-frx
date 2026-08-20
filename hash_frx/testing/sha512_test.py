# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHA-512 byte-hash — the FIPS 180-4 worked examples, then byte-match against
the universal reference `hashlib.sha512` (named by no consumer).

The lengths exercise every padding boundary — empty, sub-block, the 111/112
one-vs-two block transition (where the 0x80 byte plus the 16-byte length field
force a second block), exact block multiples, and a multi-block message. The
carry sweep is deliberate and SHA-512-specific: a 64-bit add here is two uint32
half adds joined by a comparison-based carry, and 0xFF-saturated messages are
what drive that carry continuously — a dropped carry passes low-entropy
structured vectors while corrupting these.
"""

from __future__ import annotations

import functools
import hashlib
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx import sha512
from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.sha256 import Sha256
from hash_frx.sha512 import HostSha512, Sha512
from hash_frx.testing.jit_cache import assert_single_trace

# Padding-boundary lengths: 0/1 (empty + tiny), 111/112 (the one-block/two-block
# cutoff — the 0x80 byte + 16-byte length need 17 bytes), 127/128 (block edge;
# 128 gets a whole extra padding block), 129 (one past it), 256 (multi-block).
_LENGTHS = (0, 1, 111, 112, 127, 128, 129, 256)

# FIPS 180-4's SHA-512 worked examples (the NIST example-values series), also
# the vectors every independent transcription publishes: one block ("abc" and
# the empty message) and the two-block 896-bit message.
_MSG_896 = (
    b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
    b"hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
)
_VECTORS = (
    (
        "abc",
        b"abc",
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
        "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
    ),
    (
        "empty",
        b"",
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
        "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
    ),
    (
        "two_block_896_bit",
        _MSG_896,
        "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018"
        "501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909",
    ),
)


class Sha512VectorTest(parameterized.TestCase):
    @parameterized.named_parameters(*_VECTORS)
    def test_matches_the_published_vector(self, msg: bytes, digest_hex: str) -> None:
        # Against the standard's record, not against another implementation in
        # this tree (docs/reference/conventions.md: byte-exactness is the gate).
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = np.asarray(sha512.digest(rows))[0]
        self.assertEqual(bytes(got).hex(), digest_hex)
        # And the reference the differential sweep uses is anchored to the same
        # record, so agreeing with hashlib below means agreeing with FIPS 180-4.
        self.assertEqual(hashlib.sha512(msg).hexdigest(), digest_hex)


class Sha512Test(parameterized.TestCase):
    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        got = bytes(np.asarray(sha512.digest(msg[None, :]))[0])
        self.assertEqual(got, hashlib.sha512(bytes(msg)).digest())

    @parameterized.parameters(1, 7, 64, 111, 112, 128, 256)
    def test_carry_saturated_matches_hashlib(self, length: int) -> None:
        # 0xFF-filled messages: every message word is all-ones, so the 64-round
        # add chains overflow the low half continuously — the deliberate
        # exercise of `_add64`'s comparison-based carry across half pairs.
        msg = np.full(length, 0xFF, dtype=np.uint8)
        got = bytes(np.asarray(sha512.digest(msg[None, :]))[0])
        self.assertEqual(got, hashlib.sha512(bytes(msg)).digest())

    def test_batched_equals_per_row(self) -> None:
        # One data-parallel call over a stack of equal-length messages must
        # equal the per-message hashlib digests, in order.
        length = 128
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(7, length), dtype=np.uint8)
        got = np.asarray(sha512.digest(batch))
        for i in range(batch.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.sha512(bytes(batch[i])).digest())

    @parameterized.parameters(*_LENGTHS)
    def test_marked_equals_inline(self, length: int) -> None:
        # The marker only tags the region; with no dedicated emitter wired it
        # inlines its decomposition, so the marked digest must byte-equal the
        # unmarked compression at every padding boundary.
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        blocks = sha512._padded_words(fnp.asarray(msg[None, :]))
        marked = np.asarray(sha512.sha512_merkle_damgard(sha512.INITIAL_STATE, blocks))
        state = fnp.broadcast_to(sha512.INITIAL_STATE, (1, 16))
        inline = np.asarray(sha512.serialize_digest(sha512.compress(state, blocks)))
        np.testing.assert_array_equal(marked, inline)

    def test_serialize_deserialize_roundtrip(self) -> None:
        # deserialize_digest inverts serialize_digest, so unpacking a digest
        # recovers the exact midstate a stream resumes from.
        rng = np.random.default_rng(0)
        state = fnp.asarray(rng.integers(0, 2**32, (3, 16), np.int64).astype(np.uint32))
        back = sha512.deserialize_digest(sha512.serialize_digest(state))
        np.testing.assert_array_equal(np.asarray(back), np.asarray(state))

    @parameterized.parameters(1, 2, 3)
    def test_merkle_damgard_resumes_from_midstate(self, split: int) -> None:
        # From a non-IV midstate the compression resumes: hashing a 4-block
        # message in two chained halves must equal one pass over all 4.
        blocks = fnp.asarray(
            np.random.default_rng(split)
            .integers(0, 2**32, (1, 4, 32), np.int64)
            .astype(np.uint32)
        )
        whole = sha512.sha512_merkle_damgard(sha512.INITIAL_STATE, blocks)
        mid = sha512.deserialize_digest(
            sha512.sha512_merkle_damgard(sha512.INITIAL_STATE, blocks[:, :split])
        )[0]
        resumed = sha512.sha512_merkle_damgard(mid, blocks[:, split:])
        np.testing.assert_array_equal(np.asarray(whole), np.asarray(resumed))

    def test_compress_explicit_k_matches_default(self) -> None:
        # Threading the round-constant table as an explicit `k` operand (what
        # the marked region does) matches the module-default `_Kd`.
        blocks = sha512._padded_words(
            fnp.asarray(np.arange(160, dtype=np.uint8))[None, :]
        )
        state = fnp.broadcast_to(sha512.INITIAL_STATE, (1, 16))
        default = sha512.compress(state, blocks)
        explicit = sha512.compress(state, blocks, sha512._Kd)
        np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))


def _composite(fn: Any, *args: Any) -> Any:
    """The one composite eqn in `fn`'s jaxpr — read without lowering to MLIR
    (the `grostl_test` helper)."""
    eqns = [
        e
        for e in frx.make_jaxpr(fn)(*args).jaxpr.eqns
        if e.primitive.name == "composite"
    ]
    assert len(eqns) == 1, f"expected one composite, got {len(eqns)}"
    return eqns[0]


class Sha512MarkerTest(absltest.TestCase):
    def test_no_leg_routes_a_sha512_marker_yet(self) -> None:
        # The pre-emitter pin (the vision/grostl arrangement): both module
        # flags say "no emitter", so every unpatched instance reads GENERIC on
        # every backend. When an emitter lands these flip with the frx floor
        # and this case flips to the keccak-style backend gate.
        self.assertFalse(sha512._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(sha512._EMITTER_BACKENDS, ())
        self.assertIs(Sha512().fusion_path, FusionPath.GENERIC)

    def test_digest_emits_one_composite_with_the_digest_name(self) -> None:
        # The contract's unit: an absent or split marker still computes the
        # right bytes, so only the lowered module shows it.
        msg = np.zeros((2, 100), dtype=np.uint8)
        txt = frx.jit(sha512.digest).lower(msg).as_text()
        self.assertIn(sha512.SHA512_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        # The wire surface an emitter will read: name, version, and the
        # documented [h0, k, blocks] operand order in the big-endian pair
        # layout. Three invars exactly is the captured-constants-free property
        # — an array the body closed over would be lifted in AHEAD of these,
        # one per call site (the operand-ABI rule in
        # docs/reference/conventions.md).
        msg = fnp.asarray(np.zeros((2, 100), dtype=np.uint8))
        eqn = _composite(sha512.digest, msg)
        self.assertEqual(eqn.params["name"], sha512.SHA512_MARKER)
        self.assertEqual(eqn.params["version"], sha512.SHA512_MARKER_VERSION)
        self.assertLen(eqn.invars, 3)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        # L = 100: one block once padded (100 + 17 ≤ 128).
        self.assertEqual(shapes, [(16,), (160,), (2, 1, 32)])


class Sha512TracedTest(parameterized.TestCase):
    """`digest` inside a traced region, at every padding boundary.

    The point of the padding being a function of the length: a consumer can
    hash inside its own `@jit` without reaching past the seam for the
    compression, so a scheme built on `ByteHash` does not have to name SHA-512
    to be traceable. Byte-equality with the eager path is the whole claim — a
    traced padding that is subtly different produces a self-consistent wrong
    hash.
    """

    @parameterized.parameters(*_LENGTHS)
    def test_jit_matches_eager(self, length: int) -> None:
        msg = (np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A))[None, :]
        np.testing.assert_array_equal(
            np.asarray(frx.jit(sha512.digest)(msg)), np.asarray(sha512.digest(msg))
        )

    @parameterized.parameters(*_LENGTHS)
    def test_jit_matches_hashlib(self, length: int) -> None:
        # Against the reference rather than against ourselves: the eager path
        # agreeing with a traced one that shares its bug would prove nothing.
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0xA5)
        got = np.asarray(frx.jit(sha512.digest)(msg[None, :]))[0]
        self.assertEqual(bytes(got), hashlib.sha512(bytes(msg)).digest())

    def test_vmap_matches_the_batch_axis(self) -> None:
        # `digest` already takes the batch, so mapping a one-row call over a
        # stack must reproduce it — which is what a consumer gets for free when
        # its own vmap encloses the hash.
        batch = np.random.default_rng(3).integers(0, 256, (5, 130), dtype=np.uint8)
        mapped = frx.vmap(lambda row: sha512.digest(row[None, :])[0])(batch)
        np.testing.assert_array_equal(
            np.asarray(mapped), np.asarray(sha512.digest(batch))
        )

    def test_the_seam_carries_it(self) -> None:
        # Through `ByteHash.digest` rather than the module function: the
        # consumer holds the seam, and that is the call that has to survive the
        # tracer.
        hasher: ByteHash = Sha512()
        msg = np.random.default_rng(4).integers(0, 256, (3, 100), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(hasher.digest)(msg)), np.asarray(hasher.digest(msg))
        )

    def test_traced_digest_still_emits_one_marker(self) -> None:
        # The marker is what makes a digest one device unit; a traced caller
        # must not lose it by taking a different path into the compression.
        msg = np.zeros((1, 128), dtype=np.uint8)
        txt = frx.jit(sha512.digest).lower(msg).as_text()
        self.assertIn(sha512.SHA512_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_digest_reuses_one_trace_across_instances(self) -> None:
        # The zone's cache is keyed on the message aval alone (the digest is
        # parameterless), so freshly built instances must share one trace —
        # the property `sha512_merkle_damgard`'s module-level jit zone exists
        # for. The (3, 100) aval is one `test_the_seam_carries_it` already
        # compiled, so the pin costs no fresh compile.
        msgs = fnp.asarray(
            np.random.default_rng(4).integers(0, 256, (3, 100), dtype=np.uint8)
        )
        calls = [functools.partial(Sha512().digest, msgs) for _ in range(3)]
        assert_single_trace(self, sha512.sha512_merkle_damgard, calls)

    def test_both_sha2_rows_coexist_in_one_jit_region(self) -> None:
        # Issue #66's acceptance shape: FIPS 205 §11.2.2's category-3/5
        # tweakable-hash family keeps SHA-256 for PRF/F and moves SHA-512 into
        # H/T_l, so ONE traced region must carry both hashes side by side —
        # each against its own reference.
        h256, h512 = Sha256(), Sha512()
        msgs = np.random.default_rng(66).integers(0, 256, (3, 100), dtype=np.uint8)

        @frx.jit
        def family(m: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
            return h256.digest(m), h512.digest(m)

        d256, d512 = family(msgs)
        for i in range(msgs.shape[0]):
            row = bytes(msgs[i])
            self.assertEqual(bytes(np.asarray(d256)[i]), hashlib.sha256(row).digest())
            self.assertEqual(bytes(np.asarray(d512)[i]), hashlib.sha512(row).digest())


class Sha512ByteHashTest(parameterized.TestCase):
    """The two `ByteHash` implementations, against the seam and against each
    other.

    `byte_hash_test.py` stays seam-only — a double, so it runs on a branch
    where no concrete hash exists — which leaves the real classes untested by
    it. This is the other half of that split: the assertions that need `Sha512`
    and `HostSha512` themselves and are tautologies against a double.
    """

    def test_impls_satisfy_the_seam(self) -> None:
        for h in (Sha512(), HostSha512()):
            with self.subTest(impl=type(h).__name__):
                self.assertIsInstance(h, ByteHash)
                self.assertEqual(h.digest_size, 64)
                self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_paths_pin_the_substrate(self) -> None:
        # Device GENERIC (pre-emitter, every backend), host HOST (every
        # backend) — and the traceability tie to the return type: the device
        # row returns an `Array` and takes a tracer, the host row reads bytes
        # and never can (`byte_hash.py`'s rule).
        msg = np.zeros((1, 1), dtype=np.uint8)
        device, host = Sha512(), HostSha512()
        self.assertIs(device.fusion_path, FusionPath.GENERIC)
        self.assertTrue(device.fusion_path.is_traceable)
        self.assertNotIsInstance(device.digest(msg), np.ndarray)
        self.assertIs(host.fusion_path, FusionPath.HOST)
        self.assertFalse(host.fusion_path.is_traceable)
        self.assertIsInstance(host.digest(msg), np.ndarray)

    @parameterized.parameters(*_LENGTHS)
    def test_host_matches_hashlib(self, length: int) -> None:
        # HostSha512 is a separate implementation, not a wrapper over the
        # marked path: it loops `hashlib` per row. Nothing above covers it.
        rng = np.random.default_rng(length)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        got = np.asarray(HostSha512().digest(msgs))
        self.assertEqual(got.shape, (4, 64))
        for i in range(msgs.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.sha512(bytes(msgs[i])).digest())

    @parameterized.parameters(*_LENGTHS)
    def test_device_and_host_agree(self, length: int) -> None:
        # Two implementations of identical bytes are only safe while they stay
        # identical; this is the guard that keeps them from drifting, across
        # every padding boundary rather than one convenient length.
        rng = np.random.default_rng(length + 1)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(Sha512().digest(msgs)),
            np.asarray(HostSha512().digest(msgs)),
        )

    def test_value_identity_is_by_type(self) -> None:
        # Param-free, so every instance of a type is equal and hashes alike —
        # what keeps the seam re-trace-safe as pytree aux. The two are never
        # equal, or swapping substrate would not re-trace; and neither equals
        # the SHA-256 sibling, or a family holding both would collide.
        for cls in (Sha512, HostSha512):
            with self.subTest(impl=cls.__name__):
                self.assertEqual(cls(), cls())
                self.assertEqual(hash(cls()), hash(cls()))
        self.assertNotEqual(Sha512(), HostSha512())
        self.assertNotEqual(Sha512(), Sha256())


if __name__ == "__main__":
    absltest.main()

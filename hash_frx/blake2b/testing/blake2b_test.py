# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The device BLAKE2b row — the RFC vector, the hashlib differential, the
marker, and the seam.

Values are held to the published record directly (RFC 7693 Appendix A's
worked "abc" example and the reference-suite empty-message vector) and to
`hashlib.blake2b` differentially across every padding boundary and digest
size — the free oracle this family was scheduled around (issue #161): OpenSSL
shares none of the half-pair lowering, and because the parameter block folds
`digest_size` into the initial state, agreement at a shorter length proves
the length reached the IV rather than a slice.

The lowering assertions are the usual half that values cannot see: the digest
must emit exactly one composite carrying the registered name, version, and
the four-operand ABI with no captured constants — including the zero-length
tail at a block-multiple length, the ABI's one degenerate shape. No backend
routes the name yet, so recognition is not asserted — emission and the ABI
are what an emitter will read, pinned before it exists (the Vision/Grøstl
arrangement).
"""

from __future__ import annotations

import functools
import hashlib
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

from hash_frx.blake2b import blake2b
from hash_frx.blake2b.blake2b import Blake2b
from hash_frx.blake2b.byte_hashes import MAX_DIGEST_SIZE, HostBlake2b
from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.testing.jit_cache import assert_single_trace

# Padding-boundary lengths for the differential sweep: 0 (the dd = 1 all-zero
# block), 1 (tiny), 127/128/129 (the single-block edge — 128 is exactly one
# block with an EMPTY zero tail, 129 spills into a second block and puts the
# interior/final t split live), 255/256 (the two-block edge, 256 again
# tail-less), 384 (a three-block chain).
_LENGTHS = (0, 1, 127, 128, 129, 255, 256, 384)
_SIZES = (1, 20, 32, 48, 64)

# RFC 7693 Appendix A ("abc", BLAKE2b-512) and the empty-message vector from
# the BLAKE2 reference test vectors (github.com/BLAKE2/BLAKE2, testvectors/) —
# the same anchors the host row is held to, here pinning the from-scratch
# arithmetic instead of a wrapper.
_KAT_ABC_512 = bytes.fromhex(
    "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1"
    "7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
)
_KAT_EMPTY_512 = bytes.fromhex(
    "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419"
    "d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"
)


def _message(length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(length * 31 + seed)
    return rng.integers(0, 256, size=(4, length), dtype=np.uint8)


class Blake2bVectorTest(parameterized.TestCase):
    def test_abc_matches_rfc_7693(self) -> None:
        # Against the standard's own worked example, not against another
        # implementation in this tree (docs/reference/conventions.md:
        # byte-exactness is the gate).
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        got = np.asarray(blake2b.digest(rows))[0]
        self.assertEqual(bytes(got), _KAT_ABC_512)

    def test_empty_message_matches_the_published_vector(self) -> None:
        # The dd = 1 special case: no bytes, still one all-zero block with
        # t = 0 and the final flag set (RFC 7693 §3.3).
        got = np.asarray(blake2b.digest(np.zeros((1, 0), dtype=np.uint8)))[0]
        self.assertEqual(bytes(got), _KAT_EMPTY_512)

    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        # The free differential across every padding boundary, batched: one
        # data-parallel device call equals the per-message `hashlib` digests,
        # in order — the bulk-parallel claim and the byte claim at once.
        msgs = _message(length)
        got = np.asarray(blake2b.digest(msgs))
        for i in range(msgs.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.blake2b(bytes(msgs[i])).digest())

    @parameterized.parameters(*_SIZES)
    def test_the_declared_length_reaches_the_initial_state(self, size: int) -> None:
        # Truncating BLAKE2b-512 is the WRONG bytes at every shorter length —
        # the parameter block folds `digest_size` into h[0] — so agreement
        # with `hashlib` at the same length proves the length reached the IV
        # rather than a slice. The (1, 3) aval rides the "abc" vector's
        # compile; the digest-size sweep costs no fresh trace (the h0-operand
        # arrangement, pinned in Blake2bTracedTest).
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        got = np.asarray(blake2b.digest(rows, size))
        self.assertEqual(got.shape, (1, size))
        self.assertEqual(
            bytes(got[0]), hashlib.blake2b(b"abc", digest_size=size).digest()
        )

    @parameterized.parameters(128, 384)
    def test_carry_saturated_matches_hashlib(self, length: int) -> None:
        # All-0xFF messages keep every 64-bit add near the wrap, the directed
        # exercise of `word64.add64`'s comparison-based carry across half
        # pairs (the sha512_test arrangement). The lengths reuse the sweep's
        # (4, L) avals, so no fresh compile.
        msgs = np.full((4, length), 0xFF, dtype=np.uint8)
        got = np.asarray(blake2b.digest(msgs))
        expected = hashlib.blake2b(bytes(msgs[0])).digest()
        for i in range(msgs.shape[0]):
            self.assertEqual(bytes(got[i]), expected)


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


class Blake2bMarkerTest(absltest.TestCase):
    def test_no_leg_routes_a_blake2b_marker_yet(self) -> None:
        # The pre-emitter pin (the Vision/Grøstl arrangement): both module
        # flags say "no emitter", so every unpatched instance reads GENERIC
        # on every backend. When an emitter lands these flip with the frx
        # floor and this case flips to the keccak-style backend gate.
        self.assertFalse(blake2b._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(blake2b._EMITTER_BACKENDS, ())
        for size in _SIZES:
            self.assertIs(Blake2b(size).fusion_path, FusionPath.GENERIC)

    def test_digest_emits_one_composite_with_the_digest_name(self) -> None:
        # The contract's unit: an absent or split marker still computes the
        # right bytes, so only the lowered module shows it.
        msg = np.zeros((2, 100), dtype=np.uint8)
        txt = frx.jit(blake2b.digest).lower(msg).as_text()
        self.assertIn(blake2b.BLAKE2B_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        # The wire surface an emitter will read: name, version, and the
        # documented [h0, iv, msg, tail] operand order. Four invars exactly
        # is the captured-constants-free property — an array the body closed
        # over would be lifted in AHEAD of these, one per call site (the
        # operand-ABI rule in docs/reference/conventions.md); the counter
        # schedule and the SIGMA rows must enter as scalar literals and
        # trace-time tuples, never as lifted table arrays.
        msg = fnp.asarray(np.zeros((2, 100), dtype=np.uint8))
        eqn = _composite(blake2b.digest, msg)
        self.assertEqual(eqn.params["name"], blake2b.BLAKE2B_MARKER)
        self.assertEqual(eqn.params["version"], blake2b.BLAKE2B_MARKER_VERSION)
        self.assertLen(eqn.invars, 4)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        # L = 100: one block once padded, so the zero tail is 28 bytes.
        self.assertEqual(shapes, [(16,), (16,), (2, 100), (28,)])

    def test_a_block_multiple_rides_an_empty_tail(self) -> None:
        # The ABI's degenerate shape: at L a multiple of 128 the zero tail is
        # empty, and the operand still rides — zero-length, never dropped
        # (the invar COUNT is the pinned surface; only shapes move with L).
        msg = fnp.asarray(np.zeros((2, 128), dtype=np.uint8))
        eqn = _composite(blake2b.digest, msg)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        self.assertEqual(shapes, [(16,), (16,), (2, 128), (0,)])


class Blake2bTracedTest(absltest.TestCase):
    """`digest` inside a traced region — the seam property a consumer hashing
    under its own `@jit` depends on, which the length-built zero pad exists to
    keep (`sha256.digest` states the claim). One boundary-crossing length:
    compile time scales with block count, the property does not."""

    def test_jit_matches_eager_and_hashlib(self) -> None:
        msgs = _message(129, seed=7)[:2]  # 129 bytes: the counter crosses
        eager = np.asarray(blake2b.digest(msgs))
        traced = np.asarray(frx.jit(blake2b.digest)(msgs))
        np.testing.assert_array_equal(traced, eager)
        # Against the oracle rather than only against ourselves: eager
        # agreeing with a traced path that shares its bug proves nothing.
        for i in range(msgs.shape[0]):
            self.assertEqual(bytes(traced[i]), hashlib.blake2b(bytes(msgs[i])).digest())

    def test_the_seam_carries_it(self) -> None:
        # Through `ByteHash.digest` rather than the module function — and at
        # a non-default digest size, so the traced path also carries the
        # caller-side truncation.
        hasher: ByteHash = Blake2b(32)
        msgs = _message(129)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(hasher.digest)(msgs)),
            np.asarray(hasher.digest(msgs)),
        )

    def test_one_trace_serves_every_instance_and_digest_size(self) -> None:
        # The digest-size arrangement, pinned: the zone's cache keys on the
        # operand avals, and `h0` is uint32 [16] for EVERY size — the size
        # rides h0's VALUE and the host-side slice — so after the first call
        # freshly built instances of DIFFERENT sizes must all ride the same
        # (4, 129) trace, gaining the zone nothing. Identity-keyed instances
        # or a size-keyed zone would each fail here.
        msgs = fnp.asarray(_message(129))
        calls = [
            functools.partial(Blake2b(size).digest, msgs) for size in (64, 32, 1, 64)
        ]
        assert_single_trace(self, blake2b.blake2b_bytes, calls)


class Blake2bByteHashTest(absltest.TestCase):
    """The device row against the seam, and against its host partner."""

    def test_impls_satisfy_the_seam(self) -> None:
        for h in (Blake2b(), HostBlake2b()):
            with self.subTest(impl=type(h).__name__):
                self.assertIsInstance(h, ByteHash)
                self.assertEqual(h.digest_size, MAX_DIGEST_SIZE)
                self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_paths_pin_the_substrate(self) -> None:
        # Device GENERIC (pre-emitter, every backend), host HOST (every
        # backend) — and the traceability tie to the return type: the device
        # row returns an `Array` and takes a tracer, the host row reads bytes
        # and never can (`byte_hash.py`'s rule). The (1, 3) message rides the
        # "abc" vector's aval, so the plumbing check costs no fresh compile.
        msg = np.zeros((1, 3), dtype=np.uint8)
        device, host = Blake2b(), HostBlake2b()
        self.assertIs(device.fusion_path, FusionPath.GENERIC)
        self.assertTrue(device.fusion_path.is_traceable)
        out = device.digest(msg)
        self.assertNotIsInstance(out, np.ndarray)
        self.assertIsInstance(out, Array)
        self.assertIs(host.fusion_path, FusionPath.HOST)
        self.assertFalse(host.fusion_path.is_traceable)
        self.assertIsInstance(host.digest(msg), np.ndarray)

    def test_device_and_host_agree_across_sizes(self) -> None:
        # The host-vs-device differential at the seam: the two rows must be
        # one hash per digest size. (1, 3) rides the vector aval — the size
        # sweep costs no fresh compile, per the trace pin above.
        msg = _message(3, seed=5)[:1]
        for size in _SIZES:
            with self.subTest(size=size):
                np.testing.assert_array_equal(
                    np.asarray(Blake2b(size).digest(msg)),
                    HostBlake2b(size).digest(msg),
                )

    def test_digest_shape_and_dtype(self) -> None:
        for h in (Blake2b(20), HostBlake2b(20)):
            with self.subTest(impl=type(h).__name__):
                out = np.asarray(h.digest(np.zeros((4, 1), dtype=np.uint8)))
                self.assertEqual(out.shape, (4, 20))
                self.assertEqual(out.dtype, np.uint8)

    def test_value_identity_rides_the_digest_size(self) -> None:
        # By value over the length, like the host row and the SHAKE/BLAKE3
        # rows — the param-free by-type form does not apply. Same size equal
        # (and hash-equal: what keeps the seam re-trace-safe as pytree aux),
        # cross-size unequal, and never equal to the host row: swapping
        # substrate must re-trace.
        self.assertEqual(Blake2b(32), Blake2b(32))
        self.assertEqual(hash(Blake2b(32)), hash(Blake2b(32)))
        self.assertNotEqual(Blake2b(32), Blake2b(64))
        self.assertNotEqual(Blake2b(32), HostBlake2b(32))
        self.assertNotEqual(Blake2b(32), object())

    def test_out_of_range_length_is_refused_at_construction(self) -> None:
        for size in (0, MAX_DIGEST_SIZE + 1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    Blake2b(size)
                with self.assertRaises(ValueError):
                    # The module path re-checks in `_initial_state`: a caller
                    # bypassing the row still gets the range error before any
                    # device work runs.
                    blake2b.digest(np.zeros((1, 1), dtype=np.uint8), size)


if __name__ == "__main__":
    absltest.main()

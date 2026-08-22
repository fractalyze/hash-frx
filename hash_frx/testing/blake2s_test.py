# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The BLAKE2s rows — the RFC vector, the hashlib differential, the marker,
and the seam; `blake2b_test`'s plan at the 32-bit parameters.

Values are held to the published record directly (RFC 7693 Appendix B's
worked "abc" example and the reference-suite empty-message vector) and to
`hashlib.blake2s` differentially across every padding boundary and digest
size — because the parameter block folds `digest_size` into the initial
state, agreement at a shorter length proves the length reached the IV rather
than a slice.

The lowering assertions are the usual half that values cannot see: the digest
must emit exactly one composite carrying the registered name, version, and
the four-operand ABI with no captured constants — including the zero-length
tail at a block-multiple length, the ABI's one degenerate shape. No backend
routes the name yet, so recognition is not asserted — emission and the ABI
are what an emitter will read, pinned before it exists (the Vision/Grøstl
arrangement).

Every marker/traced case reuses an aval the differential sweep compiles, and
`h0`'s aval is uint32 [8] for every digest size, so the suite pays one
compile per (batch, length) and nothing more.
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

from hash_frx import blake2s
from hash_frx.blake2s import MAX_DIGEST_SIZE, Blake2s, HostBlake2s
from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.testing.jit_cache import assert_single_trace

# Padding-boundary lengths for the differential sweep: 0 (the dd = 1 all-zero
# block), 1 (tiny), 63/64/65 (the single-block edge — 64 is exactly one block
# with an EMPTY zero tail, 65 spills into a second block and puts the
# interior/final t split live), 127/128 (the two-block edge, 128 again
# tail-less), 192 (a three-block chain).
_LENGTHS = (0, 1, 63, 64, 65, 127, 128, 192)
_SIZES = (1, 16, 20, 28, 32)

# RFC 7693 Appendix B ("abc", BLAKE2s-256) and the empty-message vector from
# the BLAKE2 reference test vectors (github.com/BLAKE2/BLAKE2, testvectors/).
_KAT_ABC_256 = bytes.fromhex(
    "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982"
)
_KAT_EMPTY_256 = bytes.fromhex(
    "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9"
)


def _message(length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(length * 31 + seed)
    return rng.integers(0, 256, size=(4, length), dtype=np.uint8)


class Blake2sVectorTest(parameterized.TestCase):
    def test_abc_matches_rfc_7693(self) -> None:
        # Against the standard's own worked example, not against another
        # implementation in this tree (docs/reference/conventions.md:
        # byte-exactness is the gate).
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        got = np.asarray(blake2s.digest(rows))[0]
        self.assertEqual(bytes(got), _KAT_ABC_256)

    def test_empty_message_matches_the_published_vector(self) -> None:
        # The dd = 1 special case: no bytes, still one all-zero block with
        # t = 0 and the final flag set (RFC 7693 §3.3).
        got = np.asarray(blake2s.digest(np.zeros((1, 0), dtype=np.uint8)))[0]
        self.assertEqual(bytes(got), _KAT_EMPTY_256)

    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        # The free differential across every padding boundary, batched: one
        # data-parallel device call equals the per-message `hashlib` digests,
        # in order — the bulk-parallel claim and the byte claim at once.
        msgs = _message(length)
        got = np.asarray(blake2s.digest(msgs))
        for i in range(msgs.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.blake2s(bytes(msgs[i])).digest())

    @parameterized.parameters(*_SIZES)
    def test_the_declared_length_reaches_the_initial_state(self, size: int) -> None:
        # Truncating BLAKE2s-256 is the WRONG bytes at every shorter length —
        # the parameter block folds `digest_size` into h[0] — so agreement
        # with `hashlib` at the same length proves the length reached the IV
        # rather than a slice. The (1, 3) aval rides the "abc" vector's
        # compile; the digest-size sweep costs no fresh trace (the h0-operand
        # arrangement, pinned in Blake2sTracedTest).
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        got = np.asarray(blake2s.digest(rows, size))
        self.assertEqual(got.shape, (1, size))
        self.assertEqual(
            bytes(got[0]), hashlib.blake2s(b"abc", digest_size=size).digest()
        )

    @parameterized.parameters(64, 192)
    def test_all_ones_matches_hashlib(self, length: int) -> None:
        # All-0xFF messages keep every word all-ones through the schedule —
        # the low-entropy saturation case the random sweep never hits. (No
        # carry machinery to exercise here, unlike the 64-bit sibling: the
        # adds wrap in the dtype.) The lengths reuse the sweep's (4, L)
        # avals, so no fresh compile.
        msgs = np.full((4, length), 0xFF, dtype=np.uint8)
        got = np.asarray(blake2s.digest(msgs))
        expected = hashlib.blake2s(bytes(msgs[0])).digest()
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


class Blake2sMarkerTest(absltest.TestCase):
    def test_no_leg_routes_a_blake2s_marker_yet(self) -> None:
        # The pre-emitter pin (the Vision/Grøstl arrangement): both module
        # flags say "no emitter", so every unpatched instance reads GENERIC
        # on every backend. When an emitter lands these flip with the frx
        # floor and this case flips to the keccak-style backend gate. One
        # instance suffices: the path reads the two flags alone, and the
        # size sweeps elsewhere already construct every length.
        self.assertFalse(blake2s._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(blake2s._EMITTER_BACKENDS, ())
        self.assertIs(Blake2s().fusion_path, FusionPath.GENERIC)

    def test_digest_emits_one_composite_with_the_digest_name(self) -> None:
        # The contract's unit: an absent or split marker still computes the
        # right bytes, so only the lowered module shows it. (4, 63) is the
        # differential sweep's aval, so the marker cases add no fresh trace.
        msg = np.zeros((4, 63), dtype=np.uint8)
        txt = frx.jit(blake2s.digest).lower(msg).as_text()
        self.assertIn(blake2s.BLAKE2S_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        # The wire surface an emitter will read: name, version, and the
        # documented [h0, iv, msg, tail] operand order. Four invars exactly
        # is the captured-constants-free property — an array the body closed
        # over would be lifted in AHEAD of these, one per call site (the
        # operand-ABI rule in docs/reference/conventions.md); the counter
        # schedule and the SIGMA rows must enter as scalar literals and
        # trace-time tuples, never as lifted table arrays.
        msg = fnp.asarray(np.zeros((4, 63), dtype=np.uint8))
        eqn = _composite(blake2s.digest, msg)
        self.assertEqual(eqn.params["name"], blake2s.BLAKE2S_MARKER)
        self.assertEqual(eqn.params["version"], blake2s.BLAKE2S_MARKER_VERSION)
        self.assertLen(eqn.invars, 4)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        # L = 63: one block once padded, so the zero tail is 1 byte.
        self.assertEqual(shapes, [(8,), (8,), (4, 63), (1,)])

    def test_a_block_multiple_rides_an_empty_tail(self) -> None:
        # The ABI's degenerate shape: at L a multiple of 64 the zero tail is
        # empty, and the operand still rides — zero-length, never dropped
        # (the invar COUNT is the pinned surface; only shapes move with L).
        msg = fnp.asarray(np.zeros((4, 64), dtype=np.uint8))
        eqn = _composite(blake2s.digest, msg)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        self.assertEqual(shapes, [(8,), (8,), (4, 64), (0,)])


class Blake2sTracedTest(absltest.TestCase):
    """`digest` inside a traced region — the seam property a consumer hashing
    under its own `@jit` depends on, which the length-built zero pad exists to
    keep (`sha256.digest` states the claim). One boundary-crossing length:
    compile time scales with block count, the property does not."""

    def test_jit_matches_eager_and_hashlib(self) -> None:
        # 65 bytes: the counter crosses a block boundary. The full (4, 65)
        # batch deliberately — the eager leg rides the sweep's shared
        # compile instead of minting a fresh aval.
        msgs = _message(65, seed=7)
        eager = np.asarray(blake2s.digest(msgs))
        traced = np.asarray(frx.jit(blake2s.digest)(msgs))
        np.testing.assert_array_equal(traced, eager)
        # Against the oracle rather than only against ourselves: eager
        # agreeing with a traced path that shares its bug proves nothing.
        for i in range(msgs.shape[0]):
            self.assertEqual(bytes(traced[i]), hashlib.blake2s(bytes(msgs[i])).digest())

    def test_the_seam_carries_it(self) -> None:
        # Through `ByteHash.digest` rather than the module function — and at
        # a non-default digest size, so the traced path also carries the
        # caller-side truncation.
        hasher: ByteHash = Blake2s(16)
        msgs = _message(65)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(hasher.digest)(msgs)),
            np.asarray(hasher.digest(msgs)),
        )

    def test_one_trace_serves_every_instance_and_digest_size(self) -> None:
        # The digest-size arrangement, pinned: the zone's cache keys on the
        # operand avals, and `h0` is uint32 [8] for EVERY size — the size
        # rides h0's VALUE and the host-side slice — so after the first call
        # freshly built instances of DIFFERENT sizes must all ride the same
        # (4, 65) trace, gaining the zone nothing. Identity-keyed instances
        # or a size-keyed zone would each fail here.
        msgs = fnp.asarray(_message(65))
        calls = [
            functools.partial(Blake2s(size).digest, msgs) for size in (32, 16, 1, 32)
        ]
        assert_single_trace(self, blake2s.blake2s_bytes, calls)


class Blake2sByteHashTest(absltest.TestCase):
    """The device row against the seam, and against its host partner."""

    def test_impls_satisfy_the_seam(self) -> None:
        for h in (Blake2s(), HostBlake2s()):
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
        device, host = Blake2s(), HostBlake2s()
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
                    np.asarray(Blake2s(size).digest(msg)),
                    HostBlake2s(size).digest(msg),
                )

    def test_digest_shape_and_dtype(self) -> None:
        for h in (Blake2s(20), HostBlake2s(20)):
            with self.subTest(impl=type(h).__name__):
                out = np.asarray(h.digest(np.zeros((4, 1), dtype=np.uint8)))
                self.assertEqual(out.shape, (4, 20))
                self.assertEqual(out.dtype, np.uint8)

    def test_value_identity_rides_the_digest_size(self) -> None:
        # By value over the length, like the sibling's rows — the param-free
        # by-type form does not apply. Same size equal (and hash-equal: what
        # keeps the seam re-trace-safe as pytree aux), cross-size unequal,
        # and never equal to the host row: swapping substrate must re-trace.
        self.assertEqual(Blake2s(16), Blake2s(16))
        self.assertEqual(hash(Blake2s(16)), hash(Blake2s(16)))
        self.assertNotEqual(Blake2s(16), Blake2s(32))
        self.assertNotEqual(Blake2s(16), HostBlake2s(16))
        self.assertNotEqual(Blake2s(16), object())

    def test_out_of_range_length_is_refused_at_construction(self) -> None:
        for size in (0, MAX_DIGEST_SIZE + 1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    Blake2s(size)
                with self.assertRaises(ValueError):
                    HostBlake2s(size)
                with self.assertRaises(ValueError):
                    # The module path re-checks in `_initial_state`: a caller
                    # bypassing the rows still gets the range error before
                    # any device work runs.
                    blake2s.digest(np.zeros((1, 1), dtype=np.uint8), size)


class SeamContractTest(absltest.TestCase):
    """The two seam invariants BLAKE2s landed alongside and so missed.

    A zero-row batch is a valid batch — the block-count reshape used to spell
    the count as `-1`, which a zero-sized total makes ambiguous (#211) — and a
    1-D message is rejected at the seam rather than from inside the marked
    region's trace (#215).
    """

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        rows: list[tuple[ByteHash, int]] = [
            (Blake2s(), 32),
            (HostBlake2s(), 32),
        ]
        for hasher, size in rows:
            got = np.asarray(hasher.digest(fnp.zeros((0, 64), dtype=fnp.uint8)))
            self.assertEqual(got.shape, (0, size))
            self.assertEqual(got.dtype, np.uint8)

    def test_a_1d_message_is_rejected_at_the_seam(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D uint8"):
            blake2s.digest(fnp.zeros(64, dtype=fnp.uint8))


if __name__ == "__main__":
    absltest.main()

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Grøstl-256 — KAT values, the S-box circuit, the marker, and the seam.

Values are held to the final-round KAT directly (the vectors `reference_test`
anchors across three independent sources) and to the oracle differentially at
lengths the transcription does not carry, so agreement means agreement with
Grøstl-256 rather than with a second copy of one misreading. The bitsliced
S-box circuit gets its own exhaustive case: it is the one component whose frx
spelling shares *nothing* with the oracle's table, and a wrong gate corrupts
every digest identically on both jit legs.

The lowering assertions are the usual half that values cannot see: the digest
must emit exactly one composite carrying the registered name, version, and
the five-operand ABI with no captured constants. No backend routes the name
yet, so recognition is not asserted — emission and the ABI are what an
emitter will read, pinned before it exists (the Vision arrangement).
"""

from __future__ import annotations

import functools
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.grostl import grostl
from hash_frx.grostl.grostl import Grostl256
from hash_frx.grostl.testing.host_grostl256 import HostGrostl256
from hash_frx.grostl.testing.reference import AES_SBOX, KAT_VECTORS
from hash_frx.testing.jit_cache import assert_single_trace

# Padding-boundary lengths for the differential sweep: 0/1 (empty + tiny),
# 55/56 (the one-vs-two-block cutoff — the 0x80 byte and the 8-byte block
# count need 9 bytes), 63/64 (block edge; 64 gets a whole extra padding
# block), 65 (one past it), 128 (multi-block).
_LENGTHS = (0, 1, 55, 56, 63, 64, 65, 128)


def _message(length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(length * 31 + seed)
    return rng.integers(0, 256, size=(4, length), dtype=np.uint8)


class Grostl256KatTest(parameterized.TestCase):
    @parameterized.parameters(*((len(m), m, d) for m, d in KAT_VECTORS))
    def test_matches_the_final_round_kat(
        self, _length: int, msg: bytes, digest_hex: str
    ) -> None:
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = np.asarray(grostl.digest(rows))[0]
        self.assertEqual(bytes(got).hex(), digest_hex)

    @parameterized.parameters(*_LENGTHS)
    def test_device_and_host_agree(self, length: int) -> None:
        # The differential partner issue #163 asks for: the device digest
        # against the oracle-backed host row, across every padding boundary
        # and a batch — not one convenient length. The host row loops the
        # oracle per row (`byte_hash.host_digest`), so the batch equality is
        # also the bulk-parallel claim: one data-parallel device call equals
        # the per-message digests, in order.
        msgs = _message(length)
        np.testing.assert_array_equal(
            np.asarray(Grostl256().digest(msgs)),
            np.asarray(HostGrostl256().digest(msgs)),
        )


class SboxCircuitTest(absltest.TestCase):
    def test_the_circuit_matches_the_standard_table_for_all_256_inputs(
        self,
    ) -> None:
        # The Boyar–Peralta transcription against the table-defined S-box,
        # exhaustively. The oracle's table is derived from the FIPS 197
        # definition and spot-anchored to the published table in
        # `reference_test`, so the two sides share no spelling.
        x = fnp.asarray(np.arange(256, dtype=np.uint8))
        got = np.asarray(frx.jit(grostl._sbox)(x))
        np.testing.assert_array_equal(got, np.array(AES_SBOX, dtype=np.uint8))


def _composite(fn: Any, *args: Any) -> Any:
    """The one composite eqn in `fn`'s jaxpr — read without lowering to MLIR
    (the `vision_test` helper)."""
    eqns = [
        e
        for e in frx.make_jaxpr(fn)(*args).jaxpr.eqns
        if e.primitive.name == "composite"
    ]
    assert len(eqns) == 1, f"expected one composite, got {len(eqns)}"
    return eqns[0]


class Grostl256MarkerTest(absltest.TestCase):
    def test_no_leg_routes_a_grostl_marker_yet(self) -> None:
        # The pre-emitter pin (the Vision arrangement): both module flags say
        # "no emitter", so every unpatched instance reads GENERIC on every
        # backend. When an emitter lands these flip with the frx floor and
        # this case flips to the keccak-style backend gate.
        self.assertFalse(grostl._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(grostl._EMITTER_BACKENDS, ())
        self.assertIs(Grostl256().fusion_path, FusionPath.GENERIC)

    def test_digest_emits_one_composite_with_the_digest_name(self) -> None:
        # The contract's unit: an absent or split marker still computes the
        # right bytes, so only the lowered module shows it.
        msg = np.zeros((2, 100), dtype=np.uint8)
        txt = frx.jit(grostl.digest).lower(msg).as_text()
        self.assertIn(grostl.GROSTL256_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        # The wire surface an emitter will read: name, version, and the
        # documented [iv, rc_p, rc_q, msg, tail] operand order. Five invars
        # exactly is the captured-constants-free property — an array the body
        # closed over would be lifted in AHEAD of these, one per call site
        # (the operand-ABI rule in docs/reference/conventions.md).
        msg = fnp.asarray(np.zeros((2, 100), dtype=np.uint8))
        eqn = _composite(grostl.digest, msg)
        self.assertEqual(eqn.params["name"], grostl.GROSTL256_MARKER)
        self.assertEqual(eqn.params["version"], grostl.GROSTL256_MARKER_VERSION)
        self.assertLen(eqn.invars, 5)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        # L = 100: two blocks once padded, so the tail is 28 bytes.
        self.assertEqual(shapes, [(64,), (10, 8, 8), (10, 8, 8), (2, 100), (28,)])


class Grostl256TracedTest(absltest.TestCase):
    """`digest` inside a traced region — the seam property a consumer hashing
    under its own `@jit` depends on, which the length-built padding exists to
    keep (`sha256.Sha256TracedTest` states the claim). One boundary-crossing
    length: compile time scales with block count, the property does not."""

    def test_jit_matches_eager_and_the_kat(self) -> None:
        msg, digest_hex = KAT_VECTORS[5]  # 65 bytes: crosses the block edge
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        eager = np.asarray(grostl.digest(rows))
        traced = np.asarray(frx.jit(grostl.digest)(rows))
        np.testing.assert_array_equal(traced, eager)
        # Against the record rather than only against ourselves: eager
        # agreeing with a traced path that shares its bug proves nothing.
        self.assertEqual(bytes(traced[0]).hex(), digest_hex)

    def test_the_seam_carries_it(self) -> None:
        # Through `ByteHash.digest` rather than the module function: the
        # consumer holds the seam, and that call must survive the tracer.
        hasher: ByteHash = Grostl256()
        msgs = _message(65, seed=3)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(hasher.digest)(msgs)),
            np.asarray(hasher.digest(msgs)),
        )

    def test_digest_reuses_one_trace_across_instances(self) -> None:
        # The zone's cache is keyed on the message aval alone (the digest is
        # parameterless), so freshly built instances must share one trace —
        # the property `grostl256_bytes`'s module-level jit zone exists for.
        # The (4, 65) aval is the one `test_the_seam_carries_it` already
        # compiled, so the pin costs no fresh compile.
        msgs = fnp.asarray(_message(65, seed=3))
        calls = [functools.partial(Grostl256().digest, msgs) for _ in range(3)]
        assert_single_trace(self, grostl.grostl256_bytes, calls)


class Grostl256ByteHashTest(absltest.TestCase):
    """The two `ByteHash` implementations, against the seam."""

    def test_impls_satisfy_the_seam(self) -> None:
        for h in (Grostl256(), HostGrostl256()):
            with self.subTest(impl=type(h).__name__):
                self.assertIsInstance(h, ByteHash)
                self.assertEqual(h.digest_size, 32)
                self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_paths_pin_the_substrate(self) -> None:
        # Device GENERIC (pre-emitter, every backend), host HOST (every
        # backend) — and the traceability tie to the return type: the device
        # row returns an `Array` and takes a tracer, the host row reads bytes
        # and never can (`byte_hash.py`'s rule).
        # A (1, 1) message: the KAT already compiled that aval, so the
        # plumbing check costs no fresh compile of the digest body.
        msg = np.zeros((1, 1), dtype=np.uint8)
        device, host = Grostl256(), HostGrostl256()
        self.assertIs(device.fusion_path, FusionPath.GENERIC)
        self.assertTrue(device.fusion_path.is_traceable)
        out = device.digest(msg)
        self.assertNotIsInstance(out, np.ndarray)
        self.assertIsInstance(out, Array)
        self.assertIs(host.fusion_path, FusionPath.HOST)
        self.assertFalse(host.fusion_path.is_traceable)
        self.assertIsInstance(host.digest(msg), np.ndarray)

    def test_digest_shape_and_dtype(self) -> None:
        # (4, 1) rides the differential sweep's aval — no fresh compile.
        for h in (Grostl256(), HostGrostl256()):
            with self.subTest(impl=type(h).__name__):
                out = np.asarray(h.digest(np.zeros((4, 1), dtype=np.uint8)))
                self.assertEqual(out.shape, (4, 32))
                self.assertEqual(out.dtype, np.uint8)

    def test_value_identity_is_by_type(self) -> None:
        # Param-free, so every instance of a type is equal and hashes alike —
        # what keeps the seam re-trace-safe as pytree aux. The two are never
        # equal, or swapping substrate would not re-trace.
        for cls in (Grostl256, HostGrostl256):
            with self.subTest(impl=cls.__name__):
                self.assertEqual(cls(), cls())
                self.assertEqual(hash(cls()), hash(cls()))
        self.assertNotEqual(Grostl256(), HostGrostl256())


class EmptyBatchTest(absltest.TestCase):
    """A zero-row batch is a valid batch (#211): every row returns
    uint8 [0, digest_size] instead of failing in a block-count reshape."""

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        rows: list[tuple[ByteHash, int]] = [(grostl.Grostl256(), 32)]
        for hasher, size in rows:
            got = np.asarray(hasher.digest(fnp.zeros((0, 64), dtype=fnp.uint8)))
            self.assertEqual(got.shape, (0, size))
            self.assertEqual(got.dtype, np.uint8)


class MessageRankTest(absltest.TestCase):
    """The row routes its message through the seam's rank check (#215): a
    single message is `B = 1`, not a bare `[L]`, and the miss used to surface
    from inside the marked region's trace instead of at the call."""

    def test_a_1d_message_is_rejected_at_the_seam(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D uint8"):
            grostl.digest(fnp.zeros(64, dtype=fnp.uint8))


if __name__ == "__main__":
    absltest.main()

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ascon-Hash256 — KAT values, the S-box circuit, the marker, and the seam.

Values are held to the SP 800-232 KAT directly (the vectors `reference_test`
anchors across three independent sources) and to the oracle differentially at
lengths the transcription does not carry, so agreement means agreement with
Ascon-Hash256 rather than with a second copy of one misreading. The bitsliced
S-box circuit gets its own exhaustive case: it is the one component whose frx
spelling shares *nothing* with the oracle's table, and a wrong gate corrupts
every digest identically on both jit legs. The precomputed initial state is
pinned the other way around — the device module transcribes Table 12, the
oracle derives it from the IV, and the two must meet.

The lowering assertions are the usual half that values cannot see: the digest
must emit exactly one composite carrying the registered name, version, and
the three-operand ABI with no captured constants. No backend routes the name
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

from hash_frx.ascon import ascon
from hash_frx.ascon.ascon import AsconHash256
from hash_frx.ascon.testing.host_ascon_hash256 import HostAsconHash256
from hash_frx.ascon.testing.reference import INITIAL_STATE, KAT_VECTORS, SBOX
from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.word import split

# Rate-boundary lengths for the differential sweep: 0/1 (empty + tiny — even
# the empty message absorbs one padded block), 7/8 (the one-block edge; 8
# aligned bytes gain a whole padding block), 9 (one past it), 15/16 (the
# two-block edge), 65 (multi-block, one past the eighth boundary — a length
# the KAT transcription does not carry).
_LENGTHS = (0, 1, 7, 8, 9, 15, 16, 65)


def _message(length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(length * 31 + seed)
    return rng.integers(0, 256, size=(4, length), dtype=np.uint8)


class AsconHash256KatTest(parameterized.TestCase):
    @parameterized.parameters(*((len(m), m, d) for m, d in KAT_VECTORS))
    def test_matches_the_sp800_232_kat(
        self, _length: int, msg: bytes, digest_hex: str
    ) -> None:
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = np.asarray(ascon.digest(rows))[0]
        self.assertEqual(bytes(got).hex(), digest_hex)

    @parameterized.parameters(*_LENGTHS)
    def test_device_and_host_agree(self, length: int) -> None:
        # The differential partner issue #188 asks for: the device digest
        # against the oracle-backed host row, across every rate boundary and
        # a batch — not one convenient length. The host row loops the oracle
        # per row (`byte_hash.host_digest`), so the batch equality is also
        # the bulk-parallel claim: one data-parallel device call equals the
        # per-message digests, in order.
        msgs = _message(length)
        np.testing.assert_array_equal(
            np.asarray(AsconHash256().digest(msgs)),
            np.asarray(HostAsconHash256().digest(msgs)),
        )


class SboxCircuitTest(absltest.TestCase):
    def test_the_circuit_matches_the_standard_table_for_all_32_inputs(
        self,
    ) -> None:
        # The masked-roll grid circuit against the table-defined S-box,
        # exhaustively — the two sides share no spelling (the oracle's table
        # is transcribed from Table 6 and corner-anchored in
        # `reference_test`). Word i's low half packs bit x_i of every 5-bit
        # value, bit position j carrying input j — the bitsliced orientation
        # the state grid has, x0 the most significant index bit (Table 6's
        # convention); the high halves ride the same gates, so zeros there
        # only re-check column 0x00.
        planes = [0, 0, 0, 0, 0]
        for j in range(32):
            for i in range(5):
                planes[i] |= ((j >> (4 - i)) & 1) << j
        lo = fnp.asarray(np.array([planes], dtype=np.uint32))
        hi = fnp.asarray(np.zeros((1, 5), dtype=np.uint32))
        out_lo, _ = frx.jit(lambda lo, hi: ascon._substitution(lo, hi, ascon._masks()))(
            lo, hi
        )
        out = np.asarray(out_lo)[0]
        got = []
        for j in range(32):
            y = 0
            for i in range(5):
                y |= (int(out[i]) >> j & 1) << (4 - i)
            got.append(y)
        self.assertEqual(tuple(got), SBOX)


class InitialStateTest(absltest.TestCase):
    def test_the_transcription_meets_the_oracle_derivation(self) -> None:
        # `ascon.py` transcribes the SP's precomputed Table 12 state; the
        # oracle derives Ascon-p[12](IV ‖ 0^256) from Table 14's IV. Meeting
        # here — through `word.split`'s exact host halves — pins the
        # transcription against an independent derivation, so a mistyped
        # digit fails even though both sides come from one document.
        derived = np.array([split(w) for w in INITIAL_STATE], dtype=np.uint32)
        np.testing.assert_array_equal(ascon._INITIAL_STATE, derived)


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


class AsconHash256MarkerTest(absltest.TestCase):
    def test_no_leg_routes_an_ascon_marker_yet(self) -> None:
        # The pre-emitter pin (the Vision arrangement): both module flags say
        # "no emitter", so every unpatched instance reads GENERIC on every
        # backend. When an emitter lands these flip with the frx floor and
        # this case flips to the keccak-style backend gate.
        self.assertFalse(ascon._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(ascon._EMITTER_BACKENDS, ())
        self.assertIs(AsconHash256().fusion_path, FusionPath.GENERIC)

    def test_digest_emits_one_composite_with_the_digest_name(self) -> None:
        # The contract's unit: an absent or split marker still computes the
        # right bytes, so only the lowered module shows it.
        msg = np.zeros((2, 100), dtype=np.uint8)
        txt = frx.jit(ascon.digest).lower(msg).as_text()
        self.assertIn(ascon.ASCON_HASH256_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        # The wire surface an emitter will read: name, version, and the
        # documented [init, msg, tail] operand order. Three invars exactly is
        # the captured-constants-free property — an array the body closed
        # over would be lifted in AHEAD of these, one per call site (the
        # operand-ABI rule in docs/reference/conventions.md); the round
        # constants ride as scalar literals, which this count is the pin for.
        msg = fnp.asarray(np.zeros((2, 100), dtype=np.uint8))
        eqn = _composite(ascon.digest, msg)
        self.assertEqual(eqn.params["name"], ascon.ASCON_HASH256_MARKER)
        self.assertEqual(eqn.params["version"], ascon.ASCON_HASH256_MARKER_VERSION)
        self.assertLen(eqn.invars, 3)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        # L = 100: 4 bytes short of the 13-block boundary, so the tail is 4.
        self.assertEqual(shapes, [(5, 2), (2, 100), (4,)])


class AsconHash256TracedTest(absltest.TestCase):
    """`digest` inside a traced region — the seam property a consumer hashing
    under its own `@jit` depends on, which the length-built padding exists to
    keep (`sha256.Sha256TracedTest` states the claim). One boundary-crossing
    length: compile time scales with block count, the property does not."""

    def test_jit_matches_eager_and_the_kat(self) -> None:
        msg, digest_hex = KAT_VECTORS[4]  # 9 bytes: crosses the block edge
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        eager = np.asarray(ascon.digest(rows))
        traced = np.asarray(frx.jit(ascon.digest)(rows))
        np.testing.assert_array_equal(traced, eager)
        # Against the record rather than only against ourselves: eager
        # agreeing with a traced path that shares its bug proves nothing.
        self.assertEqual(bytes(traced[0]).hex(), digest_hex)

    def test_the_seam_carries_it(self) -> None:
        # Through `ByteHash.digest` rather than the module function: the
        # consumer holds the seam, and that call must survive the tracer.
        hasher: ByteHash = AsconHash256()
        msgs = _message(9, seed=3)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(hasher.digest)(msgs)),
            np.asarray(hasher.digest(msgs)),
        )

    def test_digest_reuses_one_trace_across_instances(self) -> None:
        # The zone's cache is keyed on the message aval alone (the digest is
        # parameterless), so freshly built instances must share one trace —
        # the property `ascon_hash256_bytes`'s module-level jit zone exists
        # for. The (4, 9) aval is the one `test_the_seam_carries_it` already
        # compiled, so the pin costs no fresh compile.
        msgs = fnp.asarray(_message(9, seed=3))
        calls = [functools.partial(AsconHash256().digest, msgs) for _ in range(3)]
        assert_single_trace(self, ascon.ascon_hash256_bytes, calls)


class AsconHash256ByteHashTest(absltest.TestCase):
    """The two `ByteHash` implementations, against the seam."""

    def test_impls_satisfy_the_seam(self) -> None:
        for h in (AsconHash256(), HostAsconHash256()):
            with self.subTest(impl=type(h).__name__):
                self.assertIsInstance(h, ByteHash)
                self.assertEqual(h.digest_size, 32)
                self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_paths_pin_the_substrate(self) -> None:
        # Device GENERIC (pre-emitter, every backend), host HOST (every
        # backend) — and the traceability tie to the return type: the device
        # row returns an `Array` and takes a tracer, the host row reads bytes
        # and never can (`byte_hash.py`'s rule).
        # A (1, 1) message: the KAT sweep covers that aval, so the plumbing
        # check adds no distinct compile of the digest body.
        msg = np.zeros((1, 1), dtype=np.uint8)
        device, host = AsconHash256(), HostAsconHash256()
        self.assertIs(device.fusion_path, FusionPath.GENERIC)
        self.assertTrue(device.fusion_path.is_traceable)
        out = device.digest(msg)
        self.assertNotIsInstance(out, np.ndarray)
        self.assertIsInstance(out, Array)
        self.assertIs(host.fusion_path, FusionPath.HOST)
        self.assertFalse(host.fusion_path.is_traceable)
        self.assertIsInstance(host.digest(msg), np.ndarray)

    def test_digest_shape_and_dtype(self) -> None:
        # (4, 1) rides the differential sweep's aval — no distinct compile.
        for h in (AsconHash256(), HostAsconHash256()):
            with self.subTest(impl=type(h).__name__):
                out = np.asarray(h.digest(np.zeros((4, 1), dtype=np.uint8)))
                self.assertEqual(out.shape, (4, 32))
                self.assertEqual(out.dtype, np.uint8)

    def test_value_identity_is_by_type(self) -> None:
        # Param-free, so every instance of a type is equal and hashes alike —
        # what keeps the seam re-trace-safe as pytree aux. The two are never
        # equal, or swapping substrate would not re-trace.
        for cls in (AsconHash256, HostAsconHash256):
            with self.subTest(impl=cls.__name__):
                self.assertEqual(cls(), cls())
                self.assertEqual(hash(cls()), hash(cls()))
        self.assertNotEqual(AsconHash256(), HostAsconHash256())


if __name__ == "__main__":
    absltest.main()

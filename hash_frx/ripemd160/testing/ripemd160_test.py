# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""RIPEMD-160 — the designers' vectors, the marker, and the seam.

Values are held to the published vectors directly (the nine rows
`ripemd160_reference_test` anchors, Bitcoin Core's independent transcription
agreeing) and to the oracle differentially at lengths the record does not
carry, so agreement means agreement with RIPEMD-160 rather than with a second
copy of one misreading. The differential is what catches the byte-order trap
the module documents: a SHA-2-shaped (big-endian) packing or length field
passes no case here beyond the empty message.

The lowering assertions are the usual half that values cannot see: the digest
must emit exactly one composite carrying the registered name, version, and
the three-operand ABI with no captured constants. No backend routes the name
yet, so recognition is not asserted — emission and the ABI are what an
emitter will read, pinned before it exists (the Vision/Grøstl arrangement).
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

from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.ripemd160 import ripemd160
from hash_frx.ripemd160.ripemd160 import Ripemd160
from hash_frx.ripemd160.testing.reference import VECTORS
from hash_frx.ripemd160.testing.reference import ripemd160 as _ripemd160_oracle
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.testing.oracle import oracle_digest

# Padding-boundary lengths for the differential sweep: 0/1 (empty + tiny),
# 55/56 (the one-vs-two-block cutoff — the 0x80 byte and the 8-byte length
# need 9 bytes), 63/64 (block edge; 64 gets a whole extra padding block), 65
# (one past it), 128 (multi-block).
_LENGTHS = (0, 1, 55, 56, 63, 64, 65, 128)


def _message(length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(length * 31 + seed)
    return rng.integers(0, 256, size=(4, length), dtype=np.uint8)


def _hashlib_has_ripemd160() -> bool:
    try:
        hashlib.new("ripemd160")
    except ValueError:  # unsupported hash type: the legacy-provider gap
        return False
    return True


class Ripemd160VectorTest(parameterized.TestCase):
    @parameterized.parameters(*((len(m), m, d) for m, d in VECTORS))
    def test_matches_the_designers_vectors(
        self, _length: int, msg: bytes, digest_hex: str
    ) -> None:
        # Against the standard's record, not against another implementation in
        # this tree (docs/reference/conventions.md: byte-exactness is the
        # gate). The million-"a" row runs on the oracle only
        # (`ripemd160_reference_test`): at ~15k blocks the unrolled device
        # graph buys nothing a two-block message has not already proven.
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = np.asarray(ripemd160.digest(rows))[0]
        self.assertEqual(bytes(got).hex(), digest_hex)

    @parameterized.parameters(*_LENGTHS)
    def test_device_and_host_agree(self, length: int) -> None:
        # The differential partner issue #189 asks for: the device digest
        # against the reference oracle, across every padding boundary and a
        # batch — not one convenient length.
        msgs = _message(length)
        np.testing.assert_array_equal(
            np.asarray(Ripemd160().digest(msgs)),
            oracle_digest(_ripemd160_oracle, 20, msgs),
        )

    @absltest.skipUnless(
        _hashlib_has_ripemd160(),
        "hashlib lacks ripemd160 (OpenSSL 3 legacy provider)",
    )
    @parameterized.parameters(*_LENGTHS)
    def test_device_matches_hashlib_where_available(self, length: int) -> None:
        # Opportunistic third source where the interpreter's OpenSSL still
        # carries RIPEMD-160: the device digest against an implementation the
        # oracle shares nothing with. The (4, length) avals ride the sweep
        # above, so no fresh compile.
        msgs = _message(length)
        got = np.asarray(Ripemd160().digest(msgs))
        for i in range(msgs.shape[0]):
            self.assertEqual(
                bytes(got[i]), hashlib.new("ripemd160", bytes(msgs[i])).digest()
            )


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


class Ripemd160MarkerTest(absltest.TestCase):
    def test_no_leg_routes_a_ripemd160_marker_yet(self) -> None:
        # The pre-emitter pin (the Vision/Grøstl arrangement): both module
        # flags say "no emitter", so every unpatched instance reads GENERIC on
        # every backend. When an emitter lands these flip with the frx floor
        # and this case flips to the keccak-style backend gate.
        self.assertFalse(ripemd160._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(ripemd160._EMITTER_BACKENDS, ())
        self.assertIs(Ripemd160().fusion_path, FusionPath.GENERIC)

    def test_digest_emits_one_composite_with_the_digest_name(self) -> None:
        # The contract's unit: an absent or split marker still computes the
        # right bytes, so only the lowered module shows it.
        msg = np.zeros((2, 100), dtype=np.uint8)
        txt = frx.jit(ripemd160.digest).lower(msg).as_text()
        self.assertIn(ripemd160.RIPEMD160_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        # The wire surface an emitter will read: name, version, and the
        # documented [h0, msg, tail] operand order. Three invars exactly is
        # the captured-constants-free property — an array the body closed over
        # would be lifted in AHEAD of these, one per call site (the
        # operand-ABI rule in docs/reference/conventions.md); the K constants
        # and r/s schedules must enter as scalar literals and trace-time
        # slices, never as lifted table arrays.
        msg = fnp.asarray(np.zeros((2, 100), dtype=np.uint8))
        eqn = _composite(ripemd160.digest, msg)
        self.assertEqual(eqn.params["name"], ripemd160.RIPEMD160_MARKER)
        self.assertEqual(eqn.params["version"], ripemd160.RIPEMD160_MARKER_VERSION)
        # The `primitive` attribute is the OTHER half of the operation-name
        # migration and the reason the flip is a rename and nothing else: the
        # plugin resolves the family through it once the name stops carrying
        # one. Pinned here, while the marker still rides its own spelling,
        # because a family that quietly stopped emitting it would keep passing
        # every test above and then decline into its decomposition at flip time
        # rather than fail (`markers.bytes_in_digest_marker`).
        attrs = {key: leaves[0] for key, leaves, _ in eqn.params["attributes"]}
        self.assertEqual(attrs["primitive"], "ripemd160")
        self.assertLen(eqn.invars, 3)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        # L = 100: two blocks once padded, so the tail is 28 bytes.
        self.assertEqual(shapes, [(5,), (2, 100), (28,)])


class Ripemd160TracedTest(absltest.TestCase):
    """`digest` inside a traced region — the seam property a consumer hashing
    under its own `@jit` depends on, which the length-built padding exists to
    keep (`sha256.digest` states the claim). One boundary-crossing length:
    compile time scales with block count, the property does not."""

    def test_jit_matches_eager_and_the_vector(self) -> None:
        msg, digest_hex = VECTORS[5]  # 56 bytes: the padding forces block two
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        eager = np.asarray(ripemd160.digest(rows))
        traced = np.asarray(frx.jit(ripemd160.digest)(rows))
        np.testing.assert_array_equal(traced, eager)
        # Against the record rather than only against ourselves: eager
        # agreeing with a traced path that shares its bug proves nothing.
        self.assertEqual(bytes(traced[0]).hex(), digest_hex)

    def test_the_seam_carries_it(self) -> None:
        # Through `ByteHash.digest` rather than the module function: the
        # consumer holds the seam, and that call must survive the tracer.
        hasher: ByteHash = Ripemd160()
        msgs = _message(65, seed=3)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(hasher.digest)(msgs)),
            np.asarray(hasher.digest(msgs)),
        )

    def test_digest_reuses_one_trace_across_instances(self) -> None:
        # The zone's cache is keyed on the message aval alone (the digest is
        # parameterless), so freshly built instances must share one trace —
        # the property `ripemd160_bytes`'s module-level jit zone exists for.
        # The (4, 65) aval is the one `test_the_seam_carries_it` already
        # compiled, so the pin costs no fresh compile.
        msgs = fnp.asarray(_message(65, seed=3))
        calls = [functools.partial(Ripemd160().digest, msgs) for _ in range(3)]
        assert_single_trace(self, ripemd160.ripemd160_bytes, calls)


class Ripemd160ByteHashTest(absltest.TestCase):
    """The `ByteHash` row, against the seam."""

    def test_impls_satisfy_the_seam(self) -> None:
        h = Ripemd160()
        self.assertIsInstance(h, ByteHash)
        self.assertEqual(h.digest_size, 20)
        self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_paths_pin_the_substrate(self) -> None:
        # GENERIC (pre-emitter, every backend), and the traceability tie to
        # the return type: the row returns an `Array` and takes a tracer
        # (`byte_hash.py`'s rule).
        # A (1, 1) message: the "a" vector already compiled that aval, so the
        # plumbing check costs no fresh compile of the digest body.
        msg = np.zeros((1, 1), dtype=np.uint8)
        device = Ripemd160()
        self.assertIs(device.fusion_path, FusionPath.GENERIC)
        out = device.digest(msg)
        self.assertNotIsInstance(out, np.ndarray)
        self.assertIsInstance(out, Array)

    def test_digest_shape_and_dtype(self) -> None:
        # (4, 1) rides the differential sweep's aval — no fresh compile.
        out = np.asarray(Ripemd160().digest(np.zeros((4, 1), dtype=np.uint8)))
        self.assertEqual(out.shape, (4, 20))
        self.assertEqual(out.dtype, np.uint8)

    def test_value_identity_is_by_type(self) -> None:
        # Param-free, so every instance of a type is equal and hashes alike —
        # what keeps the seam re-trace-safe as pytree aux.
        self.assertEqual(Ripemd160(), Ripemd160())
        self.assertEqual(hash(Ripemd160()), hash(Ripemd160()))


class EmptyBatchTest(absltest.TestCase):
    """A zero-row batch is a valid batch (#211): every row returns
    uint8 [0, digest_size] instead of failing in a block-count reshape."""

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        got = np.asarray(
            ripemd160.Ripemd160().digest(fnp.zeros((0, 64), dtype=fnp.uint8))
        )
        self.assertEqual(got.shape, (0, 20))
        self.assertEqual(got.dtype, np.uint8)


if __name__ == "__main__":
    absltest.main()

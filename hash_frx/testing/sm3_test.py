# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SM3 byte-hash — the GB/T 32905 worked examples, then byte-match against
OpenSSL's implementation through `hashlib.new("sm3")` (the host row's own
binding; its docstring carries the availability story).

The lengths exercise every padding boundary — empty, sub-block, the 55/56
one-vs-two block transition (where the 0x80 byte plus the 8-byte length field
force a second block), exact block multiples, and a multi-block message. The
0xFF sweep saturates the TT1/TT2 add chains the way the SHA-2 suites do.

The lowering assertions are the half that values cannot see: one composite
carrying the registered name, version, and the three-operand ABI — h0, the
pre-rotated T table, and the packed blocks — with no captured constants. No
backend routes the name yet, so recognition is not asserted (the Vision/Grøstl
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

from hash_frx import sm3
from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.sha256 import Sha256
from hash_frx.sm3 import HostSm3, Sm3
from hash_frx.testing.jit_cache import assert_single_trace

# Padding-boundary lengths: 0/1 (empty + tiny), 55/56 (the one-block/two-block
# cutoff — the 0x80 byte + 8-byte length need 9 bytes), 63/64 (block edge; 64
# gets a whole extra padding block), 65 (one past it), 128 (multi-block).
_LENGTHS = (0, 1, 55, 56, 63, 64, 65, 128)

# GB/T 32905's two worked examples ("abc" and the 512-bit "abcd"×16 message),
# plus the empty message every independent transcription publishes.
_VECTORS = (
    (
        "abc",
        b"abc",
        "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0",
    ),
    (
        "one_block_512_bit",
        b"abcd" * 16,
        "debe9ff92275b8a138604889c18e5a4d6fdb70e5387e5765293dcba39c0c5732",
    ),
    (
        "empty",
        b"",
        "1ab21d8355cfa17f8e61194831e81a8f22bec8c728fefb747ed035eb5082aa2b",
    ),
)


def _sm3_oracle(data: bytes) -> bytes:
    return hashlib.new("sm3", data).digest()


class Sm3VectorTest(parameterized.TestCase):
    @parameterized.named_parameters(*_VECTORS)
    def test_matches_the_published_vector(self, msg: bytes, digest_hex: str) -> None:
        # Against the standard's record, not against another implementation in
        # this tree (docs/reference/conventions.md: byte-exactness is the gate).
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = np.asarray(sm3.digest(rows))[0]
        self.assertEqual(bytes(got).hex(), digest_hex)
        # And the reference the differential sweep uses is anchored to the
        # same record, so agreeing with it below means agreeing with GB/T.
        self.assertEqual(_sm3_oracle(msg).hex(), digest_hex)


class Sm3Test(parameterized.TestCase):
    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        got = bytes(np.asarray(sm3.digest(msg[None, :]))[0])
        self.assertEqual(got, _sm3_oracle(bytes(msg)))

    @parameterized.parameters(1, 7, 55, 56, 64, 128)
    def test_all_ones_matches_hashlib(self, length: int) -> None:
        # 0xFF-filled messages: every expanded word is high-weight, so the
        # TT1/TT2 add chains run saturated — the deliberate exercise of the
        # mod-2^32 sums (the sha256/sha512 carry-sweep arrangement; SM3's
        # feedforward is XOR, so the adds all live in the rounds).
        msg = np.full(length, 0xFF, dtype=np.uint8)
        got = bytes(np.asarray(sm3.digest(msg[None, :]))[0])
        self.assertEqual(got, _sm3_oracle(bytes(msg)))

    def test_batched_equals_per_row(self) -> None:
        # One data-parallel call over a stack of equal-length messages must
        # equal the per-message oracle digests, in order.
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(7, 64), dtype=np.uint8)
        got = np.asarray(sm3.digest(batch))
        for i in range(batch.shape[0]):
            self.assertEqual(bytes(got[i]), _sm3_oracle(bytes(batch[i])))


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


class Sm3MarkerTest(absltest.TestCase):
    def test_no_leg_routes_an_sm3_marker_yet(self) -> None:
        # The pre-emitter pin (the Vision/Grøstl arrangement): both module
        # flags say "no emitter", so every unpatched instance reads GENERIC
        # on every backend. When an emitter lands these flip with the frx
        # floor and this case flips to the keccak-style backend gate.
        self.assertFalse(sm3._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(sm3._EMITTER_BACKENDS, ())
        self.assertIs(Sm3().fusion_path, FusionPath.GENERIC)

    def test_digest_emits_one_composite_with_the_digest_name(self) -> None:
        # The contract's unit: an absent or split marker still computes the
        # right bytes, so only the lowered module shows it. (1, 55) is the
        # differential sweep's aval, so the marker cases add no fresh trace.
        msg = np.zeros((1, 55), dtype=np.uint8)
        txt = frx.jit(sm3.digest).lower(msg).as_text()
        self.assertIn(sm3.SM3_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        # The wire surface an emitter will read: name, version, and the
        # documented [h0, t, blocks] operand order. Three invars exactly is
        # the captured-constants-free property — an array the body closed
        # over would be lifted in AHEAD of these, one per call site (the
        # operand-ABI rule in docs/reference/conventions.md). The T table is
        # an OPERAND, unlike the sibling digests' scalar-literal constants,
        # because the fori_loop reads it at a traced round index.
        msg = fnp.asarray(np.zeros((1, 55), dtype=np.uint8))
        eqn = _composite(sm3.digest, msg)
        self.assertEqual(eqn.params["name"], sm3.SM3_MARKER)
        self.assertEqual(eqn.params["version"], sm3.SM3_MARKER_VERSION)
        self.assertLen(eqn.invars, 3)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        # L = 55: one block once padded (0x80 + the 8-byte length just fit).
        self.assertEqual(shapes, [(8,), (64,), (1, 1, 16)])

    def test_a_block_boundary_gains_a_padding_block(self) -> None:
        # L = 56 no longer fits the length field in block one, so the chain
        # runs two blocks; only the blocks operand's shape moves with L (the
        # invar COUNT is the pinned surface).
        msg = fnp.asarray(np.zeros((1, 56), dtype=np.uint8))
        eqn = _composite(sm3.digest, msg)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        self.assertEqual(shapes, [(8,), (64,), (1, 2, 16)])


class Sm3TracedTest(absltest.TestCase):
    """`digest` inside a traced region — the seam property a consumer hashing
    under its own `@jit` depends on, which the length-built padding exists to
    keep (`sha256.digest` states the claim). One boundary-crossing length:
    compile time scales with block count, the property does not."""

    def test_jit_matches_eager_and_hashlib(self) -> None:
        rng = np.random.default_rng(65 * 31 + 7)
        msgs = rng.integers(0, 256, size=(4, 65), dtype=np.uint8)
        eager = np.asarray(sm3.digest(msgs))
        traced = np.asarray(frx.jit(sm3.digest)(msgs))
        np.testing.assert_array_equal(traced, eager)
        # Against the oracle rather than only against ourselves: eager
        # agreeing with a traced path that shares its bug proves nothing.
        for i in range(msgs.shape[0]):
            self.assertEqual(bytes(traced[i]), _sm3_oracle(bytes(msgs[i])))

    def test_the_seam_carries_it(self) -> None:
        # Through `ByteHash.digest` rather than the module function.
        hasher: ByteHash = Sm3()
        rng = np.random.default_rng(65 * 31)
        msgs = rng.integers(0, 256, size=(4, 65), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(hasher.digest)(msgs)),
            np.asarray(hasher.digest(msgs)),
        )

    def test_digest_reuses_one_trace_across_instances(self) -> None:
        # Param-free by-type identity is what keeps the seam re-trace-safe as
        # pytree aux: fresh instances and the module function must all ride
        # one trace of the marked chain for one message shape.
        msgs = fnp.asarray(np.zeros((4, 65), dtype=np.uint8))
        calls = [
            functools.partial(Sm3().digest, msgs),
            functools.partial(Sm3().digest, msgs),
            functools.partial(sm3.digest, msgs),
        ]
        assert_single_trace(self, sm3.sm3_merkle_damgard, calls)


class Sm3ByteHashTest(parameterized.TestCase):
    """The two `ByteHash` implementations, against the seam and against each
    other — the `Sha512ByteHashTest` split at the ShangMi table."""

    def test_impls_satisfy_the_seam(self) -> None:
        for h in (Sm3(), HostSm3()):
            with self.subTest(impl=type(h).__name__):
                self.assertIsInstance(h, ByteHash)
                self.assertEqual(h.digest_size, 32)
                self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_paths_pin_the_substrate(self) -> None:
        # Device GENERIC (pre-emitter, every backend), host HOST (every
        # backend) — and the traceability tie to the return type: the device
        # row returns an `Array` and takes a tracer, the host row reads bytes
        # and never can (`byte_hash.py`'s rule).
        msg = np.zeros((1, 1), dtype=np.uint8)
        device, host = Sm3(), HostSm3()
        self.assertIs(device.fusion_path, FusionPath.GENERIC)
        self.assertTrue(device.fusion_path.is_traceable)
        out = device.digest(msg)
        self.assertNotIsInstance(out, np.ndarray)
        self.assertIsInstance(out, Array)
        self.assertIs(host.fusion_path, FusionPath.HOST)
        self.assertFalse(host.fusion_path.is_traceable)
        self.assertIsInstance(host.digest(msg), np.ndarray)

    @parameterized.parameters(*_LENGTHS)
    def test_host_matches_hashlib(self, length: int) -> None:
        # HostSm3 is a separate implementation, not a wrapper over the marked
        # path: it loops `hashlib` per row. Nothing above covers it.
        rng = np.random.default_rng(length)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        got = np.asarray(HostSm3().digest(msgs))
        self.assertEqual(got.shape, (4, 32))
        for i in range(msgs.shape[0]):
            self.assertEqual(bytes(got[i]), _sm3_oracle(bytes(msgs[i])))

    @parameterized.parameters(*_LENGTHS)
    def test_device_and_host_agree(self, length: int) -> None:
        # Two implementations of identical bytes are only safe while they
        # stay identical; the guard that keeps them from drifting, across
        # every padding boundary rather than one convenient length.
        rng = np.random.default_rng(length + 1)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(Sm3().digest(msgs)),
            np.asarray(HostSm3().digest(msgs)),
        )

    def test_value_identity_is_by_type(self) -> None:
        # Param-free, so every instance of a type is equal and hashes alike —
        # what keeps the seam re-trace-safe as pytree aux. The two are never
        # equal, or swapping substrate would not re-trace; and neither equals
        # the structural cousin, or a family holding both would collide.
        for cls in (Sm3, HostSm3):
            with self.subTest(impl=cls.__name__):
                self.assertEqual(cls(), cls())
                self.assertEqual(hash(cls()), hash(cls()))
        self.assertNotEqual(Sm3(), HostSm3())
        self.assertNotEqual(Sm3(), Sha256())


class SeamContractTest(absltest.TestCase):
    """The two seam invariants SM3 landed alongside and so missed.

    A zero-row batch is a valid batch — the block-count reshape used to spell
    the count as `-1`, which a zero-sized total makes ambiguous (#211) — and a
    1-D message is rejected at the seam rather than from inside the marked
    region's trace, where it surfaced as a concatenate error naming neither
    (#215).
    """

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        rows: list[tuple[ByteHash, int]] = [(sm3.Sm3(), 32), (sm3.HostSm3(), 32)]
        for hasher, size in rows:
            got = np.asarray(hasher.digest(fnp.zeros((0, 64), dtype=fnp.uint8)))
            self.assertEqual(got.shape, (0, size))
            self.assertEqual(got.dtype, np.uint8)

    def test_a_1d_message_is_rejected_at_the_seam(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D uint8"):
            sm3.digest(fnp.zeros(64, dtype=fnp.uint8))


if __name__ == "__main__":
    absltest.main()

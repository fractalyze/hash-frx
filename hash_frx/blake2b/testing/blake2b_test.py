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

**The keyed, salted and personalized forms are held to the reference
suite's own vectors** (`blake2-kat.json`, the single 64-byte key
`bytes(range(64))`) at the block boundaries, and to `hashlib`'s
`key=/salt=/person=` across the parameter surface. Those two are not the same
check: the KAT proves the construction, while `hashlib` agreement proves the
parameter block reached `h0` field-by-field — a transposed salt offset passes
neither, but a wrong KEY-BLOCK count passes the second alone at every length
that is not a boundary.

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
from hash_frx.blake2b.blake2b import Blake2b, Blake2bKeyed
from hash_frx.blake2b.byte_hashes import (
    MAX_DIGEST_SIZE,
    HostBlake2b,
    HostBlake2bKeyed,
)
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
        # One instance suffices: the path reads the two flags alone, and the
        # size sweeps elsewhere already construct every length.
        self.assertIs(Blake2b().fusion_path, FusionPath.GENERIC)

    def test_digest_emits_one_composite_with_the_digest_name(self) -> None:
        # The contract's unit: an absent or split marker still computes the
        # right bytes, so only the lowered module shows it. (4, 127) is the
        # differential sweep's aval, so the marker cases add no fresh trace.
        msg = np.zeros((4, 127), dtype=np.uint8)
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
        msg = fnp.asarray(np.zeros((4, 127), dtype=np.uint8))
        eqn = _composite(blake2b.digest, msg)
        self.assertEqual(eqn.params["name"], blake2b.BLAKE2B_MARKER)
        self.assertEqual(eqn.params["version"], blake2b.BLAKE2B_MARKER_VERSION)
        self.assertLen(eqn.invars, 4)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        # L = 127: one block once padded, so the zero tail is 1 byte.
        self.assertEqual(shapes, [(16,), (16,), (4, 127), (1,)])

    def test_a_block_multiple_rides_an_empty_tail(self) -> None:
        # The ABI's degenerate shape: at L a multiple of 128 the zero tail is
        # empty, and the operand still rides — zero-length, never dropped
        # (the invar COUNT is the pinned surface; only shapes move with L).
        msg = fnp.asarray(np.zeros((4, 128), dtype=np.uint8))
        eqn = _composite(blake2b.digest, msg)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        self.assertEqual(shapes, [(16,), (16,), (4, 128), (0,)])


class Blake2bTracedTest(absltest.TestCase):
    """`digest` inside a traced region — the seam property a consumer hashing
    under its own `@jit` depends on, which the length-built zero pad exists to
    keep (`sha256.digest` states the claim). One boundary-crossing length:
    compile time scales with block count, the property does not."""

    def test_jit_matches_eager_and_hashlib(self) -> None:
        # 129 bytes: the counter crosses a block boundary. The full (4, 129)
        # batch deliberately — the eager leg then rides the class's shared
        # compile instead of minting a fresh aval.
        msgs = _message(129, seed=7)
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


class EmptyBatchTest(absltest.TestCase):
    """A zero-row batch is a valid batch (#211): every row returns
    uint8 [0, digest_size] instead of failing in a block-count reshape."""

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        rows: list[tuple[ByteHash, int]] = [(Blake2b(), 64), (HostBlake2b(), 64)]
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
            blake2b.digest(fnp.zeros(64, dtype=fnp.uint8))


# `blake2-kat.json` from the BLAKE2 reference suite (github.com/BLAKE2/BLAKE2,
# testvectors/), keyed BLAKE2b-512. The suite uses one key throughout —
# `bytes(range(64))`, the full 64-byte maximum — and walks the message length
# from 0 to 255 over `bytes(range(256))`. Extracted from the published file
# programmatically rather than typed: a hand-copied digest and a wrong
# implementation are indistinguishable from a failed compare, so the hex here
# was never in a human's hands.
#
# Six lengths, chosen for what each one is the only witness to:
#   0   — the key block is the ONLY block, so it carries the final flag
#   1   — key block interior, message block final
#   127 — the message block full but for one byte
#   128 — two whole blocks, the empty-tail degenerate shape
#   129 — the message spills to a third block
#   255 — a three-block message under the key, t past two blocks
_KAT_KEY = bytes(range(64))
_KAT_DATA = bytes(range(256))
_KEYED_KAT = {
    0: bytes.fromhex(
        "10ebb67700b1868efb4417987acf4690ae9d972fb7a590c2f02871799aaa4786"
        "b5e996e8f0f4eb981fc214b005f42d2ff4233499391653df7aefcbc13fc51568"
    ),
    1: bytes.fromhex(
        "961f6dd1e4dd30f63901690c512e78e4b45e4742ed197c3c5e45c549fd25f2e4"
        "187b0bc9fe30492b16b0d0bc4ef9b0f34c7003fac09a5ef1532e69430234cebd"
    ),
    127: bytes.fromhex(
        "76d2d819c92bce55fa8e092ab1bf9b9eab237a25267986cacf2b8ee14d214d73"
        "0dc9a5aa2d7b596e86a1fd8fa0804c77402d2fcd45083688b218b1cdfa0dcbcb"
    ),
    128: bytes.fromhex(
        "72065ee4dd91c2d8509fa1fc28a37c7fc9fa7d5b3f8ad3d0d7a25626b57b1b44"
        "788d4caf806290425f9890a3a2a35a905ab4b37acfd0da6e4517b2525c9651e4"
    ),
    129: bytes.fromhex(
        "64475dfe7600d7171bea0b394e27c9b00d8e74dd1e416a79473682ad3dfdbb70"
        "6631558055cfc8a40e07bd015a4540dcdea15883cbbf31412df1de1cd4152b91"
    ),
    255: bytes.fromhex(
        "142709d62e28fcccd0af97fad0f8465b971e82201dc51070faa0372aa43e9248"
        "4be1c1e73ba10906d5d1853db6a4106e0a7bf9800d373d6dee2d46d62ef2a461"
    ),
}

# BLAKE2b's salt and personalization fields are 16 bytes each (RFC 7693 §2.8).
# Distinguishable values, because the two fields are adjacent and the same
# width — the arrangement a transposition survives silently.
_SALT = b"\xa1" * 16
_PERSON = b"\xb2" * 16


def _kat_message(length: int) -> np.ndarray:
    return (
        np.frombuffer(_KAT_DATA[:length], dtype=np.uint8).reshape(1, length)
        if length
        else np.zeros((1, 0), dtype=np.uint8)
    )


class Blake2bKeyedVectorTest(parameterized.TestCase):
    """Keyed BLAKE2b against the reference suite, then against `hashlib`."""

    @parameterized.parameters(*sorted(_KEYED_KAT))
    def test_matches_the_reference_keyed_vectors(self, length: int) -> None:
        got = np.asarray(blake2b.digest(_kat_message(length), 64, _KAT_KEY))[0]
        self.assertEqual(bytes(got), _KEYED_KAT[length])

    def test_a_keyed_empty_message_is_not_the_unkeyed_one(self) -> None:
        # The case the whole key-block arrangement turns on. Unkeyed, an empty
        # message still runs one compression over an all-zero block at t = 0;
        # keyed, it runs one over the KEY block at t = 128 with the final flag
        # set. Two hashes of nothing, and an implementation that prepended the
        # key without letting it reach the counter would agree with the wrong
        # one of them.
        empty = np.zeros((1, 0), dtype=np.uint8)
        unkeyed = bytes(np.asarray(blake2b.digest(empty, 64))[0])
        keyed = bytes(np.asarray(blake2b.digest(empty, 64, _KAT_KEY))[0])
        self.assertNotEqual(keyed, unkeyed)
        self.assertEqual(keyed, _KEYED_KAT[0])

    @parameterized.parameters(1, 32, 64)
    def test_key_length_reaches_the_parameter_block(self, key_len: int) -> None:
        # `kk` is byte 1 of §2.8's block, so two keys of DIFFERENT length
        # cannot agree even where the padded key block is identical — which is
        # exactly what a 1-byte and a 2-byte key share once zero-padded to 128
        # bytes. Only `h0` separates them. (1, 3) throughout: one keyed aval.
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        key = bytes(range(key_len))
        got = np.asarray(blake2b.digest(rows, 32, key))[0]
        self.assertEqual(
            bytes(got), hashlib.blake2b(b"abc", digest_size=32, key=key).digest()
        )

    def test_two_keys_padding_to_the_same_block_differ(self) -> None:
        # The directed version of the case above: b"\x01" and b"\x01\x00"
        # produce byte-identical key BLOCKS and differ only in `kk`.
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        one = bytes(np.asarray(blake2b.digest(rows, 32, b"\x01"))[0])
        two = bytes(np.asarray(blake2b.digest(rows, 32, b"\x01\x00"))[0])
        self.assertNotEqual(one, two)


class Blake2bParameterBlockTest(parameterized.TestCase):
    """Salt and personalization at the seam — against `hashlib`, which takes
    both directly, and against each other."""

    @parameterized.named_parameters(
        ("salt_only", _SALT, b""),
        ("person_only", b"", _PERSON),
        ("both", _SALT, _PERSON),
        ("short_salt", b"\xa1", b""),
        ("zcash_style_person", b"", b"ZcashPH"),
    )
    def test_matches_hashlib(self, salt: bytes, person: bytes) -> None:
        # (1, 3) is the "abc" vector's aval, so the whole parameter sweep
        # rides one trace: salt and personalization move `h0`'s VALUE only.
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        got = np.asarray(blake2b.digest(rows, 32, b"", salt, person))[0]
        self.assertEqual(
            bytes(got),
            hashlib.blake2b(b"abc", digest_size=32, salt=salt, person=person).digest(),
        )

    def test_salt_and_person_are_not_interchangeable(self) -> None:
        # Adjacent fields of equal width: swapping them is the error this
        # catches, and no vector in the suite would.
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        forward = bytes(np.asarray(blake2b.digest(rows, 32, b"", _SALT, _PERSON))[0])
        swapped = bytes(np.asarray(blake2b.digest(rows, 32, b"", _PERSON, _SALT))[0])
        self.assertNotEqual(forward, swapped)

    def test_the_key_composes_with_salt_and_person(self) -> None:
        # All four §2.8 fields at once — the combination no single-purpose
        # vector covers, held to `hashlib`.
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        got = np.asarray(blake2b.digest(rows, 48, _KAT_KEY, _SALT, _PERSON))[0]
        self.assertEqual(
            bytes(got),
            hashlib.blake2b(
                b"abc", digest_size=48, key=_KAT_KEY, salt=_SALT, person=_PERSON
            ).digest(),
        )


class Blake2bKeyedRowTest(absltest.TestCase):
    """The keyed rows: the seam, value identity, and what they refuse."""

    def test_rows_satisfy_the_seam_and_agree(self) -> None:
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        device: ByteHash = Blake2bKeyed(_KAT_KEY, 32, salt=_SALT, person=_PERSON)
        host: ByteHash = HostBlake2bKeyed(_KAT_KEY, 32, salt=_SALT, person=_PERSON)
        self.assertIs(device.fusion_path, FusionPath.GENERIC)
        self.assertIs(host.fusion_path, FusionPath.HOST)
        np.testing.assert_array_equal(
            np.asarray(device.digest(rows)), np.asarray(host.digest(rows))
        )

    def test_the_key_rides_the_value_surface(self) -> None:
        # A key is part of WHICH hash this is, so two rows differing only in
        # the key must not compare equal — a row cached by identity would be
        # reused for the wrong key.
        a = Blake2bKeyed(b"one", 32)
        self.assertEqual(a, Blake2bKeyed(b"one", 32))
        self.assertEqual(hash(a), hash(Blake2bKeyed(b"one", 32)))
        for other in (
            Blake2bKeyed(b"two", 32),
            Blake2bKeyed(b"one", 64),
            Blake2bKeyed(b"one", 32, salt=_SALT),
            Blake2bKeyed(b"one", 32, person=_PERSON),
        ):
            self.assertNotEqual(a, other)

    def test_salt_and_person_ride_the_unkeyed_rows_value_surface_too(self) -> None:
        self.assertNotEqual(Blake2b(32), Blake2b(32, person=b"ZcashPH"))
        self.assertNotEqual(HostBlake2b(32), HostBlake2b(32, person=b"ZcashPH"))
        self.assertEqual(Blake2b(32, salt=_SALT), Blake2b(32, salt=_SALT))

    def test_an_empty_key_is_refused_rather_than_demoted(self) -> None:
        # Returning the unkeyed digest would hide a caller's bug: `kk` reaches
        # `h0`, so the two are different hashes and the caller asked for this
        # one.
        for cls in (Blake2bKeyed, HostBlake2bKeyed):
            with self.subTest(row=cls.__name__):
                with self.assertRaisesRegex(ValueError, "key must be non-empty"):
                    cls(b"", 32)

    def test_the_module_path_rejects_an_over_long_key(self) -> None:
        # `digest` is reachable without a row, so the bound has to hold there
        # too — and it has to be the STANDARD's bound. The key block is
        # zero-padded with `ljust`, which is a no-op above 128 bytes, so a
        # 65-byte key checked too late would surface as a broadcast shape
        # error instead of RFC 7693's `kk` range.
        msg = np.zeros((1, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, r"key_size must be in 0\.\.64"):
            blake2b.digest(msg, 32, b"x" * 65)

    def test_widths_are_checked_in_the_constructor(self) -> None:
        # At construction, where the caller can still choose — not at the
        # first `digest`, which is the rule the unkeyed rows already follow
        # for `digest_size`.
        for cls in (Blake2bKeyed, HostBlake2bKeyed):
            with self.subTest(row=cls.__name__):
                with self.assertRaisesRegex(ValueError, "salt"):
                    cls(b"k", 32, salt=b"x" * 17)
                with self.assertRaisesRegex(ValueError, "person"):
                    cls(b"k", 32, person=b"x" * 17)
                with self.assertRaisesRegex(ValueError, "key_size"):
                    cls(b"k" * 65, 32)


class Blake2bKeyedMarkerTest(absltest.TestCase):
    """The keyed path's lowering: the same marker, one block longer."""

    def test_a_keyed_digest_emits_the_same_single_marker(self) -> None:
        # The claim the module docstring makes — keying is a longer message
        # and a different `h0` VALUE, not a new marker and not a second one.
        msg = np.zeros((4, 127), dtype=np.uint8)
        txt = (
            frx.jit(functools.partial(blake2b.digest, digest_size=64, key=_KAT_KEY))
            .lower(msg)
            .as_text()
        )
        self.assertIn(blake2b.BLAKE2B_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_the_operand_abi_is_unchanged_with_the_key_block_folded_in(
        self,
    ) -> None:
        # Still four operands in the documented order; `msg` is 128 bytes
        # longer because the key block was prepended OUTSIDE the marker, and
        # the tail shrinks to match. An emitter reads exactly what it read
        # before.
        msg = fnp.asarray(np.zeros((4, 127), dtype=np.uint8))
        eqn = _composite(
            functools.partial(blake2b.digest, digest_size=64, key=_KAT_KEY), msg
        )
        self.assertEqual(eqn.params["name"], blake2b.BLAKE2B_MARKER)
        self.assertLen(eqn.invars, 4)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        # 128 (key block) + 127 = 255, padding to two blocks with a 1-byte tail.
        self.assertEqual(shapes, [(16,), (16,), (4, 255), (1,)])

    def test_a_new_key_does_not_retrace(self) -> None:
        # The property that separates this from `Blake3Keyed`: the key enters
        # through the message operand rather than as a captured constant, so
        # the zone's aval-keyed cache is blind to its VALUE. Three distinct
        # keys of one length at one message length must share one trace.
        msgs = fnp.asarray(_message(129))
        calls = [
            functools.partial(Blake2bKeyed(bytes([i]) * 32, 32).digest, msgs)
            for i in range(3)
        ]
        assert_single_trace(self, blake2b.blake2b_bytes, calls)


if __name__ == "__main__":
    absltest.main()

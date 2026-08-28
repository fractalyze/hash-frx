# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The BLAKE2s rows — the RFC vector, the hashlib differential, the marker,
and the seam; `blake2b_test`'s plan at the 32-bit parameters.

Values are held to the published record directly (RFC 7693 Appendix B's
worked "abc" example and the reference-suite empty-message vector) and to
`hashlib.blake2s` differentially across every padding boundary and digest
size — because the parameter block folds `digest_size` into the initial
state, agreement at a shorter length proves the length reached the IV rather
than a slice.

**The keyed, salted and personalized forms** are held to the reference
suite's own vectors (`blake2-kat.json`, the single 32-byte key
`bytes(range(32))`) at the block boundaries and to `hashlib`'s
`key=/salt=/person=` across the parameter surface — `blake2b_test` states why
those are two different checks rather than one. The cases that are genuinely
this family's, rather than the sibling's at half the width, are the ones the
64-byte block moves: the key block is 64 bytes here, so a keyed digest shifts
by one BLAKE2s block and the boundary lengths are its own.

The lowering assertions are the usual half that values cannot see: the digest
must emit exactly one composite carrying the registered name, version, and
the four-operand ABI with no captured constants — including the zero-length
tail at a block-multiple length, the ABI's one degenerate shape. Both backends
route it now, through the plugin's shared raw-bytes Merkle-Damgard envelope
rather than an emitter of this family's own, so recognition IS asserted —
against the compiled module, where a recognized marker becomes a `kCustom`
fusion and an unrecognized one silently inlines to the same bytes.

Every marker/traced case reuses an aval the differential sweep compiles, and
`h0`'s aval is uint32 [8] for every digest size, so the suite pays one
compile per (batch, length) and nothing more.
"""

from __future__ import annotations

import functools
import hashlib

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

from hash_frx import markers
from hash_frx.blake2s import blake2s
from hash_frx.blake2s.blake2s import (
    MAX_DIGEST_SIZE,
    Blake2s,
    Blake2sKeyed,
)
from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.testing.composite_eqn import (
    composite_attrs,
    composite_eqn,
    composite_shapes,
)
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.testing.marker_recognized import (
    assert_marker_recognized,
    emitted_composites,
)
from hash_frx.testing.oracle import oracle_digest

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


def _oracle_at(size: int, data: bytes) -> bytes:
    return hashlib.blake2s(data, digest_size=size).digest()


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


_HAS_BLAKE2S_EMITTER = blake2s._routes_to_dedicated_emitter()


class Blake2sMarkerTest(absltest.TestCase):
    def test_routing_is_the_pin_and_the_backend(self) -> None:
        # Both halves are pinned because they move together with the `frx>=`
        # floor, and both backends because the envelope is shared rather than
        # per-family — it arrived with its CPU and GPU arms at once.
        # One instance suffices: the path reads the two flags alone, and the
        # size sweeps elsewhere already construct every length.
        self.assertTrue(blake2s._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(blake2s._EMITTER_BACKENDS, ("cpu", "gpu"))
        self.assertIs(
            Blake2s().fusion_path, FusionPath.from_routing(_HAS_BLAKE2S_EMITTER)
        )

    def test_digest_emits_one_composite_with_the_digest_name(self) -> None:
        # The contract's unit: an absent or split marker still computes the
        # right bytes, so only the lowered module shows it. (4, 63) is the
        # differential sweep's aval, so the marker cases add no fresh trace.
        # The wire now carries the shared operation name, with BLAKE2s named by
        # the `primitive` attribute; the list length is the composite count.
        msg = np.zeros((4, 63), dtype=np.uint8)
        self.assertEqual(
            emitted_composites(blake2s.digest, msg), [markers.BYTES_DIGEST_MARKER]
        )

    def test_the_digest_compiles_to_a_custom_fusion(self) -> None:
        # What the routing flags CLAIM, checked against what the plugin does.
        # The bytes cannot show it — an unrecognized marker inlines and
        # computes the same digest — so this reads the compiled module, where
        # a recognized marker becomes a kCustom fusion named for the envelope.
        if not _HAS_BLAKE2S_EMITTER:
            self.skipTest(f"no BLAKE2s emitter on {frx.default_backend()}")
        # (4, 65) is the aval `Blake2sTracedTest` already drives through
        # `frx.jit(blake2s.digest)`, so the two share one compile — the only
        # aval here that does, since the differential sweep calls `digest`
        # eagerly and so compiles a different function object.
        msg = fnp.asarray(np.zeros((4, 65), dtype=np.uint8))
        assert_marker_recognized(
            self, "blake2s", blake2s.digest, msg, envelope_key="bytes_digest"
        )

    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        # The wire surface an emitter will read: name, version, and the
        # documented [h0, iv, msg, tail] operand order. Four invars exactly
        # is the captured-constants-free property — an array the body closed
        # over would be lifted in AHEAD of these, one per call site (the
        # operand-ABI rule in docs/reference/conventions.md); the counter
        # schedule and the SIGMA rows must enter as scalar literals and
        # trace-time tuples, never as lifted table arrays.
        msg = fnp.asarray(np.zeros((4, 63), dtype=np.uint8))
        eqn = composite_eqn(blake2s.digest, msg)
        self.assertEqual(eqn.params["name"], markers.BYTES_DIGEST_MARKER)
        self.assertEqual(eqn.params["version"], markers.BYTES_DIGEST_MARKER_VERSION)
        attrs = composite_attrs(eqn)
        self.assertEqual(attrs["primitive"], "blake2s")
        self.assertLen(eqn.invars, 4)
        shapes = composite_shapes(eqn)
        # L = 63: one block once padded, so the zero tail is 1 byte.
        self.assertEqual(shapes, [(8,), (8,), (4, 63), (1,)])

    def test_a_block_multiple_rides_an_empty_tail(self) -> None:
        # The ABI's degenerate shape: at L a multiple of 64 the zero tail is
        # empty, and the operand still rides — zero-length, never dropped
        # (the invar COUNT is the pinned surface; only shapes move with L).
        msg = fnp.asarray(np.zeros((4, 64), dtype=np.uint8))
        eqn = composite_eqn(blake2s.digest, msg)
        shapes = composite_shapes(eqn)
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
    """The device row against the seam, and against `hashlib`."""

    def test_impls_satisfy_the_seam(self) -> None:
        h = Blake2s()
        self.assertIsInstance(h, ByteHash)
        self.assertEqual(h.digest_size, MAX_DIGEST_SIZE)
        self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_paths_pin_the_substrate(self) -> None:
        # The routed path where the pin and backend carry the envelope, and
        # the traceability tie to the return type: the row returns an `Array`
        # and takes a tracer (`byte_hash.py`'s rule). The (1, 3) message rides
        # the "abc" vector's aval, so the plumbing check costs no fresh compile.
        msg = np.zeros((1, 3), dtype=np.uint8)
        device = Blake2s()
        self.assertIs(device.fusion_path, FusionPath.from_routing(_HAS_BLAKE2S_EMITTER))
        out = device.digest(msg)
        self.assertNotIsInstance(out, np.ndarray)
        self.assertIsInstance(out, Array)

    def test_device_matches_hashlib_across_sizes(self) -> None:
        # The out-of-tree differential at the seam, per digest size. (1, 3)
        # rides the vector aval — the size sweep costs no fresh compile, per
        # the trace pin above.
        msg = _message(3, seed=5)[:1]
        for size in _SIZES:
            with self.subTest(size=size):
                np.testing.assert_array_equal(
                    np.asarray(Blake2s(size).digest(msg)),
                    oracle_digest(
                        functools.partial(_oracle_at, size),
                        size,
                        msg,
                    ),
                )

    def test_digest_shape_and_dtype(self) -> None:
        out = np.asarray(Blake2s(20).digest(np.zeros((4, 1), dtype=np.uint8)))
        self.assertEqual(out.shape, (4, 20))
        self.assertEqual(out.dtype, np.uint8)

    def test_value_identity_rides_the_digest_size(self) -> None:
        # By value over the length, like the sibling's rows — the param-free
        # by-type form does not apply. Same size equal (and hash-equal: what
        # keeps the seam re-trace-safe as pytree aux), cross-size unequal,
        self.assertEqual(Blake2s(16), Blake2s(16))
        self.assertEqual(hash(Blake2s(16)), hash(Blake2s(16)))
        self.assertNotEqual(Blake2s(16), Blake2s(32))
        self.assertNotEqual(Blake2s(16), object())

    def test_out_of_range_length_is_refused_at_construction(self) -> None:
        for size in (0, MAX_DIGEST_SIZE + 1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    Blake2s(size)
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
        got = np.asarray(Blake2s().digest(fnp.zeros((0, 64), dtype=fnp.uint8)))
        self.assertEqual(got.shape, (0, 32))
        self.assertEqual(got.dtype, np.uint8)

    def test_a_1d_message_is_rejected_at_the_seam(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D uint8"):
            blake2s.digest(fnp.zeros(64, dtype=fnp.uint8))


# `blake2-kat.json` from the BLAKE2 reference suite, keyed BLAKE2s-256: one
# key throughout — `bytes(range(32))`, the 32-byte maximum — over messages
# `bytes(range(256))[:L]`. Extracted from the published file programmatically
# rather than typed, for the reason `blake2b_test` states.
#
# The lengths are this family's boundaries, not the sibling's: 0 (the key
# block is the only block and carries the final flag), 1, 63/64/65 (the
# one-block edge, the empty-tail case, the spill), 191 (a three-block message
# under the key block).
_KAT_KEY = bytes(range(32))
_KAT_DATA = bytes(range(256))
_KEYED_KAT = {
    0: bytes.fromhex(
        "48a8997da407876b3d79c0d92325ad3b89cbb754d86ab71aee047ad345fd2c49"
    ),
    1: bytes.fromhex(
        "40d15fee7c328830166ac3f918650f807e7e01e177258cdc0a39b11f598066f1"
    ),
    63: bytes.fromhex(
        "c65382513f07460da39833cb666c5ed82e61b9e998f4b0c4287cee56c3cc9bcd"
    ),
    64: bytes.fromhex(
        "8975b0577fd35566d750b362b0897a26c399136df07bababbde6203ff2954ed4"
    ),
    65: bytes.fromhex(
        "21fe0ceb0052be7fb0f004187cacd7de67fa6eb0938d927677f2398c132317a8"
    ),
    191: bytes.fromhex(
        "91a25ec0ec0d9a567f89c4bfe1a65a0e432d07064b4190e27dfb81901fd3139b"
    ),
}

# BLAKE2s' salt and personalization fields are 8 bytes each (§2.8) — half the
# sibling's. Distinguishable values, the two fields being adjacent and equal
# in width.
_SALT = b"\xa1" * 8
_PERSON = b"\xb2" * 8


def _kat_message(length: int) -> np.ndarray:
    return (
        np.frombuffer(_KAT_DATA[:length], dtype=np.uint8).reshape(1, length)
        if length
        else np.zeros((1, 0), dtype=np.uint8)
    )


class Blake2sKeyedVectorTest(parameterized.TestCase):
    @parameterized.parameters(*sorted(_KEYED_KAT))
    def test_matches_the_reference_keyed_vectors(self, length: int) -> None:
        got = np.asarray(blake2s.digest(_kat_message(length), 32, _KAT_KEY))[0]
        self.assertEqual(bytes(got), _KEYED_KAT[length])

    def test_a_keyed_empty_message_is_not_the_unkeyed_one(self) -> None:
        # Unkeyed: one compression over an all-zero block at t = 0. Keyed: one
        # over the 64-byte KEY block at t = 64 with the final flag set. An
        # implementation that prepended the key without letting it reach the
        # counter would agree with the wrong one.
        empty = np.zeros((1, 0), dtype=np.uint8)
        unkeyed = bytes(np.asarray(blake2s.digest(empty, 32))[0])
        keyed = bytes(np.asarray(blake2s.digest(empty, 32, _KAT_KEY))[0])
        self.assertNotEqual(keyed, unkeyed)
        self.assertEqual(keyed, _KEYED_KAT[0])

    def test_two_keys_padding_to_the_same_block_differ(self) -> None:
        # Byte-identical key BLOCKS once zero-padded to 64 bytes; only `kk` in
        # §2.8's block separates them.
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        one = bytes(np.asarray(blake2s.digest(rows, 32, b"\x01"))[0])
        two = bytes(np.asarray(blake2s.digest(rows, 32, b"\x01\x00"))[0])
        self.assertNotEqual(one, two)


class Blake2sParameterBlockTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("salt_only", _SALT, b""),
        ("person_only", b"", _PERSON),
        ("both", _SALT, _PERSON),
        ("short_salt", b"\xa1", b""),
    )
    def test_matches_hashlib(self, salt: bytes, person: bytes) -> None:
        # (1, 3) is the "abc" vector's aval: the parameter sweep moves `h0`'s
        # VALUE only and rides one trace.
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        got = np.asarray(blake2s.digest(rows, 32, b"", salt, person))[0]
        self.assertEqual(
            bytes(got),
            hashlib.blake2s(b"abc", digest_size=32, salt=salt, person=person).digest(),
        )

    def test_salt_and_person_are_not_interchangeable(self) -> None:
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        forward = bytes(np.asarray(blake2s.digest(rows, 32, b"", _SALT, _PERSON))[0])
        swapped = bytes(np.asarray(blake2s.digest(rows, 32, b"", _PERSON, _SALT))[0])
        self.assertNotEqual(forward, swapped)

    def test_the_key_composes_with_salt_and_person(self) -> None:
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        got = np.asarray(blake2s.digest(rows, 20, _KAT_KEY, _SALT, _PERSON))[0]
        self.assertEqual(
            bytes(got),
            hashlib.blake2s(
                b"abc", digest_size=20, key=_KAT_KEY, salt=_SALT, person=_PERSON
            ).digest(),
        )


class Blake2sKeyedRowTest(absltest.TestCase):
    def test_rows_satisfy_the_seam_and_agree(self) -> None:
        rows = np.frombuffer(b"abc", dtype=np.uint8).reshape(1, 3)
        device: ByteHash = Blake2sKeyed(_KAT_KEY, 32, salt=_SALT, person=_PERSON)
        self.assertIs(device.fusion_path, FusionPath.from_routing(_HAS_BLAKE2S_EMITTER))
        np.testing.assert_array_equal(
            np.asarray(device.digest(rows)),
            oracle_digest(
                lambda b: hashlib.blake2s(
                    b, digest_size=32, key=_KAT_KEY, salt=_SALT, person=_PERSON
                ).digest(),
                32,
                rows,
            ),
        )

    def test_the_key_rides_the_value_surface(self) -> None:
        a = Blake2sKeyed(b"one", 32)
        self.assertEqual(a, Blake2sKeyed(b"one", 32))
        self.assertEqual(hash(a), hash(Blake2sKeyed(b"one", 32)))
        for other in (
            Blake2sKeyed(b"two", 32),
            Blake2sKeyed(b"one", 16),
            Blake2sKeyed(b"one", 32, salt=_SALT),
            Blake2sKeyed(b"one", 32, person=_PERSON),
        ):
            self.assertNotEqual(a, other)

    def test_salt_and_person_ride_the_unkeyed_rows_value_surface_too(self) -> None:
        self.assertNotEqual(Blake2s(32), Blake2s(32, person=b"WGmac1"))
        self.assertEqual(Blake2s(32, salt=_SALT), Blake2s(32, salt=_SALT))

    def test_an_empty_key_is_refused_rather_than_demoted(self) -> None:
        with self.assertRaisesRegex(ValueError, "key must be non-empty"):
            Blake2sKeyed(b"", 32)

    def test_the_module_path_rejects_an_over_long_key(self) -> None:
        # The sibling's case at this width — `digest` must give RFC 7693's
        # `kk` range rather than a broadcast shape error from the key block.
        msg = np.zeros((1, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, r"key_size must be in 0\.\.32"):
            blake2s.digest(msg, 32, b"x" * 33)

    def test_widths_are_checked_in_the_constructor(self) -> None:
        with self.assertRaisesRegex(ValueError, "salt"):
            Blake2sKeyed(b"k", 32, salt=b"x" * 9)
        with self.assertRaisesRegex(ValueError, "person"):
            Blake2sKeyed(b"k", 32, person=b"x" * 9)
        with self.assertRaisesRegex(ValueError, "key_size"):
            Blake2sKeyed(b"k" * 33, 32)


class Blake2sKeyedMarkerTest(absltest.TestCase):
    def test_the_operand_abi_is_unchanged_with_the_key_block_folded_in(
        self,
    ) -> None:
        # One marker, four operands, the documented order — `msg` 64 bytes
        # longer because the key block was prepended outside the marker.
        msg = fnp.asarray(np.zeros((4, 63), dtype=np.uint8))
        eqn = composite_eqn(
            functools.partial(blake2s.digest, digest_size=32, key=_KAT_KEY), msg
        )
        self.assertEqual(eqn.params["name"], markers.BYTES_DIGEST_MARKER)
        self.assertLen(eqn.invars, 4)
        shapes = composite_shapes(eqn)
        # 64 (key block) + 63 = 127, padding to two blocks with a 1-byte tail.
        self.assertEqual(shapes, [(8,), (8,), (4, 127), (1,)])

    def test_a_new_key_does_not_retrace(self) -> None:
        # The sibling's property at this width: the key enters through the
        # message operand, so the zone's aval-keyed cache is blind to its
        # value and three distinct keys share one trace.
        msgs = fnp.asarray(_message(65))
        calls = [
            functools.partial(Blake2sKeyed(bytes([i]) * 16, 32).digest, msgs)
            for i in range(3)
        ]
        assert_single_trace(self, blake2s.blake2s_bytes, calls)


if __name__ == "__main__":
    absltest.main()

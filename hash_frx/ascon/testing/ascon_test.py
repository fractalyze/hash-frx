# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ascon-Hash256 — KAT values, the initial state, the marker, and the seam.

Values are held to the SP 800-232 KAT directly (the vectors `reference_test`
anchors across three independent sources) and to the oracle differentially at
lengths the transcription does not carry, so agreement means agreement with
Ascon-Hash256 rather than with a second copy of one misreading. The round
body itself is Ascon-p's, so its cases — the bitsliced S-box circuit against
Table 6, the reference agreement, the operand ABI — live with the permutation
in `permutation_test`. The precomputed initial state is pinned the other way
around — the device module transcribes Table 12, the
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
from hash_frx.ascon.ascon import (
    ASCON_HASH256_DIGEST_SIZE,
    MAX_CUSTOMIZATION_BYTES,
    AsconCxof128,
    AsconHash256,
    AsconXof128,
)
from hash_frx.ascon.testing.reference import (
    CXOF_INITIAL_STATE,
    CXOF_KAT_VECTORS,
    INITIAL_STATE,
    KAT_VECTORS,
    XOF_INITIAL_STATE,
    XOF_KAT_VECTORS,
    ascon_cxof128,
    ascon_hash256,
    ascon_xof128,
    customization_prefix,
)
from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.testing.oracle import oracle_digest
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
        # against the reference oracle, across every rate boundary and a
        # batch — not one convenient length.
        msgs = _message(length)
        np.testing.assert_array_equal(
            np.asarray(AsconHash256().digest(msgs)),
            oracle_digest(ascon_hash256, 32, msgs),
        )


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
    """The `ByteHash` row, against the seam."""

    def test_impls_satisfy_the_seam(self) -> None:
        h = AsconHash256()
        self.assertIsInstance(h, ByteHash)
        self.assertEqual(h.digest_size, 32)
        self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_paths_pin_the_substrate(self) -> None:
        # GENERIC (pre-emitter, every backend), and the traceability tie to
        # the return type: the row returns an `Array` and takes a tracer
        # (`byte_hash.py`'s rule).
        # A (1, 1) message: the KAT sweep covers that aval, so the plumbing
        # check adds no distinct compile of the digest body.
        msg = np.zeros((1, 1), dtype=np.uint8)
        device = AsconHash256()
        self.assertIs(device.fusion_path, FusionPath.GENERIC)
        out = device.digest(msg)
        self.assertNotIsInstance(out, np.ndarray)
        self.assertIsInstance(out, Array)

    def test_digest_shape_and_dtype(self) -> None:
        # (4, 1) rides the differential sweep's aval — no distinct compile.
        out = np.asarray(AsconHash256().digest(np.zeros((4, 1), dtype=np.uint8)))
        self.assertEqual(out.shape, (4, 32))
        self.assertEqual(out.dtype, np.uint8)

    def test_value_identity_is_by_type(self) -> None:
        # Param-free, so every instance of a type is equal and hashes alike —
        # what keeps the seam re-trace-safe as pytree aux.
        self.assertEqual(AsconHash256(), AsconHash256())
        self.assertEqual(hash(AsconHash256()), hash(AsconHash256()))


class EmptyBatchTest(absltest.TestCase):
    """A zero-row batch is a valid batch (#211): every row returns
    uint8 [0, digest_size] instead of failing in a block-count reshape."""

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        got = np.asarray(AsconHash256().digest(fnp.zeros((0, 64), dtype=fnp.uint8)))
        self.assertEqual(got.shape, (0, 32))
        self.assertEqual(got.dtype, np.uint8)


class AsconXof128KatTest(parameterized.TestCase):
    """Ascon-XOF128 (§5.2) — the SP's own vectors, then the oracle differentially."""

    @parameterized.parameters(*((len(m), m, d) for m, d in XOF_KAT_VECTORS))
    def test_matches_the_reference_implementation_kat(
        self, _length: int, msg: bytes, out_hex: str
    ) -> None:
        out_size = len(out_hex) // 2
        batch = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = np.asarray(ascon.xof128(batch, out_size))
        self.assertEqual(got.shape, (1, out_size))
        self.assertEqual(bytes(got[0]).hex(), out_hex)

    @parameterized.parameters(*_LENGTHS)
    def test_device_and_oracle_agree(self, length: int) -> None:
        # Lengths the KAT transcription does not carry, at an output size that
        # is NOT a rate multiple so the truncation runs.
        msg = _message(length)
        got = np.asarray(ascon.xof128(msg, 37))
        for row in range(msg.shape[0]):
            self.assertEqual(
                bytes(got[row]).hex(), ascon_xof128(bytes(msg[row]), 37).hex()
            )

    def test_a_short_read_is_a_prefix_of_a_long_one(self) -> None:
        # The XOF property on the device path: fewer bytes asked for must not
        # be different bytes. `output_size` is a static argname, so every entry
        # here is a whole-region compile — two are enough to cross the 8-byte
        # rate boundary the trim sits on (7 reads one block and trims, 33 reads
        # five and trims), and the oracle carries the dense sweep
        # (`reference_test.test_a_short_read_is_a_prefix_of_a_long_one`).
        msg = _message(20)
        full = np.asarray(ascon.xof128(msg, 64))
        for n in (7, 33):
            np.testing.assert_array_equal(np.asarray(ascon.xof128(msg, n)), full[:, :n])

    def test_the_xof_is_not_the_hash_at_the_same_length(self) -> None:
        # Both squeeze four 8-byte blocks from the same absorb; only the IV
        # differs, and it must.
        msg = _message(8)
        self.assertFalse(
            np.array_equal(
                np.asarray(ascon.xof128(msg, ASCON_HASH256_DIGEST_SIZE)),
                np.asarray(ascon.digest(msg)),
            )
        )

    def test_the_initial_state_meets_the_oracle_derivation(self) -> None:
        # The device module transcribes the precomputed XOF state; the oracle
        # derives Ascon-p[12](IV ‖ 0^256) from the documented IV. Meeting here
        # pins the transcription against an independent derivation, exactly as
        # `InitialStateTest` does for the hash.
        derived = np.array([split(w) for w in XOF_INITIAL_STATE], dtype=np.uint32)
        np.testing.assert_array_equal(ascon._XOF128_INITIAL_STATE, derived)


class AsconXof128MarkerTest(absltest.TestCase):
    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        msg = fnp.asarray(_message(9))
        eqn = _composite(lambda m: ascon.xof128(m, 40), msg)
        self.assertEqual(eqn.params["name"], ascon.ASCON_XOF128_MARKER)
        self.assertEqual(eqn.params["version"], ascon.ASCON_XOF128_MARKER_VERSION)
        # The hash's three-operand ABI, with no captured constants: a
        # host-materialised array closed over by the body would be lifted into
        # an unnamed operand ahead of these.
        self.assertLen(eqn.invars, 3)
        self.assertEqual(eqn.invars[0].aval.shape, (5, 2))
        self.assertEqual(eqn.invars[1].aval.shape, msg.shape)

    def test_the_output_length_rides_as_an_attribute(self) -> None:
        # It fixes the region's SHAPE — the squeeze count and the result width —
        # which is what an attribute is for, where an operand determines a
        # value. An emitter reads it rather than inferring the squeeze count.
        eqn = _composite(lambda m: ascon.xof128(m, 40), fnp.asarray(_message(9)))
        attrs = {key: leaves[0] for key, leaves, _ in eqn.params["attributes"]}
        self.assertEqual(attrs, {"output_size": 40})

    def test_two_output_lengths_are_two_regions(self) -> None:
        msg = fnp.asarray(_message(9))
        self.assertNotEqual(
            _composite(lambda m: ascon.xof128(m, 32), msg).params["attributes"],
            _composite(lambda m: ascon.xof128(m, 64), msg).params["attributes"],
        )


class AsconXof128ByteHashTest(absltest.TestCase):
    """What the shared row sweeps do NOT cover.

    The row is registered in `testing.rows.ALL_ROWS` and in
    `fusion_path_test`'s matrix, so seam conformance, the `__eq__`/`__hash__`
    parameter law and the (row, backend) fusion-path cell are asserted there
    against every other row rather than restated here. What is left is the part
    that is specific to an extendable-output hash.
    """

    def test_digest_shape_and_dtype(self) -> None:
        # An output length that is NOT a multiple of the 8-byte rate, so the
        # squeeze overshoots and the trim runs.
        got = AsconXof128(37).digest(_message(5))
        self.assertEqual(got.shape, (4, 37))
        self.assertEqual(got.dtype, fnp.uint8)

    def test_rejects_a_zero_length_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_size"):
            AsconXof128(0)
        with self.assertRaisesRegex(ValueError, "output_size"):
            ascon.xof128(_message(1), 0)

    def test_jit_matches_eager(self) -> None:
        msg = fnp.asarray(_message(20))
        row = AsconXof128(40)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(row.digest)(msg)), np.asarray(row.digest(msg))
        )


class AsconCxof128KatTest(parameterized.TestCase):
    """Ascon-CXOF128 (§5.3) — the published vectors, then the oracle
    differentially on both axes."""

    @parameterized.parameters(
        *((len(m), len(z), m, z, d) for m, z, d in CXOF_KAT_VECTORS)
    )
    def test_matches_the_reference_implementation_kat(
        self, _mlen: int, _zlen: int, msg: bytes, customization: bytes, out_hex: str
    ) -> None:
        out_size = len(out_hex) // 2
        batch = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = np.asarray(ascon.cxof128(batch, customization, out_size))
        self.assertEqual(got.shape, (1, out_size))
        self.assertEqual(bytes(got[0]).hex(), out_hex)

    @parameterized.parameters(*_LENGTHS)
    def test_device_and_oracle_agree_over_message_length(self, length: int) -> None:
        # Message lengths the vectors do not carry, at an output size that is
        # NOT a rate multiple so the truncation runs.
        msg = _message(length)
        got = np.asarray(ascon.cxof128(msg, b"context", 37))
        for row in range(msg.shape[0]):
            self.assertEqual(
                bytes(got[row]).hex(),
                ascon_cxof128(bytes(msg[row]), b"context", 37).hex(),
            )

    @parameterized.parameters(0, 1, 7, 8, 9, 15, 16, 31, MAX_CUSTOMIZATION_BYTES)
    def test_device_and_oracle_agree_over_customization_length(self, zlen: int) -> None:
        # **The second axis, which is this row's own.** The prefix crosses the
        # 8-byte rate here rather than in the message: 7/8/9 are the one-block
        # edge — an 8-byte customization gains a whole padding block, the same
        # way a message does — and 256 is the §5.3 cap, whose prefix is 34
        # blocks and which nothing else exercises.
        customization = bytes(range(0x10, 0x10 + zlen)) if zlen <= 0xEF else bytes(zlen)
        msg = _message(20)
        got = np.asarray(ascon.cxof128(msg, customization, 32))
        for row in range(msg.shape[0]):
            self.assertEqual(
                bytes(got[row]).hex(),
                ascon_cxof128(bytes(msg[row]), customization, 32).hex(),
            )

    def test_an_empty_customization_is_not_the_plain_xof(self) -> None:
        # The IVs differ — version 4 against version 3 — so CXOF128 with no
        # customization is a DIFFERENT hash from XOF128, not the same one with
        # a setting turned off. This is exactly what folding CXOF into
        # `AsconXof128` as an optional keyword would have gotten wrong, and it
        # is wrong at every input.
        msg = _message(8)
        self.assertFalse(
            np.array_equal(
                np.asarray(ascon.cxof128(msg, b"", 32)),
                np.asarray(ascon.xof128(msg, 32)),
            )
        )

    def test_the_customization_changes_the_digest_at_one_length(self) -> None:
        # Same length, so the block count and the padding are identical and
        # only the absorbed bytes differ — a prefix that dropped the string and
        # kept its length field would pass every shape assertion and fail here.
        msg = _message(20)
        self.assertFalse(
            np.array_equal(
                np.asarray(ascon.cxof128(msg, b"aaaa", 32)),
                np.asarray(ascon.cxof128(msg, b"aaab", 32)),
            )
        )

    def test_a_short_read_is_a_prefix_of_a_long_one(self) -> None:
        msg = _message(20)
        full = np.asarray(ascon.cxof128(msg, b"z", 64))
        for n in (7, 33):
            np.testing.assert_array_equal(
                np.asarray(ascon.cxof128(msg, b"z", n)), full[:, :n]
            )

    def test_the_initial_state_meets_the_oracle_derivation(self) -> None:
        # The device module transcribes the precomputed CXOF state; the oracle
        # derives Ascon-p[12](IV ‖ 0^256) from the Table 13 assembly. Meeting
        # here pins the transcription against an independent derivation, as it
        # does for the hash and the XOF.
        derived = np.array([split(w) for w in CXOF_INITIAL_STATE], dtype=np.uint32)
        np.testing.assert_array_equal(ascon._CXOF128_INITIAL_STATE, derived)

    def test_the_device_prefix_meets_the_oracle_prefix(self) -> None:
        # Both build `Z0 ‖ Z ‖ pad`, one through `SpongePad` and one through a
        # literal pad; they must agree byte for byte at every boundary.
        for zlen in (0, 1, 7, 8, 9, 16, MAX_CUSTOMIZATION_BYTES):
            z = bytes(range(zlen)) if zlen < 256 else bytes(zlen)
            self.assertEqual(
                bytes(ascon._customization_prefix(z)), customization_prefix(z)
            )


class AsconCxof128MarkerTest(absltest.TestCase):
    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        msg = fnp.asarray(_message(9))
        eqn = _composite(lambda m: ascon.cxof128(m, b"ctx", 40), msg)
        self.assertEqual(eqn.params["name"], ascon.ASCON_CXOF128_MARKER)
        self.assertEqual(eqn.params["version"], ascon.ASCON_CXOF128_MARKER_VERSION)
        # FOUR operands, not the other rows' three: the customization blocks
        # sit between the state and the message. Captured constants would be
        # lifted in ahead of these, so the count is what says there are none.
        self.assertLen(eqn.invars, 4)
        self.assertEqual(eqn.invars[0].aval.shape, (5, 2))
        self.assertEqual(eqn.invars[1].aval.shape, (len(b"ctx") + 8 + 5,))
        self.assertEqual(eqn.invars[2].aval.shape, msg.shape)

    def test_the_prefix_shape_follows_the_rate_and_not_the_string(self) -> None:
        # **A customization's LENGTH does not determine the prefix shape; the
        # block it pads to does.** One and two bytes both ride in the same
        # 16-byte prefix — Z0's block plus one padded block — and only crossing
        # the 8-byte rate adds a block. So an emitter reading the operand shape
        # learns the block count and nothing about the string, which is the
        # same posture `tail` already has.
        msg = fnp.asarray(_message(9))

        def prefix_shape(customization: bytes) -> tuple[int, ...]:
            eqn = _composite(lambda m: ascon.cxof128(m, customization, 32), msg)
            return tuple(eqn.invars[1].aval.shape)

        self.assertEqual(prefix_shape(b"a"), prefix_shape(b"ab"))
        self.assertEqual(prefix_shape(b"a"), (16,))
        self.assertEqual(prefix_shape(bytes(7)), (16,))
        self.assertEqual(prefix_shape(bytes(8)), (24,))

    def test_two_customizations_of_one_length_are_two_traces(self) -> None:
        # The customization is a STATIC argument, so it is baked rather than
        # passed — two of the same length share every shape in the region and
        # must still be two compiled programs. The digests are what shows it,
        # since the aval cannot.
        msg = fnp.asarray(_message(9))
        self.assertFalse(
            np.array_equal(
                np.asarray(ascon.cxof128(msg, b"aaaa", 32)),
                np.asarray(ascon.cxof128(msg, b"bbbb", 32)),
            )
        )


class AsconCxof128ByteHashTest(absltest.TestCase):
    """What the shared row sweeps do NOT cover — the row is in
    `testing.rows.ALL_ROWS` and `fusion_path_test`'s matrix, so seam
    conformance, the parameter law and the fusion-path cell are asserted
    there."""

    def test_digest_shape_and_dtype(self) -> None:
        got = AsconCxof128(b"ctx", 37).digest(_message(5))
        self.assertEqual(got.shape, (4, 37))
        self.assertEqual(got.dtype, fnp.uint8)

    def test_rejects_a_zero_length_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_size"):
            AsconCxof128(b"", 0)
        with self.assertRaisesRegex(ValueError, "output_size"):
            ascon.cxof128(_message(1), b"", 0)

    def test_rejects_a_customization_over_the_cap(self) -> None:
        AsconCxof128(bytes(MAX_CUSTOMIZATION_BYTES), 32)
        with self.assertRaisesRegex(ValueError, "at most"):
            AsconCxof128(bytes(MAX_CUSTOMIZATION_BYTES + 1), 32)

    def test_the_customization_is_in_the_value_surface(self) -> None:
        # Two rows differing ONLY in the customization must not compare equal:
        # equal rows share a compiled trace, so a customization left out of
        # `_parameters` serves one hash's executable for another's.
        self.assertNotEqual(AsconCxof128(b"a", 32), AsconCxof128(b"b", 32))
        self.assertEqual(AsconCxof128(b"a", 32), AsconCxof128(b"a", 32))

    def test_jit_matches_eager(self) -> None:
        msg = fnp.asarray(_message(20))
        row = AsconCxof128(b"ctx", 40)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(row.digest)(msg)), np.asarray(row.digest(msg))
        )


if __name__ == "__main__":
    absltest.main()

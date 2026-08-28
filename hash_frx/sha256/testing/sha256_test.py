# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHA-256 byte-hash — byte-match against the universal reference `hashlib.sha256`.

Agnostic golden: `hashlib` is the FIPS 180-4 reference, named by no consumer. The
lengths exercise every padding boundary — empty, sub-block, the 55/56 one-vs-two
block transition (where the 8-byte length field forces a second block), exact
block multiples, and a multi-block message.
"""

from __future__ import annotations

import functools
import hashlib
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

from hash_frx.byte_hash import ByteHash, capacity
from hash_frx.fusion import FusionPath
from hash_frx.sha256 import sha256
from hash_frx.sha256.sha256 import Sha224, Sha256
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.testing.marker_recognized import (
    assert_marker_recognized,
    emitted_composites,
)
from hash_frx.testing.oracle import oracle_digest

# Padding-boundary lengths: 0/1 (empty + tiny), 55/56 (the one-block/two-block
# cutoff), 63/64 (block edge), 119/120 (multi-block).
_LENGTHS = (0, 1, 55, 56, 63, 64, 119, 120)


class Sha256Test(parameterized.TestCase):
    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        got = bytes(np.asarray(sha256.digest(msg[None, :]))[0])
        self.assertEqual(got, hashlib.sha256(bytes(msg)).digest())

    def test_batched_equals_per_row(self) -> None:
        # One data-parallel call over a stack of equal-length messages must equal
        # the per-message hashlib digests, in order.
        length = 64
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(7, length), dtype=np.uint8)
        got = np.asarray(sha256.digest(batch))
        for i in range(batch.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.sha256(bytes(batch[i])).digest())

    @parameterized.parameters(*_LENGTHS)
    def test_marked_equals_inline(self, length: int) -> None:
        # The hash_frx.digest.sha256 marker only tags the region; with no
        # dedicated emitter wired it inlines its decomposition, so the marked
        # digest must byte-equal the unmarked compression at every padding
        # boundary.
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        blocks = sha256._padded_words(fnp.asarray(msg[None, :]))
        marked = np.asarray(sha256.sha256_merkle_damgard(sha256.INITIAL_STATE, blocks))
        state = fnp.broadcast_to(sha256.INITIAL_STATE, (1, 8))
        inline = np.asarray(sha256.serialize_digest(sha256.compress(state, blocks)))
        np.testing.assert_array_equal(marked, inline)

    def test_emits_single_composite_marker(self) -> None:
        # digest lowers to exactly one stablehlo.composite, name-routed to the
        # dedicated hash_frx.digest.sha256 emitter (parallel to
        # hash_frx.perm.poseidon2).
        blocks = sha256._padded_words(
            fnp.asarray(np.arange(64, dtype=np.uint8))[None, :]
        )
        fn = functools.partial(sha256.sha256_merkle_damgard, sha256.INITIAL_STATE)
        txt = frx.jit(fn).lower(blocks).as_text()
        self.assertIn(sha256.SHA256_MARKER, txt)
        self.assertEqual(txt.count("stablehlo.composite"), 1)

    def test_serialize_deserialize_roundtrip(self) -> None:
        # deserialize_digest inverts serialize_digest, so unpacking a digest
        # recovers the exact midstate a stream resumes from.
        rng = np.random.default_rng(0)
        state = fnp.asarray(rng.integers(0, 2**32, (3, 8), np.int64).astype(np.uint32))
        back = sha256.deserialize_digest(sha256.serialize_digest(state))
        np.testing.assert_array_equal(np.asarray(back), np.asarray(state))

    @parameterized.parameters(1, 2, 3)
    def test_merkle_damgard_resumes_from_midstate(self, split: int) -> None:
        # From a non-IV midstate the compression resumes: hashing a 4-block
        # message in two chained halves must equal one pass over all 4.
        blocks = fnp.asarray(
            np.random.default_rng(split)
            .integers(0, 2**32, (1, 4, 16), np.int64)
            .astype(np.uint32)
        )
        whole = sha256.sha256_merkle_damgard(sha256.INITIAL_STATE, blocks)
        mid = sha256.deserialize_digest(
            sha256.sha256_merkle_damgard(sha256.INITIAL_STATE, blocks[:, :split])
        )[0]
        resumed = sha256.sha256_merkle_damgard(mid, blocks[:, split:])
        np.testing.assert_array_equal(np.asarray(whole), np.asarray(resumed))

    def test_compress_explicit_k_matches_default(self) -> None:
        # Threading the round-constant table as an explicit `k` operand (what the
        # marked region does) matches the module-default `_Kd`.
        blocks = sha256._padded_words(
            fnp.asarray(np.arange(80, dtype=np.uint8))[None, :]
        )
        state = fnp.broadcast_to(sha256.INITIAL_STATE, (1, 8))
        default = sha256.compress(state, blocks)
        explicit = sha256.compress(state, blocks, sha256._Kd)
        np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))


class Sha256TracedTest(parameterized.TestCase):
    """`digest` inside a traced region, at every padding boundary.

    The point of the padding being a function of the length: a consumer can hash
    inside its own `@jit` without reaching past the seam for the compression, so
    a scheme built on `ByteHash` does not have to name SHA-256 to be traceable.
    Byte-equality with the eager path is the whole claim — a traced padding that
    is subtly different produces a self-consistent wrong hash.
    """

    @parameterized.parameters(*_LENGTHS)
    def test_jit_matches_eager(self, length: int) -> None:
        msg = (np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A))[None, :]
        np.testing.assert_array_equal(
            np.asarray(frx.jit(sha256.digest)(msg)), np.asarray(sha256.digest(msg))
        )

    @parameterized.parameters(*_LENGTHS)
    def test_jit_matches_hashlib(self, length: int) -> None:
        # Against the reference rather than against ourselves: the eager path
        # agreeing with a traced one that shares its bug would prove nothing.
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0xA5)
        got = np.asarray(frx.jit(sha256.digest)(msg[None, :]))[0]
        self.assertEqual(bytes(got), hashlib.sha256(bytes(msg)).digest())

    def test_vmap_matches_the_batch_axis(self) -> None:
        # `digest` already takes the batch, so mapping a one-row call over a stack
        # must reproduce it — which is what a consumer gets for free when its own
        # vmap encloses the hash.
        batch = np.random.default_rng(3).integers(0, 256, (5, 70), dtype=np.uint8)
        mapped = frx.vmap(lambda row: sha256.digest(row[None, :])[0])(batch)
        np.testing.assert_array_equal(
            np.asarray(mapped), np.asarray(sha256.digest(batch))
        )

    def test_the_seam_carries_it(self) -> None:
        # Through `ByteHash.digest` rather than the module function: the consumer
        # holds the seam, and that is the call that has to survive the tracer.
        hasher: ByteHash = sha256.Sha256()
        msg = np.random.default_rng(4).integers(0, 256, (3, 100), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(hasher.digest)(msg)), np.asarray(hasher.digest(msg))
        )


class Sha256ByteHashTest(parameterized.TestCase):
    """The `ByteHash` row, against the seam and against `hashlib`.

    `byte_hash_test.py` stays seam-only — a double, so it runs on a branch where
    no concrete hash exists — which leaves the real class untested by it. This
    is the other half of that split: the assertions that need `Sha256` itself
    and are tautologies against a double.
    """

    def test_impls_satisfy_the_seam(self) -> None:
        h = sha256.Sha256()
        self.assertIsInstance(h, ByteHash)
        self.assertEqual(h.digest_size, 32)
        self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_flag_pins_the_substrate(self) -> None:
        # The only assertion in the repo that says WHICH implementation is the
        # device one. A consumer reads this at construction time to branch — see
        # zorch's byte_transcript, whose grind window is a wide nonce sweep on a
        # fused hash and a single nonce otherwise — so a silent flip here would
        # change a proof-of-work strategy rather than fail a hash.
        self.assertIs(sha256.Sha256().fusion_path.is_one_kernel, True)

    @parameterized.parameters(*_LENGTHS)
    def test_device_matches_hashlib(self, length: int) -> None:
        # `hashlib` is the out-of-tree oracle, so this is evidence about the
        # implementation rather than about one reading of the spec applied
        # twice — across every padding boundary rather than one convenient
        # length.
        rng = np.random.default_rng(length + 1)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(sha256.Sha256().digest(msgs)),
            oracle_digest(lambda b: hashlib.sha256(b).digest(), 32, msgs),
        )

    def test_value_identity_is_by_type(self) -> None:
        # Param-free, so every instance of a type is equal and hashes alike — what
        # keeps the seam re-trace-safe as pytree aux.
        self.assertEqual(sha256.Sha256(), sha256.Sha256())
        self.assertEqual(hash(sha256.Sha256()), hash(sha256.Sha256()))

    def test_marker_is_recognized_by_the_pinned_toolchain(self) -> None:
        blocks = sha256._padded_words(
            fnp.asarray(np.arange(64, dtype=np.uint8))[None, :]
        )
        fn = functools.partial(sha256.sha256_merkle_damgard, sha256.INITIAL_STATE)
        # `md_digest` is the plugin's generic words-in Merkle-Damgard envelope.
        # SHA-256 had its own routing key until the envelope landed and took
        # every words-in family; both are accepted because which one claims the
        # marker depends on the pinned plugin, not on this repo.
        assert_marker_recognized(self, "sha256", fn, blocks, envelope_key="md_digest")


# NIST CAVP `shabytetestvectors`, `SHA224ShortMsg.rsp` — extracted from the .rsp
# rather than typed. What they add over the `hashlib` sweeps, which already
# cover these lengths and more, is independence from `hashlib`: the sweeps say
# this tree agrees with CPython's SHA-224, and these say both agree with NIST.
# All 65 rows of the file were checked while extracting, so the six kept are a
# readable subset of a full agreement rather than the extent of what was
# verified. The lengths are `_LENGTHS`' padding boundaries, 56 and 64 being the
# ones where the 0x80 byte plus the 8-byte length field force a further block.
_CAVP_224 = (
    ("len_0", "", "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f"),
    ("len_1", "84", "3cd36921df5d6963e73739cf4d20211e2d8877c19cff087ade9d0e3a"),
    (
        "len_55",
        "445e8698eeb8accbaac4ffa7d934fffd16014a430ef70f3a9174c6cfe96d1e3f"
        "6ab1377f4a7212dbb30146dd17d9f470c4dffc45b8e871",
        "4c7ae028c0fe61f2a9cada61fae30685b77f04c6442576e912af9fa6",
    ),
    (
        "len_56",
        "52839f2f0853a30df14ec897a1914c685c1ac21470d00654c8c37663bfb65fa7"
        "32dbb694d9dd09ced723b48d8f545846ba168988b61cc724",
        "2f755a57674b49d5c25cb37348f35b6fd2de2552c749f2645ba63d20",
    ),
    (
        "len_63",
        "716944de41710c29b659be10480bb25a351a39e577ee30e8f422d57cf62ad95b"
        "da39b6e70c61426e33fd84aca84cc7912d5eee45dc34076a5d2323a15c7964",
        "61645ac748db567ac862796b8d06a47afebfa2e1783d5c5f3bcd81e2",
    ),
    (
        "len_64",
        "a3310ba064be2e14ad32276e18cd0310c933a6e650c3c754d0243c6c61207865"
        "b4b65248f66a08edf6e0832689a9dc3a2e5d2095eeea50bd862bac88c8bd318d",
        "b2a5586d9cbf0baa999157b4af06d88ae08d7c9faab4bc1a96829d65",
    ),
)


class Sha224Test(parameterized.TestCase):
    """SHA-224 — the §5.3.2 IV row over SHA-256's chain.

    A wrong IV here is self-consistent: it would pass every structural
    assertion in this file and produce a hash that is simply not SHA-224. The
    differential sweep is what catches that, and the CAVP vectors are what make
    the sweep's reference NIST rather than only CPython.
    """

    @parameterized.named_parameters(*_CAVP_224)
    def test_matches_the_cavp_vector(self, msg_hex: str, digest_hex: str) -> None:
        msg = bytes.fromhex(msg_hex)
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = bytes(np.asarray(sha256.sha224_digest(rows))[0])
        self.assertEqual(got.hex(), digest_hex)
        # Anchors the sweeps' reference to the same record.
        self.assertEqual(hashlib.sha224(msg).hexdigest(), digest_hex)

    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        got = bytes(np.asarray(sha256.sha224_digest(msg[None, :]))[0])
        self.assertEqual(got, hashlib.sha224(bytes(msg)).digest())

    def test_rides_the_sha256_marker_with_truncation_outside(self) -> None:
        # One composite, the blocks marker, whose output is the FULL [B, 32]
        # serialized state — the truncation is the caller's slice outside it,
        # so the wire contract a recognizer reads is byte-for-byte SHA-256's.
        msg = fnp.asarray(np.zeros((1, 64), dtype=np.uint8))
        self.assertEqual(
            emitted_composites(sha256.sha224_digest, msg), [sha256.SHA256_MARKER]
        )
        self.assertEqual(np.asarray(sha256.sha224_digest(msg)).shape, (1, 28))

    def test_never_takes_the_whole_message_marker(self) -> None:
        # `sha256_bytes` carries SHA-256's own IV as operand 0 of its region, so
        # the whole-message form cannot serve a variant. `digest` switches onto
        # it wherever the recognizer exists; this asserts the variant does not
        # follow, which a sweep run on a backend without that recognizer would
        # never notice.
        msg = np.zeros((1, 64), dtype=np.uint8)
        with mock.patch.object(sha256, "_routes_to_bytes_marker", lambda: True):
            self.assertEqual(
                emitted_composites(sha256.sha224_digest, msg), [sha256.SHA256_MARKER]
            )

    def test_one_chain_trace_serves_the_variant_and_the_parent(self) -> None:
        # A variant is DATA (a different h0 value on the same [8] aval), not
        # ABI, so SHA-256's and SHA-224's chains share one trace per shape.
        msgs = fnp.asarray(np.zeros((1, 64), dtype=np.uint8))
        with mock.patch.object(sha256, "_routes_to_bytes_marker", lambda: False):
            assert_single_trace(
                self,
                sha256.sha256_merkle_damgard,
                [
                    functools.partial(sha256.digest, msgs),
                    functools.partial(sha256.sha224_digest, msgs),
                ],
            )


class Sha224ByteHashTest(parameterized.TestCase):
    """The `Sha224` row against the seam, `hashlib`, and its parent."""

    def test_impls_satisfy_the_seam(self) -> None:
        h = Sha224()
        self.assertIsInstance(h, ByteHash)
        self.assertEqual(h.digest_size, 28)
        self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_path_is_inherited_from_the_parent(self) -> None:
        # An IV row over the same marker, so it routes wherever SHA-256 routes
        # rather than having a path of its own.
        self.assertIs(Sha224().fusion_path, Sha256().fusion_path)

    @parameterized.parameters(*_LENGTHS)
    def test_device_matches_hashlib(self, length: int) -> None:
        rng = np.random.default_rng(length + 3)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        got = np.asarray(Sha224().digest(msgs))
        self.assertEqual(got.shape, (4, 28))
        np.testing.assert_array_equal(
            got, oracle_digest(lambda b: hashlib.sha224(b).digest(), 28, msgs)
        )

    def test_value_identity_is_by_type(self) -> None:
        self.assertEqual(Sha224(), Sha224())
        self.assertEqual(hash(Sha224()), hash(Sha224()))
        # Never equal to its parent, or a consumer holding both would collide.
        self.assertNotEqual(Sha224(), Sha256())

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        got = np.asarray(Sha224().digest(fnp.zeros((0, 64), dtype=fnp.uint8)))
        self.assertEqual(got.shape, (0, 28))
        self.assertEqual(got.dtype, np.uint8)


class Sha256DigestRoutingTest(absltest.TestCase):
    """Which marker `digest` puts on the wire, and with which operands."""

    def test_routes_by_the_recognizer_flag(self) -> None:
        # `digest` must not emit a marker the pinned plugin cannot claim: with
        # the whole-message switch off, the wire carries the blocks marker only.
        # Flipping that switch is what moves it, and this is the test that makes
        # the flip deliberate.
        #
        # Matched whole (`emitted_composites`): the blocks marker's name is a
        # PREFIX of the whole-message one, so a substring assertion for the
        # blocks marker is satisfied by the wire carrying the other instead —
        # which is exactly the branch this test exists to tell apart.
        msg = np.zeros((1, 64), dtype=np.uint8)
        names = emitted_composites(sha256.digest, msg)
        if sha256._routes_to_bytes_marker():
            self.assertEqual(names, [sha256.SHA256_BYTES_MARKER])
        else:
            self.assertEqual(names, [sha256.SHA256_MARKER])

    def test_the_blocks_fallback_stays_reachable(self) -> None:
        # `digest`'s other arm, forced. Both switches carry the same backends,
        # so every leg takes the whole-message one and this arm would otherwise
        # run nowhere — the padded-words path and its `device_message` call
        # would rot untested. Byte-exactness is asserted with it, since the two
        # arms agreeing is the whole claim that makes the routing free.
        msgs = np.random.default_rng(11).integers(0, 256, (2, 100), dtype=np.uint8)
        with mock.patch.object(sha256, "_BYTES_EMITTER_BACKENDS", ("nonesuch",)):
            self.assertFalse(sha256._routes_to_bytes_marker())
            self.assertEqual(
                emitted_composites(sha256.digest, msgs), [sha256.SHA256_MARKER]
            )
            got = np.asarray(sha256.digest(msgs))
        for row, msg in zip(got, msgs):
            self.assertEqual(bytes(row), hashlib.sha256(bytes(msg)).digest())

    def test_carries_the_operands_its_routing_promises(self) -> None:
        # The composite name does not say which operand ABI reached the wire —
        # the recognizer claims two under it — so a name assertion cannot tell a
        # routed digest from one that emitted into a decline. The FUSION's
        # routing key can, because the rewriter picks it off the operands, so it
        # is what pins that the length operand actually arrived.
        if not sha256._routes_to_bytes_marker():
            self.skipTest("no whole-message recognizer on this backend")
        assert_marker_recognized(
            self,
            "sha256_bytes_len",
            sha256.digest,
            fnp.asarray(np.zeros((1, 64), dtype=np.uint8)),
        )


class Sha256BytesTest(parameterized.TestCase):
    """The whole-message marker, whose message length is an operand rather than
    part of the message shape.

    The buffer's extent is a CAPACITY here: the emitter loops on the length
    operand and never reads past it, so every case below fills the bytes from
    `length` to the capacity with 0xFF. A kernel that hashed the whole buffer, or
    sized its block count from the extent, cannot accidentally agree with
    `hashlib` under that fill — which is what makes these tests about the length
    operand rather than about SHA-256.
    """

    @staticmethod
    def _buffer(msg: np.ndarray, capacity: int) -> Array:
        """`msg` in a `capacity`-wide row whose spare bytes are 0xFF."""
        buf = np.full((msg.shape[0], capacity), 0xFF, dtype=np.uint8)
        buf[:, : msg.shape[-1]] = msg
        return fnp.asarray(buf)

    @parameterized.product(length=_LENGTHS, slack=(0, 256))
    def test_matches_hashlib_at_any_capacity(self, length: int, slack: int) -> None:
        # slack 0 isolates the synthesized padding at an exact fit (`max(., 1)`
        # is the recognizer's floor, which the empty message would sit under);
        # slack 256 is the point of the form — a message digests as itself in a
        # buffer far wider than it, so the padded length is what `length`
        # implies rather than what the extent does.
        msg = (np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A))[None, :]
        got = sha256.sha256_bytes(
            self._buffer(msg, max(length + slack, 1)), np.int32(length)
        )
        self.assertEqual(
            bytes(np.asarray(got)[0]), hashlib.sha256(bytes(msg[0])).digest()
        )

    @parameterized.parameters(*_LENGTHS)
    def test_matches_the_blocks_marker(self, length: int) -> None:
        # Two wire forms of one digest: this must byte-equal the blocks marker,
        # and so, transitively, every consumer's goldens.
        msgs = np.random.default_rng(length).integers(
            0, 256, size=(4, length), dtype=np.uint8
        )
        np.testing.assert_array_equal(
            np.asarray(sha256.sha256_bytes(self._buffer(msgs, 512), np.int32(length))),
            np.asarray(
                sha256.sha256_merkle_damgard(
                    sha256.INITIAL_STATE, sha256._padded_words(fnp.asarray(msgs))
                )
            ),
        )

    def test_two_lengths_in_one_program_stay_distinct(self) -> None:
        # The kernel reuse key must fold the CAPACITY and not the length: two
        # lengths sharing one kernel is the entire point of the form, and baking
        # a length instead returns one digest as the other. Silent either way —
        # right shape, wrong bytes — and it is the failure a wheel shipped once
        # for the bytes marker (fractalyze/xla#562), so the floor rests on it.
        short, long = b"abc", b"hello world, a longer one"

        def two(a: Array, la: Array, b: Array, lb: Array) -> tuple[Array, Array]:
            return sha256.sha256_bytes(a, la), sha256.sha256_bytes(b, lb)

        got_a, got_b = frx.jit(two)(
            self._buffer(np.frombuffer(short, dtype=np.uint8)[None, :], 128),
            np.int32(len(short)),
            self._buffer(np.frombuffer(long, dtype=np.uint8)[None, :], 128),
            np.int32(len(long)),
        )
        self.assertEqual(bytes(np.asarray(got_a)[0]), hashlib.sha256(short).digest())
        self.assertEqual(bytes(np.asarray(got_b)[0]), hashlib.sha256(long).digest())

    def test_recognized_where_routed(self) -> None:
        # The fusion contract: where `digest` routes to the marker, the pinned
        # plugin must claim it as one custom fusion. An unrecognized name inlines
        # its decomposition — right bytes, no kernel — which no value-level test
        # above can tell apart.
        if not sha256._routes_to_bytes_marker():
            self.skipTest("no whole-message recognizer on this backend")
        # `sha256_bytes_len` is the FUSION's routing key, not the marker name:
        # the two are different namespaces. This form travels under the
        # `hash_frx.digest.sha256_bytes` composite name, and Fractalyze XLA's
        # rewriter routes it — by its operands, not by its name — to a custom
        # fusion called `sha256_bytes_len`. So this key does not follow the
        # Python or the marker spelling.
        assert_marker_recognized(
            self,
            "sha256_bytes_len",
            sha256.sha256_bytes,
            fnp.asarray(np.zeros((2, 128), dtype=np.uint8)),
            np.int32(100),
        )

    def test_it_emits_exactly_one_composite(self) -> None:
        # Whole-name match, and the list length is the composite count:
        # `hash_frx.digest.sha256` is a prefix of this name, so the two nest and
        # a substring assertion would not tell them apart.
        self.assertEqual(
            emitted_composites(
                sha256.sha256_bytes,
                fnp.asarray(np.zeros((2, 128), dtype=np.uint8)),
                np.int32(100),
            ),
            [sha256.SHA256_BYTES_MARKER],
        )

    def test_the_length_is_an_operand_not_a_baked_constant(self) -> None:
        # What the whole form rests on: `len` reaching the marker as an operand.
        # Baked in, every length would be a fresh module again — and the module
        # would still compute the right bytes, so only the signature shows it.
        txt = (
            frx.jit(sha256.sha256_bytes)
            .lower(fnp.asarray(np.zeros((1, 128), dtype=np.uint8)), np.int32(100))
            .as_text()
        )
        signature = next(
            line for line in txt.splitlines() if "func.func public @main" in line
        )
        self.assertIn("tensor<i32>", signature)

    def test_rejects_a_zero_width_buffer(self) -> None:
        # The recognizer declines `LMAX < 1`, which would hand the work to a
        # decomposition that indexes the message through a clamp with no byte to
        # clamp to. An empty message is `length = 0` in a non-empty buffer.
        with self.assertRaisesRegex(ValueError, "LMAX >= 1"):
            sha256.sha256_bytes(
                fnp.asarray(np.zeros((1, 0), dtype=np.uint8)), np.int32(0)
            )

    def test_its_own_backend_tuple_is_pinned(self) -> None:
        # A separate tuple from the family's `_EMITTER_BACKENDS`, because
        # Fractalyze XLA gates this marker on its own flag: the two agreeing is
        # a fact about the pinned wheel rather than a property, so a widening
        # has to stay a conscious edit. `fusion_path_test._MATRIX` is where a
        # family's row is spelled, and it carries one tuple per module, so this
        # per-marker one is pinned here until the matrix grows a column for it.
        self.assertEqual(sha256._BYTES_EMITTER_BACKENDS, ("cpu", "gpu"))


class Sha256CapacityTest(parameterized.TestCase):
    """SHA-256's use of the seam's capacity ladder. The ladder's own cases —
    the width policy and the widening — live in `capacity_test`; what is
    SHA-256's is which block size it hands over and what that buys `digest`."""

    def test_digest_compiles_once_per_width_not_once_per_length(self) -> None:
        # The property the form exists for. Counting compilations directly is
        # backend-plumbing, so this counts what drives them: the distinct
        # (capacity, length-aval) pairs `digest` hands to the marker.
        lengths = range(20, 400, 7)
        widths = {
            capacity(np.zeros((1, length), np.uint8), sha256._PAD.block_size)
            for length in lengths
        }
        self.assertLess(len(widths), len(list(lengths)) // 8)
        self.assertEqual(widths, {64, 128, 256, 512})


class EmptyBatchTest(absltest.TestCase):
    """A zero-row batch is a valid batch (#211): every row returns
    uint8 [0, digest_size] instead of failing in a block-count reshape."""

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        got = np.asarray(sha256.Sha256().digest(fnp.zeros((0, 64), dtype=fnp.uint8)))
        self.assertEqual(got.shape, (0, 32))
        self.assertEqual(got.dtype, np.uint8)


class MessageRankTest(absltest.TestCase):
    """The row routes its message through the seam's rank check (#215): a
    single message is `B = 1`, not a bare `[L]`, and the miss used to surface
    from inside the marked region's trace instead of at the call."""

    def test_a_1d_message_is_rejected_at_the_seam(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D uint8"):
            sha256.digest(fnp.zeros(64, dtype=fnp.uint8))


if __name__ == "__main__":
    absltest.main()

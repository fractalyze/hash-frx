# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHA-512 byte-hash — the FIPS 180-4 worked examples, then byte-match against
the universal reference `hashlib.sha512` (named by no consumer).

The lengths exercise every padding boundary — empty, sub-block, the 111/112
one-vs-two block transition (where the 0x80 byte plus the 16-byte length field
force a second block), exact block multiples, and a multi-block message. The
carry sweep is deliberate and SHA-512-specific: a 64-bit add here is two uint32
half adds joined by a comparison-based carry, and 0xFF-saturated messages are
what drive that carry continuously — a dropped carry passes low-entropy
structured vectors while corrupting these.
"""

from __future__ import annotations

import functools
import hashlib
from typing import Any, Callable, NamedTuple

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx import markers
from hash_frx.byte_hash import ByteHash
from hash_frx.fusion import FusionPath
from hash_frx.sha256 import sha256 as sha256_mod
from hash_frx.sha256.sha256 import Sha256
from hash_frx.sha512 import sha512
from hash_frx.sha512.sha512 import (
    Sha384,
    Sha512,
    Sha512_224,
    Sha512_256,
)
from hash_frx.testing.jit_cache import assert_single_trace
from hash_frx.testing.marker_recognized import (
    assert_marker_recognized,
    emitted_composites,
)
from hash_frx.testing.oracle import oracle_digest

# Padding-boundary lengths: 0/1 (empty + tiny), 111/112 (the one-block/two-block
# cutoff — the 0x80 byte + 16-byte length need 17 bytes), 127/128 (block edge;
# 128 gets a whole extra padding block), 129 (one past it), 256 (multi-block).
_LENGTHS = (0, 1, 111, 112, 127, 128, 129, 256)

# FIPS 180-4's SHA-512 worked examples (the NIST example-values series), also
# the vectors every independent transcription publishes: one block ("abc" and
# the empty message) and the two-block 896-bit message.
_MSG_896 = (
    b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
    b"hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
)
_VECTORS = (
    (
        "abc",
        b"abc",
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
        "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
    ),
    (
        "empty",
        b"",
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
        "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
    ),
    (
        "two_block_896_bit",
        _MSG_896,
        "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018"
        "501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909",
    ),
)


class Sha512VectorTest(parameterized.TestCase):
    @parameterized.named_parameters(*_VECTORS)
    def test_matches_the_published_vector(self, msg: bytes, digest_hex: str) -> None:
        # Against the standard's record, not against another implementation in
        # this tree (docs/reference/conventions.md: byte-exactness is the gate).
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = np.asarray(sha512.digest(rows))[0]
        self.assertEqual(bytes(got).hex(), digest_hex)
        # And the reference the differential sweep uses is anchored to the same
        # record, so agreeing with hashlib below means agreeing with FIPS 180-4.
        self.assertEqual(hashlib.sha512(msg).hexdigest(), digest_hex)


class Sha512Test(parameterized.TestCase):
    @parameterized.parameters(*_LENGTHS)
    def test_matches_hashlib(self, length: int) -> None:
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        got = bytes(np.asarray(sha512.digest(msg[None, :]))[0])
        self.assertEqual(got, hashlib.sha512(bytes(msg)).digest())

    @parameterized.parameters(1, 7, 64, 111, 112, 128, 256)
    def test_carry_saturated_matches_hashlib(self, length: int) -> None:
        # 0xFF-filled messages: every message word is all-ones, so the 64-round
        # add chains overflow the low half continuously — the deliberate
        # exercise of `_add64`'s comparison-based carry across half pairs.
        msg = np.full(length, 0xFF, dtype=np.uint8)
        got = bytes(np.asarray(sha512.digest(msg[None, :]))[0])
        self.assertEqual(got, hashlib.sha512(bytes(msg)).digest())

    def test_batched_equals_per_row(self) -> None:
        # One data-parallel call over a stack of equal-length messages must
        # equal the per-message hashlib digests, in order.
        length = 128
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(7, length), dtype=np.uint8)
        got = np.asarray(sha512.digest(batch))
        for i in range(batch.shape[0]):
            self.assertEqual(bytes(got[i]), hashlib.sha512(bytes(batch[i])).digest())

    @parameterized.parameters(*_LENGTHS)
    def test_marked_equals_inline(self, length: int) -> None:
        # The marker only tags the region; with no dedicated emitter wired it
        # inlines its decomposition, so the marked digest must byte-equal the
        # unmarked compression at every padding boundary.
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        blocks = sha512._padded_words(fnp.asarray(msg[None, :]))
        marked = np.asarray(sha512.sha512_merkle_damgard(sha512.INITIAL_STATE, blocks))
        state = fnp.broadcast_to(sha512.INITIAL_STATE, (1, 16))
        inline = np.asarray(sha512.serialize_digest(sha512.compress(state, blocks)))
        np.testing.assert_array_equal(marked, inline)

    def test_serialize_deserialize_roundtrip(self) -> None:
        # deserialize_digest inverts serialize_digest, so unpacking a digest
        # recovers the exact midstate a stream resumes from.
        rng = np.random.default_rng(0)
        state = fnp.asarray(rng.integers(0, 2**32, (3, 16), np.int64).astype(np.uint32))
        back = sha512.deserialize_digest(sha512.serialize_digest(state))
        np.testing.assert_array_equal(np.asarray(back), np.asarray(state))

    @parameterized.parameters(1, 2, 3)
    def test_merkle_damgard_resumes_from_midstate(self, split: int) -> None:
        # From a non-IV midstate the compression resumes: hashing a 4-block
        # message in two chained halves must equal one pass over all 4.
        blocks = fnp.asarray(
            np.random.default_rng(split)
            .integers(0, 2**32, (1, 4, 32), np.int64)
            .astype(np.uint32)
        )
        whole = sha512.sha512_merkle_damgard(sha512.INITIAL_STATE, blocks)
        mid = sha512.deserialize_digest(
            sha512.sha512_merkle_damgard(sha512.INITIAL_STATE, blocks[:, :split])
        )[0]
        resumed = sha512.sha512_merkle_damgard(mid, blocks[:, split:])
        np.testing.assert_array_equal(np.asarray(whole), np.asarray(resumed))

    def test_compress_explicit_k_matches_default(self) -> None:
        # Threading the round-constant table as an explicit `k` operand (what
        # the marked region does) matches the module-default `_Kd`.
        blocks = sha512._padded_words(
            fnp.asarray(np.arange(160, dtype=np.uint8))[None, :]
        )
        state = fnp.broadcast_to(sha512.INITIAL_STATE, (1, 16))
        default = sha512.compress(state, blocks)
        explicit = sha512.compress(state, blocks, sha512._Kd)
        np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))


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


_HAS_SHA512_EMITTER = sha512._routes_to_dedicated_emitter()


class Sha512MarkerTest(absltest.TestCase):
    def test_routing_is_the_pin_and_the_backend(self) -> None:
        # The conjunction the keccak family's gate test states. Both halves are
        # pinned because they move together with the `frx>=` floor, and both
        # backends are here because the envelope that routes this is shared
        # rather than per-family — it arrived with both arms at once.
        self.assertTrue(sha512._DEDICATED_EMITTER_AVAILABLE)
        self.assertEqual(sha512._EMITTER_BACKENDS, ("cpu", "gpu"))
        self.assertIs(
            Sha512().fusion_path, FusionPath.from_routing(_HAS_SHA512_EMITTER)
        )

    def test_digest_emits_one_composite_with_the_digest_name(self) -> None:
        # The contract's unit: an absent or split marker still computes the
        # right bytes, so only the lowered module shows it.
        #
        # The name is the OPERATION one now — the family moved off its own
        # spelling when the plugin grew a shared words-in envelope, and rides
        # the `primitive` attribute instead. Matched whole, and the list length
        # is the composite count, so this pins both which marker is on the wire
        # and that there is exactly one.
        msg = np.zeros((2, 100), dtype=np.uint8)
        self.assertEqual(
            emitted_composites(sha512.digest, msg), [markers.MD_DIGEST_MARKER]
        )

    def test_the_digest_compiles_to_a_custom_fusion(self) -> None:
        # What the routing flags CLAIM, checked against what the pinned plugin
        # actually does. The bytes cannot show it — an unrecognized marker
        # inlines and computes the same digest — so this asserts the compiled
        # module, which is where a recognized marker becomes a kCustom fusion.
        #
        # `md_digest` is the envelope: SHA-512 has no emitter of its own, so
        # the instruction is named for the shared one and the family is pinned
        # by the `primitive` its config names.
        if not _HAS_SHA512_EMITTER:
            self.skipTest(f"no SHA-512 emitter on {frx.default_backend()}")
        msg = fnp.asarray(np.zeros((2, 100), dtype=np.uint8))
        assert_marker_recognized(
            self, "sha512", sha512.digest, msg, envelope_key="md_digest"
        )

    def test_the_marker_carries_its_name_version_and_operand_abi(self) -> None:
        # The wire surface an emitter will read: name, version, and the
        # documented [h0, k, blocks] operand order in the big-endian pair
        # layout. Three invars exactly is the captured-constants-free property
        # — an array the body closed over would be lifted in AHEAD of these,
        # one per call site (the operand-ABI rule in
        # docs/reference/conventions.md).
        msg = fnp.asarray(np.zeros((2, 100), dtype=np.uint8))
        eqn = _composite(sha512.digest, msg)
        # One name for every words-in family; SHA-512 is named by the attribute
        # the plugin resolves through, not by a suffix.
        self.assertEqual(eqn.params["name"], markers.MD_DIGEST_MARKER)
        self.assertEqual(eqn.params["version"], markers.MD_DIGEST_MARKER_VERSION)
        attrs = {key: leaves[0] for key, leaves, _ in eqn.params["attributes"]}
        self.assertEqual(attrs["primitive"], "sha512")
        self.assertLen(eqn.invars, 3)
        shapes = [tuple(v.aval.shape) for v in eqn.invars]
        # L = 100: one block once padded (100 + 17 ≤ 128).
        self.assertEqual(shapes, [(16,), (160,), (2, 1, 32)])


class Sha512TracedTest(parameterized.TestCase):
    """`digest` inside a traced region.

    The point of the padding being a function of the length: a consumer can
    hash inside its own `@jit` without reaching past the seam for the
    compression, so a scheme built on `ByteHash` does not have to name SHA-512
    to be traceable. One boundary-crossing length: compile time scales with
    block count, the property does not (the eager sweeps already own value
    coverage at every boundary).
    """

    def test_jit_matches_eager_and_hashlib(self) -> None:
        # Against the reference as well as the eager path: eager agreeing with
        # a traced path that shares its bug would prove nothing.
        msg = (np.arange(129, dtype=np.uint8) ^ np.uint8(0x5A))[None, :]
        traced = np.asarray(frx.jit(sha512.digest)(msg))
        np.testing.assert_array_equal(traced, np.asarray(sha512.digest(msg)))
        self.assertEqual(bytes(traced[0]), hashlib.sha512(bytes(msg[0])).digest())

    def test_vmap_matches_the_batch_axis(self) -> None:
        # `digest` already takes the batch, so mapping a one-row call over a
        # stack must reproduce it — which is what a consumer gets for free when
        # its own vmap encloses the hash.
        batch = np.random.default_rng(3).integers(0, 256, (5, 130), dtype=np.uint8)
        mapped = frx.vmap(lambda row: sha512.digest(row[None, :])[0])(batch)
        np.testing.assert_array_equal(
            np.asarray(mapped), np.asarray(sha512.digest(batch))
        )

    def test_the_seam_carries_it(self) -> None:
        # Through `ByteHash.digest` rather than the module function: the
        # consumer holds the seam, and that is the call that has to survive the
        # tracer.
        hasher: ByteHash = Sha512()
        msg = np.random.default_rng(4).integers(0, 256, (3, 100), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(hasher.digest)(msg)), np.asarray(hasher.digest(msg))
        )

    def test_digest_reuses_one_trace_across_instances(self) -> None:
        # The zone's cache is keyed on the message aval alone (the digest is
        # parameterless), so freshly built instances must share one trace —
        # the property `sha512_merkle_damgard`'s module-level jit zone exists
        # for. The (3, 100) aval is one `test_the_seam_carries_it` already
        # compiled, so the pin costs no fresh compile.
        msgs = fnp.asarray(
            np.random.default_rng(4).integers(0, 256, (3, 100), dtype=np.uint8)
        )
        calls = [functools.partial(Sha512().digest, msgs) for _ in range(3)]
        assert_single_trace(self, sha512.sha512_merkle_damgard, calls)

    def test_both_sha2_rows_coexist_in_one_jit_region(self) -> None:
        # Issue #66's acceptance shape: FIPS 205 §11.2.2's category-3/5
        # tweakable-hash family keeps SHA-256 for PRF/F and moves SHA-512 into
        # H/T_l, so ONE traced region must carry both hashes side by side —
        # each against its own reference.
        h256, h512 = Sha256(), Sha512()
        msgs = np.random.default_rng(66).integers(0, 256, (3, 100), dtype=np.uint8)

        @frx.jit
        def family(m: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
            return h256.digest(m), h512.digest(m)

        d256, d512 = family(msgs)
        for i in range(msgs.shape[0]):
            row = bytes(msgs[i])
            self.assertEqual(bytes(np.asarray(d256)[i]), hashlib.sha256(row).digest())
            self.assertEqual(bytes(np.asarray(d512)[i]), hashlib.sha512(row).digest())


class Sha512ByteHashTest(parameterized.TestCase):
    """The `ByteHash` row, against the seam and against `hashlib`.

    `byte_hash_test.py` stays seam-only — a double, so it runs on a branch
    where no concrete hash exists — which leaves the real classes untested by
    it. This is the other half of that split: the assertions that need `Sha512`
    itself and are tautologies against a double.
    """

    def test_impls_satisfy_the_seam(self) -> None:
        h = Sha512()
        self.assertIsInstance(h, ByteHash)
        self.assertEqual(h.digest_size, 64)
        self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_paths_pin_the_substrate(self) -> None:
        # The routing gate, and the traceability tie to the return type: the
        # row returns an `Array` and takes a tracer (`byte_hash.py`'s rule).
        msg = np.zeros((1, 1), dtype=np.uint8)
        device = Sha512()
        self.assertIs(device.fusion_path, FusionPath.from_routing(_HAS_SHA512_EMITTER))
        self.assertNotIsInstance(device.digest(msg), np.ndarray)

    @parameterized.parameters(*_LENGTHS)
    def test_device_matches_hashlib(self, length: int) -> None:
        # `hashlib` is the out-of-tree oracle, across every padding boundary
        # rather than one convenient length.
        rng = np.random.default_rng(length + 1)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        np.testing.assert_array_equal(
            np.asarray(Sha512().digest(msgs)),
            oracle_digest(lambda b: hashlib.sha512(b).digest(), 64, msgs),
        )

    def test_value_identity_is_by_type(self) -> None:
        # Param-free, so every instance of a type is equal and hashes alike —
        # what keeps the seam re-trace-safe as pytree aux; and it does not
        # equal the SHA-256 sibling, or a family holding both would collide.
        self.assertEqual(Sha512(), Sha512())
        self.assertEqual(hash(Sha512()), hash(Sha512()))
        self.assertNotEqual(Sha512(), Sha256())


# ---------------------------------------------------------------------------
# The truncated variants (§6.5 SHA-384, §6.7 SHA-512/256): the published
# worked examples, the hashlib differential, the shared-marker pins, and the
# rows. Every message shape below reuses an aval the SHA-512 cases above
# already compiled, and `h0`'s aval is the same uint32 [16] for every variant
# — so this whole section adds no fresh trace or compile of the 80-round
# chain (the one-trace case pins exactly that).
# ---------------------------------------------------------------------------

# The NIST example-values series for the truncated variants ("abc" and the
# two-block 896-bit message), plus the empty message every independent
# transcription publishes.
_VARIANT_VECTORS = (
    (
        "sha384_abc",
        "sha384",
        b"abc",
        "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed"
        "8086072ba1e7cc2358baeca134c825a7",
    ),
    (
        "sha384_empty",
        "sha384",
        b"",
        "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da"
        "274edebfe76f65fbd51ad2f14898b95b",
    ),
    (
        "sha384_two_block_896_bit",
        "sha384",
        _MSG_896,
        "09330c33f71147e83d192fc782cd1b4753111b173b3b05d22fa08086e3b0f712"
        "fcc7c71a557e2db966c3e9fa91746039",
    ),
    (
        "sha512_256_abc",
        "sha512_256",
        b"abc",
        "53048e2681941ef99b2e29b76b4c7dabe4c2d0c634fc6d46e0e2f13107e7af23",
    ),
    (
        "sha512_256_empty",
        "sha512_256",
        b"",
        "c672b8d1ef56ed28ab87c3622c5114069bdd3ad7b8f9737498d0c01ecef0967a",
    ),
    (
        "sha512_256_two_block_896_bit",
        "sha512_256",
        _MSG_896,
        "3928e184fb8690f840da3988121d31be65cb9d3ef83ee6146feac861e19b563a",
    ),
)


class _Variant(NamedTuple):
    """One truncated SHA-512 row, carrying everything a sweep needs of it.

    The table is the single source: every variant sweep below derives its rows
    from it rather than listing them, so a row added here arrives with the
    vector, differential, seam, routing and identity assertions the others
    already have. Listing them is what let `SeamConformanceTest` fall a family
    behind in the keccak file.
    """

    digest: Callable[..., Any]  # the module-level function
    size: int
    oracle: Callable[[bytes], Any]  # hashlib constructor over the same bytes
    row: type[ByteHash]  # the seam row over it


# The two SHA-512/t rows reach hashlib through OpenSSL (`hashlib.new`) rather
# than a named constructor.
_VARIANTS = {
    "sha384": _Variant(sha512.sha384_digest, 48, lambda b: hashlib.sha384(b), Sha384),
    "sha512_224": _Variant(
        sha512.sha512_224_digest,
        28,
        lambda b: hashlib.new("sha512_224", b),
        Sha512_224,
    ),
    "sha512_256": _Variant(
        sha512.sha512_256_digest,
        32,
        lambda b: hashlib.new("sha512_256", b),
        Sha512_256,
    ),
}

# NIST CAVP `shabytetestvectors`, `SHA512_224ShortMsg.rsp` — extracted from the
# .rsp rather than typed. What they add over the `hashlib` sweeps, which already
# cover these lengths and more, is independence from `hashlib`: the sweeps say
# this tree agrees with OpenSSL's SHA-512/224, and these say both agree with
# NIST. All 129 rows of the file were checked while extracting, so the six kept
# are a readable subset of a full agreement rather than the extent of what was
# verified. The lengths are the padding boundaries `_LENGTHS` names, 112 and 128
# being the ones where the 0x80 byte plus the 16-byte length field force a
# further block.
_CAVP_512_224 = (
    ("len_0", "", "6ed0dd02806fa89e25de060c19d3ac86cabb87d6a0ddd05c333b84f4"),
    ("len_1", "cf", "4199239e87d47b6feda016802bf367fb6e8b5655eff6225cb2668f4a"),
    (
        "len_111",
        "15cb777ef3e451b928dbf288e46a3627044ff5de42add884a9af6b424d6e7399"
        "381581a6a743c7a577b02bb5da149ada4e449f48d09e34df4ca8d8f259f4e14c"
        "23471475a8f97331289f564ad6e8bd8fd4c5e51d5ecd19dd46dfcb4ea009e385"
        "bea857725fd1fc6423f09ccf42af48",
        "9fa12561f1df9a2d793292e0f4df5327af529336b5b2118952f5c24e",
    ),
    (
        "len_112",
        "d24df75a00cf92677bb41a620fae519723937ebfe1f7b430970056505d76db4f"
        "f91acf16ff391a7a3d8085b655127a18acd80bfa831837f4644a6850c0273fbe"
        "d6029449d65bb98a47b2ff1ca6997c50500d0b21a206936a5e4d8d56508ec018"
        "32ae4fddce5ef6ff62f1917c486adea6",
        "5cae12ea9652269ea2aafc656cb83424746ea1d5d491f9a159594b2a",
    ),
    (
        "len_127",
        "4b9895235cb4956aefffe815415252e7d6b21921bd7f675315eff071d0bbd429"
        "b718c774aee96f6c3a330d5d40d1601e1069c7a2a19ea5ca1e87097da2608ffb"
        "4180816e478b42c3c4e9edb748773935eb7ca0df90dec0eb6b960130c1617880"
        "efb80b39ae03d617950ace4ce0aca4d36fd3ed0112a77f5d03021eb1b42458",
        "9a9176e97aec99ab07f468f6a226876710d6d877021d27061d4d0132",
    ),
    (
        "len_128",
        "9e127870be2431bcb4f4eb4efd5c2a6c5870c55e7a5e3b7503994a4cb136be4e"
        "d396887801450f600b22cb772fc00f8b8f0d2690e231a29f69b9f13f24f531e4"
        "479e45b5e8bc2992fac782567e0d7a59f853ca3a20bf18dbdbf684ac69817e2d"
        "e075daaed9532659692d3b73530a12df7b8cd9e49ed0463041962c1ce7a24c31",
        "7e2cf6226623535784c59cd6a7b27dac60ee23fdce8a804dbd6dfedd",
    ),
)


def _bytes_oracle(variant: _Variant) -> Callable[[bytes], bytes]:
    """`variant`'s hashlib oracle as bytes-in/bytes-out, bound to this variant.

    A closure over the loop variable would read every row against whichever
    variant the loop ended on, and a default-argument lambda that avoids it is
    something mypy cannot infer a type for.
    """
    return lambda b: variant.oracle(b).digest()


# variant x length, so a failure names `sha512_224_112` rather than an ordinal.
_VARIANT_LENGTHS = tuple(
    (f"{name}_{length}", name, length) for name in _VARIANTS for length in _LENGTHS
)


class Sha2VariantVectorTest(parameterized.TestCase):
    @parameterized.named_parameters(*_VARIANT_VECTORS)
    def test_matches_the_published_vector(
        self, variant: str, msg: bytes, digest_hex: str
    ) -> None:
        v = _VARIANTS[variant]
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        got = np.asarray(v.digest(rows))[0]
        self.assertEqual(bytes(got).hex(), digest_hex)
        # The reference the differential sweep uses is anchored to the same
        # record, so agreeing with hashlib below means agreeing with FIPS.
        self.assertEqual(v.oracle(msg).hexdigest(), digest_hex)


class Sha512_224CavpTest(parameterized.TestCase):
    """SHA-512/224 against NIST's own published vectors.

    `Sha2VariantVectorTest` covers the other two rows from the example-values
    series; this row's published evidence is CAVP instead, which is also the
    only source here that lands on a length ending exactly on a block boundary
    with the padding taking a whole further block.
    """

    @parameterized.named_parameters(*_CAVP_512_224)
    def test_matches_the_cavp_vector(self, msg_hex: str, digest_hex: str) -> None:
        msg = bytes.fromhex(msg_hex)
        rows = np.frombuffer(msg, dtype=np.uint8).reshape(1, len(msg))
        self.assertEqual(
            bytes(np.asarray(sha512.sha512_224_digest(rows))[0]).hex(), digest_hex
        )
        # Anchors the differential sweep's reference to the same record, so
        # agreeing with hashlib elsewhere means agreeing with NIST.
        self.assertEqual(hashlib.new("sha512_224", msg).hexdigest(), digest_hex)


class Sha2VariantTest(parameterized.TestCase):
    @parameterized.named_parameters(*_VARIANT_LENGTHS)
    def test_matches_hashlib(self, variant: str, length: int) -> None:
        v = _VARIANTS[variant]
        msg = np.arange(length, dtype=np.uint8) ^ np.uint8(0x5A)
        got = bytes(np.asarray(v.digest(msg[None, :]))[0])
        self.assertEqual(got, v.oracle(bytes(msg)).digest())

    def test_variants_ride_the_sha512_marker_with_truncation_outside(self) -> None:
        # The #199 acceptance pin: ONE composite naming sha512 (by attribute,
        # since the family rides the shared words-in marker), whose output is
        # the FULL [B, 64] serialized state — the
        # truncation is the caller's slice, outside the marker, so the wire
        # contract a recognizer reads is byte-for-byte SHA-512's.
        msg = fnp.asarray(np.zeros((1, 1), dtype=np.uint8))
        for variant, v in _VARIANTS.items():
            with self.subTest(variant=variant):
                eqn = _composite(v.digest, msg)
                self.assertEqual(eqn.params["name"], markers.MD_DIGEST_MARKER)
                self.assertEqual(
                    eqn.params["version"], markers.MD_DIGEST_MARKER_VERSION
                )
                attrs = {key: leaves[0] for key, leaves, _ in eqn.params["attributes"]}
                self.assertEqual(attrs["primitive"], "sha512")
                self.assertEqual(tuple(eqn.outvars[0].aval.shape), (1, 64))
                self.assertEqual(np.asarray(v.digest(msg)).shape, (1, v.size))

    def test_sha224s_iv_is_sha384s_low_halves(self) -> None:
        # Both tables are the fractional parts of the square roots of the 9th
        # to 16th primes: SHA-384 takes the leading 64 bits of each and SHA-224
        # the second 32. Asserted here because this file already reaches both
        # families. Independent of `hashlib`, so a transcription slip in either
        # table fails even if both files were edited to agree with each other.
        self.assertEqual(
            tuple(int(w) for w in sha256_mod._H224),
            tuple(v & 0xFFFFFFFF for v in sha512._H384_64),
        )

    def test_one_chain_trace_serves_every_variant(self) -> None:
        # The issue-#199 design pin: a variant is DATA (a different h0 value
        # on the same [16] aval), not ABI — so SHA-512, SHA-384 and
        # SHA-512/256 of one message shape must all ride one trace of the
        # marked chain.
        msgs = fnp.asarray(np.zeros((1, 1), dtype=np.uint8))
        calls = [
            functools.partial(fn, msgs)
            for fn in (sha512.digest, *(v.digest for v in _VARIANTS.values()))
        ]
        assert_single_trace(self, sha512.sha512_merkle_damgard, calls)


class Sha2VariantByteHashTest(parameterized.TestCase):
    """The four variant rows against the seam and against each other — the
    `Sha512ByteHashTest` split at the truncated tables. The device/host
    agreement plus the module-function sweeps above close the triangle."""

    def test_impls_satisfy_the_seam(self) -> None:
        for v in _VARIANTS.values():
            h = v.row()
            with self.subTest(impl=type(h).__name__):
                self.assertIsInstance(h, ByteHash)
                self.assertEqual(h.digest_size, v.size)
                self.assertIsInstance(h.fusion_path, FusionPath)

    def test_fusion_paths_pin_the_substrate(self) -> None:
        # The variants are IV rows over the same marker, so they inherit the
        # family's routing rather than having one of their own.
        msg = np.zeros((1, 1), dtype=np.uint8)
        for v in _VARIANTS.values():
            device = v.row()
            with self.subTest(variant=type(device).__name__):
                self.assertIs(
                    device.fusion_path, FusionPath.from_routing(_HAS_SHA512_EMITTER)
                )
                self.assertNotIsInstance(device.digest(msg), np.ndarray)

    @parameterized.parameters(*_LENGTHS)
    def test_device_matches_hashlib(self, length: int) -> None:
        rng = np.random.default_rng(length + 2)
        msgs = rng.integers(0, 256, size=(4, length), dtype=np.uint8)
        for v in _VARIANTS.values():
            device = v.row()
            with self.subTest(variant=type(device).__name__):
                got = np.asarray(device.digest(msgs))
                self.assertEqual(got.shape, (4, device.digest_size))
                one = _bytes_oracle(v)
                np.testing.assert_array_equal(
                    got, oracle_digest(one, device.digest_size, msgs)
                )

    def test_value_identity_is_by_type(self) -> None:
        for v in _VARIANTS.values():
            with self.subTest(impl=v.row.__name__):
                self.assertEqual(v.row(), v.row())
                self.assertEqual(hash(v.row()), hash(v.row()))
        # No two rows of the family are ever equal — across variants and the
        # parent — or a consumer holding several would collide. Sha512_224 and
        # Sha512_256 make that non-trivial: same type shape, same marker, and
        # 224 shares its digest width with SHA-224 in the other family.
        distinct = [Sha512(), *(v.row() for v in _VARIANTS.values())]
        for i, a in enumerate(distinct):
            for b in distinct[i + 1 :]:
                self.assertNotEqual(a, b)


class EmptyBatchTest(absltest.TestCase):
    """A zero-row batch is a valid batch (#211): every row returns
    uint8 [0, digest_size] instead of failing in a block-count reshape."""

    def test_zero_rows_digest_to_zero_rows(self) -> None:
        rows: list[tuple[ByteHash, int]] = [
            (sha512.Sha512(), 64),
            *((v.row(), v.size) for v in _VARIANTS.values()),
        ]
        for hasher, size in rows:
            got = np.asarray(hasher.digest(fnp.zeros((0, 64), dtype=fnp.uint8)))
            self.assertEqual(got.shape, (0, size))
            self.assertEqual(got.dtype, np.uint8)


if __name__ == "__main__":
    absltest.main()

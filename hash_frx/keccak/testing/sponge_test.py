# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Keccak sponge's whole-hash marker and the ABI a KeccakFusion emitter reads.

Values live in `byte_hashes_test`, which holds the published rows to `hashlib`.
What is checked here is the property values cannot show: that a whole padded
absorb and squeeze is ONE marked region rather than a marked permute per block
with the XOR glue outside, and that it says so in the layout an emitter reads.

Both routings are exercised, because the marker chooses a kernel and never a
result — so the two must agree, and they are each checked against `hashlib`
rather than against each other, so a shared mistake cannot pass.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hash_frx.fusion import FUSED_REGION_MARKER
from hash_frx.keccak import permutation as permutation_mod
from hash_frx.keccak import sponge as sponge_mod
from hash_frx.keccak.byte_hashes import SHAKE128_RATE, SHAKE_SUFFIX
from hash_frx.keccak.sponge import (
    KECCAK_SPONGE_MARKER,
    KECCAK_SPONGE_MARKER_VERSION,
    KeccakSponge,
)
from hash_frx.testing.composite_eqn import (
    composite_attrs,
    composite_eqn,
    composite_eqns,
)
from hash_frx.testing.marker_recognized import assert_marker_recognized
from hash_frx.testing.routing_mock import (
    dedicated_emitter,
    generic_emitter,
)

# Read off the shipped condition rather than restated, so a backend gaining an
# arm lifts the cases below with it and the two cannot drift.
_HAS_KECCAK_EMITTER = permutation_mod._routes_to_dedicated_emitter()

# SHA3-256's parameters: one rate block absorbed, one squeezed.
_SHA3_256 = KeccakSponge(rate=136, suffix=0x06, output_size=32)
# Two absorb blocks and two squeezes, so the glue between permutes is real: at
# rate 168 a 200-byte message pads to two blocks, and 336 bytes out is two.
_SHAKE128_LONG = KeccakSponge(
    rate=SHAKE128_RATE, suffix=SHAKE_SUFFIX, output_size=2 * SHAKE128_RATE
)


def _message(batch: int, length: int) -> frx.Array:
    rng = np.random.default_rng(0)
    return fnp.asarray(rng.integers(0, 256, size=(batch, length), dtype=np.uint8))


def _stacked(outer: int, batch: int, length: int) -> frx.Array:
    """A `[outer, B, L]` message, the shape a caller's own `vmap` hands `hash`."""
    rng = np.random.default_rng(0)
    return fnp.asarray(
        rng.integers(0, 256, size=(outer, batch, length), dtype=np.uint8)
    )


class KeccakSpongeMarkerTest(absltest.TestCase):
    def test_the_whole_hash_is_one_region_on_the_dedicated_path(self) -> None:
        # The point of the marker. Unmarked, a two-block absorb plus a second
        # squeeze is three marked permutes — two absorbing, one between the
        # squeezed blocks — with the XOR absorb glue between them left outside.
        # That is a launch per block, and the reason an unrecognized Keccak sits
        # in the hot path of every ML-DSA sign.
        msg = _message(2, 200)
        with generic_emitter(permutation_mod):
            generic = composite_eqns(_SHAKE128_LONG.hash, msg)
        self.assertLen(generic, 3)
        for eqn in generic:
            self.assertEqual(eqn.params["name"], FUSED_REGION_MARKER)

        with dedicated_emitter(permutation_mod):
            dedicated = composite_eqns(_SHAKE128_LONG.hash, msg)
        self.assertLen(dedicated, 1)
        self.assertEqual(dedicated[0].params["name"], KECCAK_SPONGE_MARKER)
        self.assertEqual(dedicated[0].params["version"], KECCAK_SPONGE_MARKER_VERSION)

    def test_the_region_carries_the_permutation_and_the_sponge_shape(self) -> None:
        # The permutation's attrs ride alongside the construction's, so the
        # marker says which primitive runs inside it as well as what wraps it.
        with dedicated_emitter(permutation_mod):
            eqn = composite_eqn(_SHA3_256.hash, _message(1, 64))
        attrs = composite_attrs(eqn)
        self.assertEqual(
            attrs,
            {
                "permutation": "keccak_f",
                "width": 50,
                "rounds": 24,
                "rate": 136,
                "output_size": 32,
            },
        )

    def test_the_region_takes_the_padded_message_and_the_permutation_abi(self) -> None:
        # Operand 0 is the padded byte matrix — rate-aligned, so the emitter
        # never needs the domain suffix — and the rest is whatever the
        # permutation's own ABI names. No captured constants ahead of them.
        with dedicated_emitter(permutation_mod):
            eqn = composite_eqn(_SHA3_256.hash, _message(3, 64))
        self.assertLen(eqn.invars, 2)
        self.assertEqual(eqn.invars[0].aval.shape, (3, 136))  # one padded block
        self.assertEqual(eqn.invars[0].aval.dtype, fnp.uint8)
        self.assertEqual(eqn.invars[1].aval.shape, (5, 5))

    def test_both_routings_match_hashlib(self) -> None:
        # A marker chooses a kernel, never a result. Held to `hashlib` on both
        # sides rather than to each other.
        cases = (
            ("sha3_256", _SHA3_256, 64, lambda m: hashlib.sha3_256(m).digest()),
            (
                "shake128",
                _SHAKE128_LONG,
                200,
                lambda m: hashlib.shake_128(m).digest(2 * SHAKE128_RATE),
            ),
        )
        for name, sponge, length, want in cases:
            msg = _message(2, length)
            rows = [bytes(np.asarray(row)) for row in np.asarray(msg)]
            for label, ctx in (
                ("generic", generic_emitter(permutation_mod)),
                ("dedicated", dedicated_emitter(permutation_mod)),
            ):
                with self.subTest(case=name, routing=label), ctx:
                    got = np.asarray(frx.jit(sponge.hash)(msg))
                    for i, row in enumerate(rows):
                        self.assertEqual(bytes(got[i]), want(row))

    def test_the_marked_body_falls_back_to_the_reference_decomposition(self) -> None:
        # An unrecognized marker inlines, so the body inside it has to BE the
        # hash. Compared against the generic path's lowering, which is the same
        # computation with the region boundary in a different place.
        msg = _message(1, 64)
        with generic_emitter(permutation_mod):
            plain = np.asarray(frx.jit(_SHA3_256.hash)(msg))
        with dedicated_emitter(permutation_mod):
            marked = np.asarray(frx.jit(_SHA3_256.hash)(msg))
        np.testing.assert_array_equal(plain, marked)


class VmappedMarkerTest(absltest.TestCase):
    """The rank contract, which only a caller's own `vmap` exercises.

    `hash` guards `msg.ndim != 2`, but under `frx.vmap` a tracer's *logical* ndim
    is 2 while the lowered operand carries the batch axis — so the guard passes
    and the region reaches the emitter one rank deeper than the shape it was
    written against. A Python rank check cannot protect a wire ABI.

    Nothing vmapped a marked *sponge* before this: the only `vmap` coverage in
    the repo is on the permutation (`permutation_test`), which is one vmap deep
    by construction and so was rank-tolerant from the start. On the dedicated
    path the sponge is the OUTERMOST marker, so it is the one nobody batched —
    which is how a whole-hash marker admitting exactly rank 2 shipped and took
    every consumer that batches with `vmap` down on GPU while CPU stayed green.
    """

    def test_a_vmapped_hash_is_still_one_region_a_rank_deeper(self) -> None:
        # Two absorb blocks and two squeezes, so the region is the whole chain
        # rather than a single permute, and vmap's axis stays outermost ahead of
        # the sponge's own batch.
        with dedicated_emitter(permutation_mod):
            eqns = composite_eqns(frx.vmap(_SHAKE128_LONG.hash), _stacked(3, 2, 200))
        self.assertLen(eqns, 1)
        self.assertEqual(eqns[0].params["name"], KECCAK_SPONGE_MARKER)
        # 200 B at rate 168 pads to two blocks, under both batch axes.
        self.assertEqual(eqns[0].invars[0].aval.shape, (3, 2, 2 * SHAKE128_RATE))
        self.assertEqual(eqns[0].invars[0].aval.dtype, fnp.uint8)

    def test_a_leading_axis_of_one_is_still_a_rank_it_has_to_admit(self) -> None:
        # The degenerate case a fix that squeezes rather than collapses would
        # pass, so it is pinned apart from the case above.
        with dedicated_emitter(permutation_mod):
            eqns = composite_eqns(frx.vmap(_SHA3_256.hash), _stacked(1, 3, 64))
        self.assertLen(eqns, 1)
        self.assertEqual(eqns[0].invars[0].aval.shape, (1, 3, 136))

    def test_a_vmapped_hash_matches_its_flattening(self) -> None:
        # What the collapse has to preserve: vmapping over `[V, B, L]` is the
        # same set of digests as hashing `[V*B, L]` in one go, just reshaped.
        # Both routings, since the marker chooses a kernel and never a result.
        stacked = _stacked(3, 2, 200)
        flat = fnp.reshape(stacked, (3 * 2, 200))
        for label, ctx in (
            ("generic", generic_emitter(permutation_mod)),
            ("dedicated", dedicated_emitter(permutation_mod)),
        ):
            with self.subTest(routing=label), ctx:
                got = np.asarray(frx.jit(frx.vmap(_SHAKE128_LONG.hash))(stacked))
                want = np.asarray(frx.jit(_SHAKE128_LONG.hash)(flat))
                np.testing.assert_array_equal(got.reshape(want.shape), want)


class WholeHashRoutingTest(absltest.TestCase):
    """What the backend gate is actually protecting.

    `hash` routes to the whole-chain `keccak_sponge` region only when the
    permutation reports a `DEDICATED` fusion path, and that region is the expensive
    thing to trace: it unrolls the entire padded absorb and squeeze into one
    composite. Where the emitter exists that buys one kernel; where it does not
    it buys nothing and the trace is spent anyway.
    """

    def test_a_leg_without_an_arm_does_not_build_the_whole_hash_region(self) -> None:
        msg = _message(1, 200)
        with mock.patch.object(permutation_mod, "_EMITTER_BACKENDS", ("nonesuch",)):
            names = [e.params["name"] for e in composite_eqns(_SHAKE128_LONG.hash, msg)]
        self.assertNotIn(KECCAK_SPONGE_MARKER, names)
        self.assertEqual(set(names), {FUSED_REGION_MARKER})

    def test_this_leg_builds_the_region_exactly_when_it_can_honour_it(self) -> None:
        msg = _message(1, 200)
        names = [e.params["name"] for e in composite_eqns(_SHAKE128_LONG.hash, msg)]
        if _HAS_KECCAK_EMITTER:
            self.assertEqual(names, [KECCAK_SPONGE_MARKER])
        else:
            self.assertEqual(set(names), {FUSED_REGION_MARKER})


class DedicatedRegionTraceCountTest(absltest.TestCase):
    """The whole-hash region is built once per shape, not once per call.

    `lax.composite` re-traces its decomposition on every emission, and this
    marker's decomposition is the entire absorb and squeeze — so without a zone
    an eager caller rebuilds all of it every call. That is not a slow path, it is
    200-800x the generic one (#151), and no value test can see it: the bytes are
    right either way and every published vector passes.

    Counted rather than timed, so it fails on the property instead of on a
    threshold that a loaded machine can trip.
    """

    def _builds(self, calls: int, hash_: KeccakSponge, msg: frx.Array) -> int:
        """How many times `calls` eager hashes construct the marked region."""
        built = 0
        real = sponge_mod.fused_region_over

        def counting(*args: Any, **kwargs: Any) -> Any:
            nonlocal built
            built += 1
            return real(*args, **kwargs)

        # The zone is module-level, so its cache outlives any one test and a
        # sibling case may already hold this key. Cleared so the count is this
        # test's own.
        sponge_mod._fused_hash.clear_cache()
        with (
            dedicated_emitter(permutation_mod),
            mock.patch.object(sponge_mod, "fused_region_over", counting),
        ):
            for _ in range(calls):
                hash_.hash(msg)
        return built

    def test_repeated_eager_calls_build_the_region_once(self) -> None:
        self.assertEqual(self._builds(4, _SHA3_256, _message(1, 64)), 1)

    def test_a_second_shape_builds_its_own(self) -> None:
        """The key is the shape, so a different one is a different region —
        which is what makes the count above a cache hit rather than a no-op."""
        built = 0
        real = sponge_mod.fused_region_over

        def counting(*args: Any, **kwargs: Any) -> Any:
            nonlocal built
            built += 1
            return real(*args, **kwargs)

        sponge_mod._fused_hash.clear_cache()
        with (
            dedicated_emitter(permutation_mod),
            mock.patch.object(sponge_mod, "fused_region_over", counting),
        ):
            _SHA3_256.hash(_message(1, 64))
            _SHA3_256.hash(_message(2, 64))
        self.assertEqual(built, 2)


class MarkerRecognizedTest(absltest.TestCase):
    """`keccak_sponge` reaches the emitter, read off the COMPILED module.

    The whole-hash marker had no recognition case at all, which is why nothing
    noticed it was being emitted onto a backend that declines it.
    """

    def setUp(self) -> None:
        super().setUp()
        if not _HAS_KECCAK_EMITTER:
            self.skipTest(f"no Keccak emitter on {frx.default_backend()}")

    def test_a_whole_hash_compiles_to_a_keccak_sponge_custom_fusion(self) -> None:
        assert_marker_recognized(self, "keccak_sponge", _SHA3_256.hash, _message(1, 64))

    def test_a_multi_block_hash_is_still_one_custom_fusion(self) -> None:
        # Two absorb blocks and two squeezes — the case where the glue between
        # permutes would otherwise sit outside the region.
        assert_marker_recognized(
            self, "keccak_sponge", _SHAKE128_LONG.hash, _message(1, 200)
        )

    def test_a_vmapped_hash_is_still_one_custom_fusion(self) -> None:
        # The rank the emitter has to admit, and the case that has to COMPILE
        # rather than merely lower: the rewriter collapses the leading axes into
        # `B` instead of rejecting the operand. Before it did, this was a hard
        # INVALID_ARGUMENT on GPU that no CPU leg could see, because CPU declines
        # the marker and inlines it before any ABI is checked.
        assert_marker_recognized(
            self, "keccak_sponge", frx.vmap(_SHA3_256.hash), _stacked(3, 2, 64)
        )

    def test_a_vmapped_multi_block_hash_is_still_one_custom_fusion(self) -> None:
        # The same collapse where the region spans two absorb blocks and two
        # squeezes, so a fix that only handled the single-block shape is caught.
        assert_marker_recognized(
            self, "keccak_sponge", frx.vmap(_SHAKE128_LONG.hash), _stacked(2, 3, 200)
        )


class XorIntoRateTest(absltest.TestCase):
    """The merge, whose whole point is that one spelling serves both ranks.

    The one-shot sponge absorbs into `[B, 50]` lanes and the incremental SHAKE
    into an unbatched `(50,)`; those were two transcriptions of this
    slice-and-concatenate until the trailing-axis form replaced both. A version
    indexed on axis 0 would silently merge along the BATCH axis instead, which
    the batched KAT would catch but the unbatched one would not.
    """

    def test_only_the_rate_prefix_changes(self) -> None:
        state = fnp.asarray(np.arange(10, dtype=np.uint32))
        block = fnp.asarray(np.array([0xFF, 0xFF, 0xFF], dtype=np.uint32))
        want = np.arange(10, dtype=np.uint32)
        want[:3] ^= np.uint32(0xFF)
        np.testing.assert_array_equal(
            np.asarray(sponge_mod._xor_into_rate(state, block)), want
        )

    def test_one_spelling_serves_both_state_ranks(self) -> None:
        row = np.arange(10, dtype=np.uint32)
        blk = np.array([1, 2, 3], dtype=np.uint32)
        unbatched = np.asarray(
            sponge_mod._xor_into_rate(fnp.asarray(row), fnp.asarray(blk))
        )
        batched = np.asarray(
            sponge_mod._xor_into_rate(
                fnp.asarray(np.stack([row, row + 100])),
                fnp.asarray(np.stack([blk, blk])),
            )
        )
        np.testing.assert_array_equal(batched[0], unbatched)

    def test_a_full_width_block_leaves_no_capacity(self) -> None:
        np.testing.assert_array_equal(
            np.asarray(
                sponge_mod._xor_into_rate(
                    fnp.asarray(np.zeros(4, dtype=np.uint32)),
                    fnp.asarray(np.arange(4, dtype=np.uint32)),
                )
            ),
            np.arange(4, dtype=np.uint32),
        )


if __name__ == "__main__":
    absltest.main()

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Incremental SHAKE — byte-exact against `hashlib`, and pytree-threadable.

`hashlib.shake_*` is the agnostic golden, the same one `byte_hashes_test` uses:
it implements FIPS 202 without sharing a line with this tree, so agreement is
agreement with the standard.

Two properties matter here and neither is a value check in the ordinary sense.
The first is that *where the message is split cannot matter* — the schedule
carries a pending buffer whose length is traced, and a split that lands
anywhere but a rate boundary is what exercises it. The second is that the state
threads a `@jit` boundary and a `lax.scan` carry, which is the whole reason it is
a fixed-shape pytree rather than a Python object; a rejection-sampling loop is a
scan whose carry is a squeeze state, so that shape is tested directly rather
than by proxy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import tree_util

from hash_frx.keccak.byte_hashes import (
    SHA3_256_RATE,
    SHA3_SUFFIX,
    SHAKE128_RATE,
    SHAKE256_RATE,
)
from hash_frx.keccak.sponge import KeccakSponge
from hash_frx.keccak.streaming import (
    ShakeAbsorb,
    ShakeBlockSqueeze,
    ShakeSqueeze,
    shake128_init,
    shake256_init,
    shake_init,
)
from hash_frx.testing.marker_recognized import emitted_composites

# Long enough to span several blocks at either rate.
_MESSAGE = bytes((i * 7 + 3) & 0xFF for i in range(400))

_Init = Callable[[], ShakeAbsorb]
# `hashlib.shake_*` returns a private `_Hash`-alike; what is used is `.digest(n)`.
_Reference = Callable[[bytes], Any]

# (name, init, rate, hashlib factory)
_CASES = (
    ("shake128", shake128_init, SHAKE128_RATE, hashlib.shake_128),
    ("shake256", shake256_init, SHAKE256_RATE, hashlib.shake_256),
)


def _u8(data: bytes) -> frx.Array:
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _absorb_all(init: _Init, chunks: list[bytes]) -> ShakeAbsorb:
    state = init()
    for chunk in chunks:
        state = state.absorb(_u8(chunk))
    return state


def _squeeze_once(init: _Init, chunks: list[bytes], nbytes: int) -> bytes:
    _, out = _absorb_all(init, chunks).finalize().squeeze(nbytes)
    return bytes(np.asarray(out))


def _squeeze_blocks_lowering(rate: int, nblocks: int) -> str:
    """The lowered module for a whole-block squeeze, absorb kept out of it.

    The squeezer is built eagerly and lowered as an argument, so the module holds
    the squeeze alone — the shape `SqueezePermutationCountTest` uses for the
    general squeezer.
    """
    squeezer = shake_init(rate).finalize_blocks()
    return frx.jit(lambda s: s.squeeze_blocks(nblocks)).lower(squeezer).as_text()


class ShakeStreamTest(parameterized.TestCase):
    @parameterized.parameters(*_CASES)
    def test_single_absorb_matches_hashlib(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # Lengths straddling the rate: empty, one short of a block (so the domain
        # suffix and the pad's closing bit share a byte), exactly a block (so the
        # padding takes a whole extra one), and multi-block.
        for length in (0, 1, rate - 1, rate, rate + 1, 2 * rate, 2 * rate + 1, 400):
            with self.subTest(length=length):
                msg = _MESSAGE[:length]
                self.assertEqual(
                    _squeeze_once(init, [msg], 32), reference(msg).digest(32)
                )

    @parameterized.parameters(*_CASES)
    def test_every_split_point_gives_the_same_digest(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # The point of the pending buffer: the cut lands on every residue mod the
        # rate exactly once, plus two past the wrap, so no pending length goes
        # unexercised. Each cut is a distinct static length and therefore its own
        # trace, so sweeping the residues twice doubles the cost for no coverage.
        msg = _MESSAGE[: rate + 2]
        want = reference(msg).digest(32)
        for cut in range(len(msg) + 1):
            with self.subTest(cut=cut):
                self.assertEqual(_squeeze_once(init, [msg[:cut], msg[cut:]], 32), want)

    @parameterized.parameters(*_CASES)
    def test_three_way_split_gives_the_same_digest(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # Two pending carries in a row, so the second absorb starts from a
        # non-zero pending length rather than from a fresh state.
        msg = _MESSAGE[: 2 * rate + 8]
        want = reference(msg).digest(32)
        for a in (0, 1, rate - 1, rate, rate + 1):
            for b in (0, 1, rate // 2, rate):
                with self.subTest(first=a, second=b):
                    self.assertEqual(
                        _squeeze_once(
                            init, [msg[:a], msg[a : a + b], msg[a + b :]], 32
                        ),
                        want,
                    )

    @parameterized.parameters(*_CASES)
    def test_repeated_squeezes_equal_one_squeeze_of_the_total(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # The rejection-sampling access pattern: take a little at a time and get
        # the same stream as one long take. Splits chosen to leave the offset
        # mid-block, on a boundary, and past several blocks.
        for parts in (
            [32, 32],
            [1, 31, 32],
            [rate, rate],
            [rate - 1, 1, 1],
            [200, 200],
        ):
            with self.subTest(parts=tuple(parts)):
                squeezer = _absorb_all(init, [_MESSAGE]).finalize()
                got = b""
                for part in parts:
                    squeezer, out = squeezer.squeeze(part)
                    got += bytes(np.asarray(out))
                self.assertEqual(got, reference(_MESSAGE).digest(sum(parts)))

    def test_the_suffix_is_the_domain_and_not_a_constant(self) -> None:
        # What this streams is a sponge; SHAKE is the domain the two `init`
        # helpers pick. `shake_init` takes the FIPS 202 domain byte, and at
        # SHAKE256's rate the SHA-3 byte gives SHA3-256 — the standard's own
        # construction, and the reason the suffix is a parameter rather than a
        # constant. Nothing else reaches this argument.
        state = shake_init(SHA3_256_RATE, SHA3_SUFFIX).absorb(_u8(_MESSAGE))
        _, out = state.finalize().squeeze(32)
        self.assertEqual(bytes(np.asarray(out)), hashlib.sha3_256(_MESSAGE).digest())


class PytreeThreadingTest(parameterized.TestCase):
    """The reason this is a registered dataclass rather than a Python object."""

    @parameterized.parameters(*_CASES)
    def test_treedef_is_stable_across_absorbs(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # An unstable treedef re-traces the enclosing zone on every hand-off,
        # which does not error — it just makes every call slow.
        fresh = tree_util.tree_structure(init())
        used = tree_util.tree_structure(init().absorb(_u8(_MESSAGE[:100])))
        self.assertEqual(fresh, used)

    def test_the_two_rates_are_distinct_treedefs(self) -> None:
        # `rate` rides as static aux, so a SHAKE128 state cannot be substituted
        # into a zone traced for SHAKE256.
        self.assertNotEqual(
            tree_util.tree_structure(shake128_init()),
            tree_util.tree_structure(shake256_init()),
        )

    @parameterized.parameters(*_CASES)
    def test_the_whole_pipeline_traces(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        @frx.jit
        def run(data: frx.Array) -> frx.Array:
            _, out = init().absorb(data).finalize().squeeze(64)
            return out

        got = bytes(np.asarray(run(_u8(_MESSAGE))))
        self.assertEqual(got, reference(_MESSAGE).digest(64))

    @parameterized.parameters(*_CASES)
    def test_absorb_threads_a_scan_carry(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        chunks = np.frombuffer(_MESSAGE, dtype=np.uint8).reshape(8, 50)

        def body(carry: ShakeAbsorb, chunk: frx.Array) -> tuple[ShakeAbsorb, None]:
            return carry.absorb(chunk), None

        final, _ = frx.lax.scan(body, init(), fnp.asarray(chunks))
        _, out = final.finalize().squeeze(32)
        self.assertEqual(bytes(np.asarray(out)), reference(_MESSAGE).digest(32))

    @parameterized.parameters(*_CASES)
    def test_squeeze_threads_a_scan_carry(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # This is the shape ML-DSA's sampling loop has: squeeze a block, test it,
        # squeeze again — with the sponge as the carry.
        squeezer = _absorb_all(init, [_MESSAGE]).finalize()

        def body(carry: ShakeSqueeze, _: None) -> tuple[ShakeSqueeze, frx.Array]:
            carry, out = carry.squeeze(32)
            return carry, out

        _, outs = frx.lax.scan(body, squeezer, None, length=5)
        got = b"".join(bytes(row) for row in np.asarray(outs))
        self.assertEqual(got, reference(_MESSAGE).digest(160))


class ShakeBlockSqueezeTest(parameterized.TestCase):
    """The whole-block squeezer: same bytes, without the offset's two costs.

    Byte-equality against the general squeezer is the gate — the narrow type is
    an optimization and may not move a single output byte. The structural cases
    below are what make it worth having, and they are read off the lowered
    module because no value test can see a `select` that computes the right
    answer twice.
    """

    @parameterized.parameters(*_CASES)
    def test_blocks_match_the_general_squeezer(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        for nblocks in (1, 2, 3):
            with self.subTest(nblocks=nblocks):
                wide = _squeeze_once(init, [_MESSAGE], nblocks * rate)
                _, out = (
                    _absorb_all(init, [_MESSAGE])
                    .finalize_blocks()
                    .squeeze_blocks(nblocks)
                )
                narrow = bytes(np.asarray(out))
                self.assertEqual(narrow, wide)
                self.assertEqual(narrow, reference(_MESSAGE).digest(nblocks * rate))

    @parameterized.parameters(*_CASES)
    def test_k_blocks_cost_k_permutations(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # The state's rate prefix is block 0, so the permutation after the last
        # block is the one that leaves the carry pointing at the next.
        for nblocks in (1, 2, 3):
            with self.subTest(nblocks=nblocks):
                squeezer = shake_init(rate).finalize_blocks()
                emitted = emitted_composites(
                    lambda s: s.squeeze_blocks(nblocks), squeezer
                )
                self.assertEqual(len(emitted), nblocks)

    @parameterized.parameters(*_CASES)
    def test_the_offset_costs_are_gone(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # The two things the offset buys and a block-aligned caller does not
        # want: a traced slice of the output, and a chain-select for the carry.
        #
        # The slice is asserted absolutely — the block form has none. The selects
        # are asserted as a DIFFERENCE, because Keccak-f's own body carries 97 of
        # them per permutation, so an absolute `assertNotIn` here would be
        # asserting something about the permutation rather than the schedule.
        narrow = _squeeze_blocks_lowering(rate, 2)
        self.assertNotIn("dynamic_slice", narrow)
        self.assertNotIn("dynamic-slice", narrow)

        squeezer = shake_init(rate).finalize()
        wide = frx.jit(lambda s: s.squeeze(2 * rate)).lower(squeezer).as_text()
        self.assertLess(
            narrow.count("stablehlo.select"), wide.count("stablehlo.select")
        )

    @parameterized.parameters(*_CASES)
    def test_the_carry_is_a_narrower_pytree(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # The point of a distinct type rather than a view: the sampler's scan
        # carry is the squeezer, so the offset has to leave the TREEDEF, not
        # just go unread.
        blocks = init().finalize_blocks()
        wide = init().finalize()
        self.assertNotEqual(
            tree_util.tree_structure(blocks), tree_util.tree_structure(wide)
        )
        self.assertEqual(len(tree_util.tree_leaves(blocks)), 1)
        self.assertEqual(len(tree_util.tree_leaves(wide)), 2)

    @parameterized.parameters(*_CASES)
    def test_it_threads_a_sampler_shaped_scan(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # ML-DSA's RejNTTPoly shape: one rate block per iteration, sponge as the
        # carry. The carry must stay fixed-shape across iterations or the scan
        # will not trace at all.
        squeezer = _absorb_all(init, [_MESSAGE]).finalize_blocks()

        def body(
            carry: ShakeBlockSqueeze, _: None
        ) -> tuple[ShakeBlockSqueeze, frx.Array]:
            carry, out = carry.squeeze_blocks(1)
            return carry, out

        _, outs = frx.lax.scan(body, squeezer, None, length=4)
        got = b"".join(bytes(row) for row in np.asarray(outs))
        self.assertEqual(got, reference(_MESSAGE).digest(4 * rate))

    @parameterized.parameters(*_CASES)
    def test_widen_lands_where_the_general_squeezer_would(
        self, name: str, init: _Init, rate: int, reference: _Reference
    ) -> None:
        # A caller that takes whole blocks and then an odd tail: widening after
        # k blocks must equal squeezing k*rate + tail straight through.
        narrow, first = (
            _absorb_all(init, [_MESSAGE]).finalize_blocks().squeeze_blocks(2)
        )
        _, tail = narrow.widen().squeeze(37)
        got = bytes(np.asarray(first)) + bytes(np.asarray(tail))
        self.assertEqual(got, reference(_MESSAGE).digest(2 * rate + 37))

    def test_squeeze_blocks_rejects_an_empty_request(self) -> None:
        with self.assertRaises(ValueError):
            shake256_init().finalize_blocks().squeeze_blocks(0)


class ShakeStreamValidationTest(absltest.TestCase):
    def test_absorb_rejects_a_batched_message(self) -> None:
        # The streaming state is one sponge; a batch axis would silently absorb
        # the rows concatenated.
        with self.assertRaises(ValueError):
            shake256_init().absorb(fnp.zeros((2, 64), dtype=fnp.uint8))

    def test_absorb_rejects_a_message_that_is_not_bytes(self) -> None:
        # Coerced instead, a value above 255 is truncated and the sponge returns
        # a well-formed digest of a message the caller never passed.
        with self.assertRaises(TypeError):
            shake256_init().absorb(fnp.asarray(np.array([300, 65], dtype=np.int32)))

    def test_squeeze_rejects_an_empty_request(self) -> None:
        with self.assertRaises(ValueError):
            shake256_init().finalize().squeeze(0)


class SqueezePermutationCountTest(parameterized.TestCase):
    """`squeeze(n)` emits exactly the permutations a traced offset can reach
    (#213): floor((rate - 1 + n) / rate), never the ceil.

    Byte-identity tests cannot see this — the dead permutation fed an
    unreachable select arm and every output byte stayed right — so the count
    is asserted on the lowered text, the way the fusion contract's marker
    tests do. The whole-block row (n == rate) is the one a rejection
    sampler's loop body squeezes, where the ceil doubled the real work.
    """

    @parameterized.parameters(
        (1, 1),  # n ≡ 1 (mod rate): floor == ceil, the last permute is live
        (32, 1),
        (167, 1),
        (168, 1),  # whole block: one permutation, not two
        (169, 2),  # 169 = rate + 1: the second permute is reachable
        (336, 2),
    )
    def test_emits_only_reachable_permutations(self, nbytes: int, want: int) -> None:
        squeezer = shake128_init().absorb(fnp.zeros(34, dtype=fnp.uint8)).finalize()
        text = frx.jit(lambda s: s.squeeze(nbytes)).lower(squeezer).as_text()
        emitted = text.count(
            'stablehlo.composite "hash_frx.perm.keccak_f"'
        ) + text.count('stablehlo.composite "zorch.fused_region"')
        self.assertEqual(emitted, want)


class ShakeInitValidationTest(absltest.TestCase):
    """`shake_init` shares the one-shot sponge's parameter checks (#215).

    Before, a rate of 0 returned a well-formed garbage state, a rate of 7 failed
    deep inside the packer, and a suffix with bit 7 set silently became a
    different domain — differently in each path, since the one-shot tail ORs the
    `0x80` terminator while the traced block XORs it.
    """

    def test_rejects_a_rate_that_is_not_a_positive_lane_multiple(self) -> None:
        for rate in (0, -8, 7):
            with self.assertRaisesRegex(ValueError, "rate"):
                shake_init(rate)

    def test_rejects_a_suffix_that_collides_with_the_terminator(self) -> None:
        for suffix in (0x80, 0x9F, 0xFF):
            with self.assertRaisesRegex(ValueError, "suffix"):
                shake_init(SHAKE128_RATE, suffix)

    def test_the_one_shot_sponge_rejects_the_same_suffix(self) -> None:
        with self.assertRaisesRegex(ValueError, "suffix"):
            KeccakSponge(rate=SHAKE128_RATE, suffix=0x9F, output_size=32)


if __name__ == "__main__":
    absltest.main()

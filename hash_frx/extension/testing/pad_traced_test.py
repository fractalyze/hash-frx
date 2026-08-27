# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`nblocks` under a tracer — the property the runtime-length paths stand on.

`pad.py` is frx-free and `pad_test.py` keeps it that way, which is what lets a
padding rule be read and tested without a device. But `nblocks` carries a claim
that only a backend can check: it is the member a RUNTIME-LENGTH path sizes its
block loop from, so it has to survive a traced length where `tail` — a host
constant built from a static one — cannot.

That claim is why this file exists separately rather than as a few cases in
`pad_test.py`: pinning it costs an frx dependency, and only these cases need one.

The failure it guards is not a wrong number. A Python `if` or `bool()` on a
tracer raises at trace time, so a rule that reaches for one is not slightly
wrong on the traced path — it has no traced path at all, and nothing on the
host path can tell.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.extension.md import padded_message_region, trailer_field
from hash_frx.extension.pad import PadRule, SpongePad
from hash_frx.extension.testing.rules import (
    ASCON_PAD,
    BLAKE2B,
    BLAKE2S,
    SHA3_256_PAD,
    SHA256,
    SHA512,
    TRAILER_RULES,
)


class NblocksSurvivesATracerTest(parameterized.TestCase):
    """Every rule answers `nblocks` for a length that is only known at runtime,
    and answers it with the same number the host path gives."""

    @parameterized.named_parameters(
        ("sha256", SHA256),
        ("sha512", SHA512),
        ("blake2s", BLAKE2S),
        ("blake2b", BLAKE2B),
        ("ascon", ASCON_PAD),
        ("sha3_256", SHA3_256_PAD),
    )
    def test_traced_length_matches_the_host_answer(
        self, rule: PadRule | SpongePad
    ) -> None:
        traced = frx.jit(lambda ln: fnp.asarray(rule.nblocks(ln)))
        for length in (0, 1, 63, 64, 65, 127, 128, 129, 271, 272):
            with self.subTest(length=length):
                self.assertEqual(int(traced(np.int32(length))), rule.nblocks(length))


class NblocksSizesALoopBoundTest(absltest.TestCase):
    """The shape a runtime-length caller actually writes: the block count times
    the block size, as `md.padded_message_region` spells it. A rule that
    only works on host ints fails here rather than at the first family that
    tries to adopt the ABI."""

    def test_active_bytes_from_a_traced_length(self) -> None:
        # `ln` is annotated `int` because that is what `nblocks` declares, and
        # what `sha256`'s own runtime-length decomposition passes it -- a traced
        # `Array`. The declaration is the host contract; mypy cannot see the
        # difference either way, since frx's stubs collapse `Array` to `Any`.
        def active(ln: int) -> object:
            return BLAKE2S.nblocks(ln) * fnp.int32(BLAKE2S.block_size)

        self.assertEqual(int(frx.jit(active)(np.int32(65))), 128)


class TrailerFieldMirrorsTheHostTailTest(parameterized.TestCase):
    """`md.trailer_field` against `PadRule.tail`, rule by rule.

    The static tail is the oracle rather than a second transcription of the
    standards: it is already held to all seven families by `pad_test`, and what
    is under test here is only that the TRACED path reaches the same bytes. So a
    disagreement is the traced branch being wrong, never both being wrong
    together — which is what a hand-written per-family field could not claim,
    since it had nothing to be compared against.

    Every axis is covered because every axis is a shipped rule: bit length
    against block count, big-endian against little, and eight reserved bytes
    against sixteen. `Trailer.NONE` is the one rule with no field at all.
    """

    @parameterized.named_parameters(*TRAILER_RULES)
    def test_matches_the_last_reserved_bytes_of_the_host_tail(
        self, pad: PadRule
    ) -> None:
        # Lengths across the block boundary in both directions, including the
        # empty message and one that pushes `nblocks` up a block.
        for length in (0, 1, 55, 56, 63, 64, 65, 119, 120, 1000):
            with self.subTest(length=length):
                want = pad.tail(length)[-pad.reserve :]
                got = np.asarray(trailer_field(pad, fnp.int32(length)))
                np.testing.assert_array_equal(got, want)

    @parameterized.named_parameters(*TRAILER_RULES)
    def test_survives_a_tracer(self, pad: PadRule) -> None:
        # The reason this file exists: the field is built for a length only the
        # runtime knows, so a branch that reaches for `bool()` on the count has
        # no traced path at all and the host cases above cannot tell.
        traced = frx.jit(lambda n: trailer_field(pad, n))(fnp.int32(120))
        np.testing.assert_array_equal(np.asarray(traced), pad.tail(120)[-pad.reserve :])

    @parameterized.named_parameters(("sha256", SHA256), ("sha512", SHA512))
    def test_a_bit_length_matches_the_standard_encoding(self, pad: PadRule) -> None:
        # Reaches where the host tail cannot be the oracle: `PadRule.tail`
        # memoizes per length, so a gigabyte case would materialize a gigabyte
        # of padding to check eight bytes. Against the arithmetic instead.
        for count in (1 << 20, (1 << 29) - 1, 1 << 29, (1 << 31) - 1):
            with self.subTest(count=count):
                got = np.asarray(trailer_field(pad, fnp.int32(count))).tobytes()
                self.assertEqual(got, (count * 8).to_bytes(pad.reserve, "big"))

    @parameterized.named_parameters(("sha256", SHA256), ("sha512", SHA512))
    def test_a_bit_length_survives_the_int32_counter_wrap(self, pad: PadRule) -> None:
        """Past 2 GiB the int32 byte counter is negative, and the encoding is
        still right: it reinterprets the wrapped value as a uint32, which is the
        same bit pattern. That is why the ceiling is 4 GiB, not 2.

        Held once on the rule rather than once per family: SHA-256 and SHA-512
        each carried this case against their own copy of the field, and the two
        copies were the thing that could drift."""
        for count in (1 << 31, (1 << 31) + 12345, (1 << 32) - 1):
            with self.subTest(count=count):
                wrapped = np.asarray(count & 0xFFFFFFFF, np.uint32).astype(np.int32)
                got = np.asarray(trailer_field(pad, fnp.asarray(wrapped))).tobytes()
                self.assertEqual(got, (count * 8).to_bytes(pad.reserve, "big"))

    @parameterized.named_parameters(("blake2s", BLAKE2S), ("blake2b", BLAKE2B))
    def test_haifa_has_no_field_to_build(self, pad: PadRule) -> None:
        # Rejected rather than answered with zeros: HAIFA's length reaches the
        # compression as a counter, so a caller asking for a field here has
        # confused the two arrangements.
        with self.assertRaisesRegex(ValueError, "Trailer.NONE has no length field"):
            trailer_field(pad, fnp.int32(64))


class PaddedMessageRegionMirrorsTheHostPathTest(parameterized.TestCase):
    """`md.padded_message_region` against the static padding, rule by rule.

    The field got a rule-parameterized test when it was extracted; the region
    did not, and it is exercised end to end only by SHA-256 and Grøstl — two
    rules that agree on all three axes (`block_size` 64, `reserve` 8,
    big-endian). So the `reserve = 16` and little-endian paths THROUGH THE
    REGION would first execute when SHA-512 and RIPEMD-160 adopt the ABI, which
    is the opposite of the claim the extraction makes: that a new family is a
    `PadRule` and nothing else.

    The oracle is the static path — `msg ‖ pad.tail(len)` — for the reason the
    field's test gives: `pad_test` already holds it to all seven families, so a
    disagreement is the traced side being wrong rather than both being wrong
    together.
    """

    @parameterized.named_parameters(*TRAILER_RULES)
    def test_the_live_prefix_is_the_statically_padded_message(
        self, pad: PadRule
    ) -> None:
        rng = np.random.default_rng(0)
        for length in (0, 1, pad.block_size - 9, pad.block_size, pad.block_size + 1):
            for slack in (0, 1, pad.block_size):
                with self.subTest(length=length, capacity=length + slack):
                    width = max(length + slack, 1)
                    buf = np.full((2, width), 0xFF, dtype=np.uint8)
                    msg = rng.integers(0, 256, size=(2, length), dtype=np.uint8)
                    buf[:, :length] = msg
                    got = np.asarray(
                        padded_message_region(pad, fnp.asarray(buf), fnp.int32(length))
                    )
                    # The region is as wide as the BUFFER could need; what the
                    # message occupies is the first `nblocks(length)` blocks,
                    # and only those are read by the caller's masked loop.
                    live = pad.nblocks(length) * pad.block_size
                    want = np.concatenate(
                        [msg, np.broadcast_to(pad.tail(length), (2, live - length))],
                        axis=1,
                    )
                    np.testing.assert_array_equal(got[:, :live], want)

    @parameterized.named_parameters(("blake2s", BLAKE2S), ("blake2b", BLAKE2B))
    def test_a_haifa_rule_is_refused_at_the_region(self, pad: PadRule) -> None:
        # Refused here rather than two frames down in `trailer_field`: HAIFA
        # wants neither the 0x80 byte this region writes nor a length field, so
        # the message a caller sees should name the region, not the field.
        with self.assertRaisesRegex(ValueError, "needs a rule with a trailer"):
            padded_message_region(
                pad, fnp.asarray(np.zeros((1, 64), dtype=np.uint8)), fnp.int32(8)
            )


if __name__ == "__main__":
    absltest.main()

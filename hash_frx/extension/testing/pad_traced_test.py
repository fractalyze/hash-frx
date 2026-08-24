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

from hash_frx.extension.pad import PadRule, SpongePad, Trailer

SHA256 = PadRule(64, Trailer.BIT_LENGTH)
SHA512 = PadRule(128, Trailer.BIT_LENGTH, reserve=16)
BLAKE2S = PadRule(64, Trailer.NONE)
BLAKE2B = PadRule(128, Trailer.NONE)
ASCON_PAD = SpongePad(rate=8, head=0x01, final_bit=False)
SHA3_256_PAD = SpongePad(rate=136, head=0x06)


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
    the block size, as `sha256._runtime_padded_words` spells it. A rule that
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


if __name__ == "__main__":
    absltest.main()

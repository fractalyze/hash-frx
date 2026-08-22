# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What `MdStream.absorb` costs, for both families that run it.

The absorb's live block count depends on `pending_len`, which is runtime data,
so it computes both static candidates and selects. Running each from the base
midstate costs `min + max` compressions; continuing the high one from the low
one costs `min + 1`, because Merkle-Damgard composes.

Nothing else can see the difference — the digests are identical either way — so
the count is asserted directly. It lives here rather than in either family's
stream test because after the extraction it is a property of the shared
schedule, and asserting it for SHA-256 alone left SHA-512 running the identical
code untested.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx import sha256, sha512

# (family, block size, marker, words per block, absorb entry points)
_FAMILIES = (
    (
        "sha256",
        64,
        sha256.SHA256_MARKER,
        16,
        sha256.sha256_stream_init,
        sha256.sha256_stream_absorb,
    ),
    (
        "sha512",
        128,
        sha512.SHA512_MARKER,
        32,
        sha512.sha512_stream_init,
        sha512.sha512_stream_absorb,
    ),
)


def _blocks_compressed(
    marker: str,
    words: int,
    init: Callable[[], Any],
    absorb: Callable[[Any, Any], Any],
    length: int,
) -> int:
    """Total blocks fed to the marked chain by one absorb of `length` bytes."""
    data = fnp.asarray(np.arange(length, dtype=np.uint8))
    text = frx.jit(lambda d: absorb(init(), d)).lower(data).as_text()
    pattern = rf'composite "{re.escape(marker)}".*?tensor<1x(\d+)x{words}xui32>'
    return sum(int(n) for n in re.findall(pattern, text))


class AbsorbCostTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (f"{name}_{length}", name, block, marker, words, init, absorb, length)
        for name, block, marker, words, init, absorb in _FAMILIES
        for length in (0, 1, 100, 200, 1000, 4097)
    )
    def test_absorb_costs_min_plus_one_at_most(
        self,
        name: str,
        block: int,
        marker: str,
        words: int,
        init: Callable[[], Any],
        absorb: Callable[[Any, Any], Any],
        length: int,
    ) -> None:
        got = _blocks_compressed(marker, words, init, absorb, length)
        # min + 1 is the bound the prefix form buys; the two-candidate form
        # would spend min + max, which is nearly twice this for a long absorb.
        self.assertLessEqual(got, length // block + 1)

    @parameterized.named_parameters(*_FAMILIES)
    def test_a_block_multiple_costs_exactly_its_blocks(
        self,
        block: int,
        marker: str,
        words: int,
        init: Callable[[], Any],
        absorb: Callable[[Any, Any], Any],
    ) -> None:
        # Where `min == max` only one candidate ever existed, so the prefix form
        # must not add work — the half of an optimization that is easy to break.
        for blocks in (1, 2, 8):
            with self.subTest(blocks=blocks):
                self.assertEqual(
                    _blocks_compressed(marker, words, init, absorb, blocks * block),
                    blocks,
                )

    @parameterized.named_parameters(*_FAMILIES)
    def test_a_long_non_multiple_beats_the_two_candidate_form(
        self,
        block: int,
        marker: str,
        words: int,
        init: Callable[[], Any],
        absorb: Callable[[Any, Any], Any],
    ) -> None:
        length = 16 * block + 1
        got = _blocks_compressed(marker, words, init, absorb, length)
        two_candidate = length // block + (block - 1 + length) // block
        self.assertEqual(got, length // block + 1)
        self.assertLess(got, two_candidate)


if __name__ == "__main__":
    absltest.main()

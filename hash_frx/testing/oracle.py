# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`oracle_digest` — a one-message hash applied per row, for differential tests.

A device row is data-parallel over axis B; an oracle is not. Comparing the two
is therefore two claims at once: that the values agree, and that one batched
call equals the per-message digests in order. That pairing is what the removed
host rows used to provide as a side effect of being rows, and it is worth
keeping — as a test utility rather than as shipped surface, because nothing but
a test wants a Python loop over a batch.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def oracle_digest(
    hash_one: Callable[[bytes], bytes],
    digest_size: int,
    msgs: np.ndarray,
) -> np.ndarray:
    """`hash_one` per row of `msgs`, as uint8 `[B, digest_size]`.

    Preallocated rather than built from a list, so a zero-row batch returns the
    right shape without a reshape to rescue it.
    """
    out = np.empty((len(msgs), digest_size), dtype=np.uint8)
    for i, row in enumerate(msgs):
        out[i] = np.frombuffer(hash_one(bytes(row)), dtype=np.uint8)
    return out

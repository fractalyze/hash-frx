# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Merkle-Damgard: the schedule that feeds a `CompressionFunction` a message.

Seven families in this package run this schedule and, until now, each ran its
own copy of it. The clearest measure was `_padding_tail`, defined nine times
across the tree — seven here plus the two sponges — with six of the seven MD
docstrings citing `sha256._padding_tail` as the arrangement they reproduce.

**The rule is parameterized against all seven, not generalized from SHA-2.**
That distinction is the one #169 paid for by deferring the byte-sponge seam
until a second byte sponge existed: a surface shaped from one implementation
encodes that implementation's accidents as if they were the family's. Reading
all seven first is what surfaced these, and only the first would have survived
a SHA-2-derived design:

- the length field is **little**-endian in RIPEMD-160 and big-endian elsewhere;
- SHA-512 reserves **16** bytes for a 128-bit length field while writing 8;
- Grostl's trailer is the **block count**, not the bit length;
- BLAKE2b/2s have no trailer and no 0x80 at all — HAIFA carries the length as a
  counter into the compression, so their padding is a zero-fill to the block.

`PadRule` names those axes and nothing else, so a family is its compression
function plus four numbers rather than its own transcription of this file.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


class Trailer(enum.Enum):
    """What the last 8 bytes of the padding encode.

    `NONE` is HAIFA: no strengthening bytes at all, because the length reaches
    the compression as a counter operand instead of through the message.
    """

    BIT_LENGTH = "bit_length"  # SHA-2, RIPEMD-160, SM3
    BLOCK_COUNT = "block_count"  # Grostl, which counts blocks rather than bits
    NONE = "none"  # BLAKE2's HAIFA padding


@dataclass(frozen=True)
class PadRule:
    """How a message becomes a whole number of blocks.

    block_size : bytes per block.
    trailer    : what the final 8 bytes encode, or `NONE` for HAIFA.
    reserve    : bytes the length field must fit in when sizing the last block.
                 8 everywhere except SHA-512, whose standard reserves 16 for a
                 128-bit field it never fills past 64 bits.
    big_endian : byte order of the trailer. RIPEMD-160 is the little-endian one.
    """

    block_size: int
    trailer: Trailer
    reserve: int = 8
    big_endian: bool = True

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.block_size % 8:
            raise ValueError(
                f"block_size must be a positive multiple of 8, got {self.block_size}"
            )
        if self.trailer is not Trailer.NONE and self.reserve < 8:
            raise ValueError(
                f"a trailer needs at least 8 reserved bytes, got {self.reserve}"
            )

    # Memoized because all seven families memoized their own copy: a digest
    # rebuilds the tail on every trace otherwise, and the rule is a frozen
    # dataclass over four hashable fields, so keying on `self` is sound. The
    # returned array is shared, and every caller passes it to `fnp.asarray`
    # rather than mutating it.
    @lru_cache(maxsize=None)
    def tail(self, length: int) -> np.ndarray:
        """The bytes appended to a `length`-byte message, built from the length
        alone.

        Data-independent by construction, which is what lets `digest` take a
        traced message: the tail is a host constant threaded through the marked
        region as an operand, never something read off the message.
        """
        if self.trailer is Trailer.NONE:
            # HAIFA: zero-fill to a block, and a whole empty block for the empty
            # message, since the compression still has to run once.
            if length == 0:
                return np.zeros(self.block_size, dtype=np.uint8)
            return np.zeros(-length % self.block_size, dtype=np.uint8)

        nblocks = (length + self.reserve) // self.block_size + 1
        tail = np.zeros(nblocks * self.block_size - length, dtype=np.uint8)
        tail[0] = 0x80
        value = length * 8 if self.trailer is Trailer.BIT_LENGTH else nblocks
        encoded = (
            int(value).to_bytes(8, "big")
            if self.big_endian
            else int(value).to_bytes(8, "little")
        )
        tail[-8:] = np.frombuffer(encoded, dtype=np.uint8)
        return tail

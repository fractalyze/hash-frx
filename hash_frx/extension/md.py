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
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import frx.numpy as fnp
import numpy as np
from frx import Array

from hash_frx.fusion import fused_region


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


def chain(
    h0: Array,
    blocks: Array,
    *,
    constants: Array,
    compress_block: Callable[[Array, Array, Array], Array],
    serialize: Callable[[Array], Array],
    state_words: int,
    marker: tuple[str, int],
) -> Array:
    """The compression chain from midstate `h0` over `blocks`, as one marked
    region: `[B, nblocks, ...]` -> the serialized final state.

    **The loop is the schedule, so it lives here.** `compress_block` absorbs one
    block; walking the blocks is Merkle-Damgard's job, not the primitive's, and
    every family that had its own copy of this wrote the same `for i in
    range(nblocks)` over a static, small count.

    `h0` is the shared midstate — an initial state for a whole-message digest,
    or a resumed one, which is what lets a streaming transcript and a batch
    digest share a single marker. It broadcasts across the batch here rather
    than being materialized per row.

    **The operand order is the wire ABI.** `[h0, constants, blocks]` is what the
    recognizing emitters read positionally, so all three are passed explicitly
    rather than captured: a captured constant is lifted into an unnamed operand
    ahead of the declared ones and lands at position 0, silently handing the
    emitter a different message (`hash_frx.fusion` states the rule). That is why
    `constants` is threaded through untouched — the chain never reads the table,
    it only has to put it in the right place.

    `marker` is the (name, version) pair the region carries. A whole-hash MD
    marker is exempt from the generic single-kernel rule the way a permutation's
    is: a compression is a round loop rather than straight-line, so it takes a
    name-routed marker an emitter expands. With no emitter wired the marker
    inlines its decomposition and the bytes are unchanged.
    """
    name, version = marker

    def decomposition(
        h0: Array, constants: Array, blocks: Array, **_attrs: object
    ) -> Array:
        state = fnp.broadcast_to(h0, (blocks.shape[0], state_words))
        for i in range(blocks.shape[1]):  # static, small
            state = compress_block(state, blocks[:, i], constants)
        return serialize(state)

    return fused_region(
        decomposition, h0, constants, blocks, name=name, version=version
    )

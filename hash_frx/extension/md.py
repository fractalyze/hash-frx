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
from typing import Any, Protocol

import frx
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


class _Midstate(Protocol):
    """What every family's streaming state carries, whatever it calls itself.

    The state stays a per-family pytree — `Sha256State` and `Sha512State` are
    registered types a consumer threads through `scan`, and collapsing them into
    one would change every consumer's type. What is shared is the SHAPE, so
    `MdStream` codes against this and each family keeps its own class.
    """

    # Read-only, because the concrete states are frozen dataclasses and a
    # Protocol declaring a mutable attribute does not match one.
    @property
    def h(self) -> Array:
        """The midstate over every complete block so far."""

    @property
    def pending(self) -> Array:
        """The trailing partial block, valid prefix `[:pending_len]`."""

    @property
    def counts(self) -> Array:
        """int32[2] = [pending_len, total bytes absorbed]."""

    @property
    def pending_len(self) -> Array:
        """Bytes currently buffered in `pending`."""


@dataclass(frozen=True)
class MdStream:
    """The incremental half of Merkle-Damgard, shared by the SHA-2 families.

    #192 measured these two absorbs at zero differing code lines after rename
    normalization, and re-measuring on the current tree agrees: the eight lines
    that differ are comment reflow and one block size inside a comment. The
    gap-skip gather, the two-candidate compress-and-select and its "the
    discarded candidate is the only one that ever sees the junk tail" rationale
    were transcribed in full, twice.

    Which is the kind of duplication that is a correctness hazard rather than an
    annoyance: this is the subtlest traced-index machinery in the package, a
    consumer runs it (`Sha256FieldTranscript`), and a drift between the copies
    would be a wrong digest rather than a slow one.

    The pieces below are the family's, and the schedule is this class's.
    """

    block_size: int
    block_to_words: Callable[[Array], Array]
    deserialize: Callable[[Array], Array]
    # (h0, blocks) -> serialized final state; the family's own marked chain, so
    # the streaming path and the batch digest go through ONE marker.
    chain: Callable[[Array, Array], Array]
    make_state: Callable[[Array, Array, Array], Any]
    # Bytes the trailing length field occupies: 8 for SHA-256, 16 for SHA-512's
    # 128-bit field. It sets how much of the final block the padding claims, and
    # so how wide a tail `finalize` can still accept.
    length_bytes: int
    length_field: Callable[[Array], Array]

    def absorb(self, state: _Midstate, data: Array) -> Any:
        """Fold every newly-complete block of `data` (uint8 [L], L static) into
        the midstate and keep the remainder pending.

        The block loop is Python-unrolled and active-count-masked over STATIC
        slices — never a traced-index gather or a scan-carry scatter — which is
        the CPU-safe pattern `transcript.DuplexTranscript` uses.
        """
        block = self.block_size
        length = data.shape[0]
        pl = state.pending_len
        combined_src = fnp.concatenate([state.pending, data.astype(fnp.uint8)])
        new_len = pl + fnp.int32(length)
        active_blocks = new_len // block
        max_blocks = (block - 1 + length) // block  # static upper bound

        # Drop the pending buffer's invalid gap [pending_len:block] from the
        # stream: for stream position j the source index is j while
        # j < pending_len, and shifted past the gap after that.
        total_slots = (max_blocks + 1) * block
        pos = fnp.arange(total_slots, dtype=fnp.int32)
        src_idx = pos + fnp.where(pos < pl, fnp.int32(0), block - pl)
        src_idx = fnp.clip(src_idx, 0, combined_src.shape[0] - 1)
        combined = combined_src[src_idx]  # valid prefix [0:new_len]

        # The live block count depends on pending_len by AT MOST one — (pl + L)
        # // block spans {L // block, (block - 1 + L) // block} — so run the
        # chain at both static candidates and select. The discarded candidate is
        # the only one that ever sees the gap-shifted junk tail block.
        h = state.h
        min_blocks = length // block
        if max_blocks == 0:
            h_new = h
        else:
            words = self.block_to_words(
                combined[: max_blocks * block].reshape(1, max_blocks * block)
            )
            h_hi = self.deserialize(self.chain(h, words))[0]
            if min_blocks == max_blocks:
                h_new = h_hi
            else:
                h_lo = (
                    self.deserialize(self.chain(h, words[:, :min_blocks]))[0]
                    if min_blocks > 0
                    else h
                )
                h_new = fnp.where(active_blocks == max_blocks, h_hi, h_lo)

        tail_len = new_len - active_blocks * block
        tail = frx.lax.dynamic_slice(combined, (active_blocks * block,), (block,))
        slot = fnp.arange(block, dtype=fnp.int32)
        pending = fnp.where(slot < tail_len, tail, fnp.uint8(0))
        # One fused counter update: [pending_len', total'] = [tail_len, total+L].
        counts = fnp.stack([tail_len, state.counts[1] + fnp.int32(length)])
        return self.make_state(h_new, pending, counts)

    def finalize(self, state: _Midstate, extras: Array) -> Array:
        """`H(absorbed ‖ extras[b])` for each row of `extras` (uint8 [B, E], E
        static), without mutating the state — the hash finished at the current
        length, once per row, from one shared midstate.

        The trailing content is `pending[:pending_len] ‖ extras[b]`, so with the
        0x80 byte and the length field it spans at most two blocks. Both shapes
        are emitted and the 1-vs-2-block choice selected, because it depends on
        `pending_len`, which is runtime data.

        That is also what bounds `E`: only a tail that fits the two-block layout
        at EVERY reachable `pending_len` is representable, so a wider one is
        rejected rather than silently overlapping the padding. Absorb the prefix
        and finalize with the remainder.
        """
        block, lb = self.block_size, self.length_bytes
        batch, e = extras.shape
        if e > block - lb:
            raise ValueError(
                f"extras width ({e}) must be <= {block - lb}: pending_len is "
                "runtime data, so a wider tail cannot be guaranteed to fit the "
                "two-block finalize layout at every offset — absorb the prefix "
                "instead"
            )
        pl = state.pending_len
        content_len = pl + fnp.int32(e)
        len_bytes = self.length_field(state.counts[1] + fnp.int32(e))

        # One block only if the content, the 0x80 and the length field all fit.
        two_blocks = content_len > fnp.int32(block - lb - 1)
        active_bytes = fnp.where(two_blocks, fnp.int32(2 * block), fnp.int32(block))

        pos = fnp.arange(2 * block, dtype=fnp.int32)
        # content = pending[:pl] ‖ extras[b], skipping the pending gap [pl:block].
        combined_src = fnp.concatenate(
            [fnp.broadcast_to(state.pending, (batch, block)), extras.astype(fnp.uint8)],
            axis=1,
        )
        src_idx = fnp.clip(
            pos + fnp.where(pos < pl, fnp.int32(0), block - pl), 0, block + e - 1
        )
        content = combined_src[:, src_idx]

        is_content = (pos < content_len)[None, :]
        is_pad80 = (pos == content_len)[None, :]
        len_start = active_bytes - fnp.int32(lb)
        is_len = ((pos >= len_start) & (pos < active_bytes))[None, :]
        len_val = len_bytes[fnp.clip(pos - len_start, 0, lb - 1)][None, :]
        region = fnp.where(
            is_content,
            content,
            fnp.where(
                is_pad80, fnp.uint8(0x80), fnp.where(is_len, len_val, fnp.uint8(0))
            ),
        )

        words = self.block_to_words(region)
        return fnp.where(
            two_blocks, self.chain(state.h, words), self.chain(state.h, words[:, :1])
        )

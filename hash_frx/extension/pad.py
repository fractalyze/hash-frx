# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""How a message becomes a whole number of blocks.

Nine families in this package pad, and until now each padded with its own copy
of the rule — `_padding_tail` was defined nine times across the tree, with six
of the seven Merkle-Damgard docstrings citing `sha256._padding_tail` as the
arrangement they reproduce.

**Two rules, not one, and that split was measured rather than assumed.**
Merkle-Damgard's `PadRule` and the sponges' `SpongePad` compute the same shape
of thing — a never-empty fill to the next block boundary — and share three
lines doing it. Folding the sponges into `PadRule` would need a `head` and a
`final_bit` axis dead for all seven MD rows, while `reserve`, `big_endian` and
`Trailer` stay dead for both sponge rows, which `pad_test`'s own axis doctrine
forbids: each axis changes an outcome, or it is a parameter nobody needs. They
are siblings here instead, which is where the shared three lines are cheap
enough to spell twice and the dead axes cost nothing.

**The rule is parameterized against all seven, not generalized from SHA-2.**
That distinction is what the byte-sponge seam paid for by waiting until a
second byte sponge existed: a surface shaped from one implementation encodes
that implementation's accidents as if they were the family's. Reading all seven
first is what surfaced these, and only the first would have survived a
SHA-2-derived design:

- the length field is **little**-endian in RIPEMD-160 and big-endian elsewhere;
- SHA-512 reserves **16** bytes for a 128-bit length field while writing 8;
- Grostl's trailer is the **block count**, not the bit length;
- BLAKE2b/2s have no trailer and no 0x80 at all — HAIFA carries the length as a
  counter into the compression, so their padding is a zero-fill to the block,
  and `haifa_counter` below is the other half of that arrangement.

Host arithmetic over a message LENGTH and never its bytes, so this module pulls
no frx: it is what lets a padding rule be read, and tested, without a device.
The schedule that feeds the padded blocks to a compression is `md.py`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def _frozen(tail: np.ndarray) -> np.ndarray:
    """Mark a memoized tail read-only before it is shared."""
    tail.setflags(write=False)
    return tail


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

    def nblocks(self, length: int) -> int:
        """How many blocks a `length`-byte message pads to, under a trailer.

        Plain `//` and `+`, so it holds over a TRACED length as readily as a host
        one — which is what lets a runtime-length path size its block loop from
        the rule instead of restating `block_size` and `reserve`. Restating them
        is what let a family's batch digest and its streaming finalize disagree
        about where the length field goes (`extension/md.py`), and `reserve`
        is exactly the field that varies: SHA-512 claims 16 bytes where the rest
        claim 8.

        `Trailer.NONE` has no length field to make room for, so its block count
        is not this — `tail` handles that family separately.
        """
        return (length + self.reserve) // self.block_size + 1

    # Memoized because all seven families memoized their own copy: a digest
    # rebuilds the tail on every trace otherwise, and the rule is a frozen
    # dataclass over four hashable fields, so keying on `self` is sound.
    #
    # That keying is by VALUE, which makes the sharing wider than it looks:
    # SHA-256's rule and SM3's are equal, so they share one entry. The array is
    # therefore handed out read-only — a caller that wrote through it would
    # change another family's padding, silently and everywhere after.
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
                return _frozen(np.zeros(self.block_size, dtype=np.uint8))
            return _frozen(np.zeros(-length % self.block_size, dtype=np.uint8))

        nblocks = self.nblocks(length)
        tail = np.zeros(nblocks * self.block_size - length, dtype=np.uint8)
        tail[0] = 0x80
        value = length * 8 if self.trailer is Trailer.BIT_LENGTH else nblocks
        encoded = (
            int(value).to_bytes(8, "big")
            if self.big_endian
            else int(value).to_bytes(8, "little")
        )
        tail[-8:] = np.frombuffer(encoded, dtype=np.uint8)
        return _frozen(tail)


def haifa_counter(
    index: int, nblocks: int, length: int, block_size: int
) -> tuple[int, bool]:
    """BLAKE2's per-block `(t, last)`: bytes hashed through the end of block
    `index`, and whether it is the final one.

    HAIFA is Merkle-Damgard with the length fed to the compression rather than
    into the message, which is why these rows pad with a bare zero fill
    (`Trailer.NONE`) — and why the count here is the TRUE message length on the
    final block rather than the padded one. RFC 7693 §3.2 counts a full block
    per interior block; §3.3 has the final block report the real length, so the
    zero pad is never counted. Getting that wrong is a wrong digest on exactly
    the messages whose length is not a block multiple.

    Shared by BLAKE2b and BLAKE2s, whose copies were identical down to the
    comment. Not folded into a general bytes-in chain: measured across the four
    bytes-in families, such a helper would share three lines out of forty to
    fifty-six and need five callbacks to do it, so the loop stays where it reads.
    """
    last = index == nblocks - 1
    return (length if last else (index + 1) * block_size), last


@dataclass(frozen=True)
class SpongePad:
    """How a message becomes a whole number of sponge blocks: `pad10*1`.

    rate      : bytes absorbed per permutation.
    head      : the byte that opens the padding — a domain-separation suffix
                with the padding's leading `1` packed into the next bit up
                (FIPS 202 sections 5.1 and 6), or the bare `1` where the
                standard defines no domain bits.
    final_bit : whether the block's last byte also carries the padding's
                trailing `1`. FIPS 202's `pad10*1` sets it; Ascon's pad
                (SP 800-232 Algorithm 2) has no trailing bit at all, which is
                the axis that keeps the two rules one dataclass rather than two.

    The pad is never empty, so a rate-aligned message gains a whole padding
    block — the standards' rule, and what makes the block count a function of
    the length alone.

    Host arithmetic over a message LENGTH and never its bytes, like `PadRule`
    above: the tail is a host constant built *from* the length rather than
    written *into* the message, which is what lets a `digest` take a traced
    message.

    Keccak spelled the size as `nblocks * rate - length` for
    `nblocks = length // rate + 1` and Ascon as `rate - length % rate`; the two
    reduce to each other, and this is the surviving spelling.
    """

    rate: int
    head: int
    final_bit: bool = True

    def __post_init__(self) -> None:
        if self.rate < 1:
            raise ValueError(f"rate ({self.rate}) must be >= 1")
        if not 0 <= self.head <= 0xFF:
            raise ValueError(f"head ({self.head:#x}) must be a byte")
        # Bit 7 belongs to `pad10*1`'s trailing 1 wherever there is one, so a
        # head that sets it collides on a one-byte pad. What that collision cost
        # in practice — two spellings of the same byte disagreeing on the digest
        # — is in `keccak.sponge.validate_sponge_params`, which enforces the
        # same rule on the path that does not build a `SpongePad`.
        if self.final_bit and self.head >= 0x80:
            raise ValueError(
                f"head ({self.head:#x}) must be a byte below 0x80: bit 7 carries "
                "pad10*1's trailing 1 (FIPS 202 section 5.1)"
            )

    # Memoized and handed out read-only for the reason `PadRule.tail` states:
    # keying is by VALUE, so two rows with equal parameters share one entry and
    # a caller writing through the array would change the other's padding.
    @lru_cache(maxsize=None)
    def tail(self, length: int) -> np.ndarray:
        """The bytes appended to a `length`-byte message: uint8 [P].

        `head ‖ 0x00* ‖ 0x80` where `final_bit`, else `head ‖ 0x00*`. When the
        message ends one byte short of a block the two ends land on the same
        byte and it becomes `head | 0x80`, which is the standard's single-byte
        pad rather than a special case.
        """
        tail = np.zeros(self.rate - length % self.rate, dtype=np.uint8)
        tail[0] = self.head
        if self.final_bit:
            tail[-1] |= 0x80
        return _frozen(tail)

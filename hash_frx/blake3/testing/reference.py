# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A plain-Python BLAKE3 compression function, as the oracle for the frx one.

Transcribed from the BLAKE3 specification over 32-bit Python integers, sharing
neither the batched lane layout nor the single-kernel authoring rules that shape
`compress.py`. An oracle that shared those would fail the same way the thing it
checks does.

`hash_single_chunk` is here only so the oracle can be anchored — `reference_test`
matches it against the published vectors, and `vectors.py` sets out why those
reach a compression function at all.

Tree mode is deliberately absent: it belongs to the multi-chunk work, and this
file only needs to reach far enough to anchor. So the flags below are hash
mode's; the rest of Table 3 belongs with the modes that use it.
"""

from __future__ import annotations

_MASK = 0xFFFFFFFF
BLOCK_LEN = 64
CHUNK_LEN = 1024

# BLAKE3 spec section 2.2, Table 1 — the SHA-2 IV, unchanged. Fenced from the
# formatter, which would flatten the transcribed table one word per line.
# fmt: off
IV = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)
# fmt: on

# Hash mode's domain-separation flags (spec section 2.2, Table 3).
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
ROOT = 1 << 3

# The message word schedule applied between rounds (spec section 2.2, Table 2).
MSG_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)


def rotr32(value: int, n: int) -> int:
    """Rotate a 32-bit word right by `n`."""
    return ((value >> n) | (value << (32 - n))) & _MASK


def _g(state: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    """The G mixing function, in place on four state words."""
    state[a] = (state[a] + state[b] + mx) & _MASK
    state[d] = rotr32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & _MASK
    state[b] = rotr32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b] + my) & _MASK
    state[d] = rotr32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & _MASK
    state[b] = rotr32(state[b] ^ state[c], 7)


def _round(state: list[int], m: list[int]) -> None:
    """One round: four column mixes, then four diagonal mixes."""
    _g(state, 0, 4, 8, 12, m[0], m[1])
    _g(state, 1, 5, 9, 13, m[2], m[3])
    _g(state, 2, 6, 10, 14, m[4], m[5])
    _g(state, 3, 7, 11, 15, m[6], m[7])
    _g(state, 0, 5, 10, 15, m[8], m[9])
    _g(state, 1, 6, 11, 12, m[10], m[11])
    _g(state, 2, 7, 8, 13, m[12], m[13])
    _g(state, 3, 4, 9, 14, m[14], m[15])


def compress(
    chaining_value: list[int],
    block_words: list[int],
    counter: int,
    block_len: int,
    flags: int,
) -> list[int]:
    """The compression function: 8-word CV + 16-word block -> 16 output words."""
    state = [
        *chaining_value[:8],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter & _MASK,
        (counter >> 32) & _MASK,
        block_len,
        flags,
    ]
    m = list(block_words)
    for r in range(7):
        _round(state, m)
        if r < 6:
            m = [m[i] for i in MSG_PERMUTATION]

    for i in range(8):
        state[i] ^= state[i + 8]
        state[i + 8] ^= chaining_value[i]
    return state


def words_of(block: bytes) -> list[int]:
    """A 64-byte block as 16 little-endian words, zero-padded if short."""
    padded = bytes(block) + bytes(BLOCK_LEN - len(block))
    return [int.from_bytes(padded[4 * i : 4 * i + 4], "little") for i in range(16)]


def blocks_of(data: bytes) -> list[bytes]:
    """A chunk's bytes cut into blocks — an empty chunk still has one block."""
    return [data[i : i + BLOCK_LEN] for i in range(0, len(data), BLOCK_LEN)] or [b""]


def hash_single_chunk(data: bytes) -> bytes:
    """BLAKE3 of an input that fits in one chunk (<= 1024 bytes): 32 bytes.

    One chunk means no tree, so the root output is the chunk's own chaining
    value — which is why this reaches the published vectors without any of the
    parent-node machinery.
    """
    if len(data) > CHUNK_LEN:
        raise ValueError(f"{len(data)} bytes exceeds one {CHUNK_LEN}-byte chunk")
    blocks = blocks_of(data)

    cv = list(IV)
    for i, block in enumerate(blocks):
        flags = 0
        if i == 0:
            flags |= CHUNK_START
        if i == len(blocks) - 1:
            flags |= CHUNK_END | ROOT
        cv = compress(cv, words_of(block), 0, len(block), flags)[:8]
    return b"".join(w.to_bytes(4, "little") for w in cv)

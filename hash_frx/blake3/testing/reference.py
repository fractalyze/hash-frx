# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A plain-Python BLAKE3 compression function, as the oracle for the frx one.

Transcribed from the BLAKE3 specification over 32-bit Python integers, sharing
neither the batched lane layout nor the single-kernel authoring rules that shape
`compress.py`. An oracle that shared those would fail the same way the thing it
checks does.

`hash_tree` is a whole BLAKE3 hash, so `reference_test` anchors it against the
published vectors at every length they cover; `vectors.py` sets out why those
reach a compression function at all. `chunk_output` underneath it then serves as
an oracle for a chunk whose counter or flags no published vector reaches.

Only hash mode is here — the flags below are its three, and the rest of Table 3
belongs with the keyed and derive-key modes that use it.
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
PARENT = 1 << 2
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


def blocks_of(data: bytes, size: int = BLOCK_LEN) -> list[bytes]:
    """`data` cut into `size`-byte pieces — empty input still yields one piece.

    The empty case is the whole subtlety: an empty message is one empty chunk
    holding one empty block, not zero of either.
    """
    return [data[i : i + size] for i in range(0, len(data), size)] or [b""]


def chunk_output(
    data: bytes,
    counter: int = 0,
    final_flags: int = 0,
    final_counter: int | None = None,
) -> list[int]:
    """One chunk's last compression, all 16 output words.

    `final_flags` rides on that last compression and no other — ROOT for a chunk
    that is the root of its tree, nothing for one a tree stands on. Everything
    below it is fixed by the chunk: CHUNK_START on the first block, CHUNK_END on
    the last, `counter` on every one, and each block's own byte count.

    `final_counter` replaces `counter` on that last compression alone, which is
    what an extendable output does to a root chunk: the chain below it already
    ran at the chunk's own index and does not re-run per output block.
    """
    if len(data) > CHUNK_LEN:
        raise ValueError(f"{len(data)} bytes exceeds one {CHUNK_LEN}-byte chunk")
    blocks = blocks_of(data)

    cv = list(IV)
    for i, block in enumerate(blocks[:-1]):
        flags = CHUNK_START if i == 0 else 0
        cv = compress(cv, words_of(block), counter, len(block), flags)[:8]

    last = blocks[-1]
    flags = final_flags | CHUNK_END | (CHUNK_START if len(blocks) == 1 else 0)
    final = counter if final_counter is None else final_counter
    return compress(cv, words_of(last), final, len(last), flags)


def _subtree_output(
    chunks: list[bytes], lo: int, hi: int, final_flags: int, counter: int = 0
) -> list[int]:
    """The output words of the node covering chunks `[lo, hi)`.

    `counter` is the output-block counter and reaches the root's own
    compression only; every node below it keeps the counter its own role fixes.

    Spec section 2.1 stated as the recursion it is written as: the left subtree
    takes the largest power of two strictly below the chunk count and the right
    takes the rest, so the left is always a perfect tree and the split is not a
    halving. Written this way on purpose — the frx side reduces level by level
    instead, and an oracle that reduced the same way could only confirm that
    both had read the spec the same way.
    """
    if hi - lo == 1:
        # The output-block counter reaches a root chunk's *last* compression
        # only. Its leading blocks already ran at the chunk's own index and are
        # not re-run per output block — reading section 2.6 as replacing the
        # counter everywhere gets every multi-block single-chunk message wrong
        # and every other length right.
        final = counter if final_flags & ROOT else None
        return chunk_output(chunks[lo], lo, final_flags, final)

    left = 1
    while left * 2 < hi - lo:
        left *= 2
    cv = _subtree_output(chunks, lo, lo + left, 0)[:8]
    right = _subtree_output(chunks, lo + left, hi, 0)[:8]
    # A parent reads the key words rather than a chaining value, and its block
    # is the two child chaining values (section 2.5). Its counter is zero for
    # every node below the root; at the root it carries the output-block index,
    # which is the one place section 2.6 reaches.
    return compress(list(IV), cv + right, counter, BLOCK_LEN, final_flags | PARENT)


def hash_xof(data: bytes, out_len: int) -> bytes:
    """BLAKE3 of any input, read out to `out_len` bytes.

    Spec section 2.6: the root's compression is repeated with the output block
    counter running 0, 1, 2, … and each run's sixteen words, little-endian, are
    64 bytes of the stream. Nothing chains — every block reads the same root
    node — which is why a digest is the head of this rather than a different
    computation, and why the frx side can run the blocks in one batched call.
    """
    chunks = blocks_of(data, CHUNK_LEN)
    stream = bytearray()
    for counter in range(-(-out_len // BLOCK_LEN)):
        words = _subtree_output(chunks, 0, len(chunks), ROOT, counter)
        stream += b"".join(w.to_bytes(4, "little") for w in words)
    return bytes(stream[:out_len])


def hash_tree(data: bytes) -> bytes:
    """BLAKE3 of an input of any length: 32 bytes.

    `ROOT` rides on the topmost node and no other, which for a one-chunk input
    is the chunk itself — so that case reaches the published vectors with none
    of the parent-node machinery.
    """
    chunks = blocks_of(data, CHUNK_LEN)
    words = _subtree_output(chunks, 0, len(chunks), ROOT)[:8]
    return b"".join(w.to_bytes(4, "little") for w in words)

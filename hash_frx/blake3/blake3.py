# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""BLAKE3's chunk — the compression function chained over the chunk's blocks.

Spec section 2.4. A chunk is up to 1024 bytes split into 64-byte blocks, each
compressed into the next one's chaining value, with `CHUNK_START` on the first
block, `CHUNK_END` on the last, the chunk's index as the counter on every one,
and the trailing block's true byte count riding as `block_len` — so the zeros
that fill that block out to 64 bytes are never mistaken for message.

An input of at most one chunk is a whole hash on its own: that chunk is the root
of its tree, so its last compression sets `ROOT` and the first eight output
words, little-endian, are the digest. That is the whole of `digest` here. A
longer input needs the parent-node tree, which this module does not have, so
`digest` refuses it rather than hashing a prefix of it.

**The chunk's last compression stays pending.** One chunk becomes a chaining
value under a tree, a digest at the root, and — by repeating that same
compression with an incrementing counter (spec section 2.6) — an output stream.
The chain below is identical in all three and only the last call's flags and
counter differ, so `chunk_output` returns the compression it has *not* run and
the caller finishes it. Finishing it a second way then costs one compression
rather than the whole chunk again.

Message length is static, so the block count, the flag schedule, and every block
length are compile-time constants: the loop over blocks is an unrolled Python
`for` over them and nothing here reads a message byte. That is what lets
`digest` take a tracer, the property `sha256.digest` holds for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
from frx import Array
from frx.typing import ArrayLike

from hash_frx.blake3.compress import CHUNK_END, CHUNK_START, IV, ROOT, compress

U32 = fnp.uint32

BLOCK_LEN = 64
CHUNK_LEN = 1024


@dataclass(frozen=True)
class Output:
    """A node's final compression, assembled but not run — the five operands.

    Batched over a leading `[B, ...]` axis like `compress` itself. The flags are
    the node's own (`CHUNK_END` for a chunk); what the finishing call adds to
    them is the node's role in the tree, which the node does not know.
    """

    chaining_value: Array  # uint32 [B, 8]
    block: Array  # uint32 [B, 16]
    counter: Array  # uint32 [B, 2]
    block_len: Array  # uint32 [B]
    flags: Array  # uint32 [B]


def chunk_output(words: Array, chunk_len: int, counter: Array) -> Output:
    """Chain one chunk's blocks up to, but not including, its last compression.

    words     : uint32 `[B, nblocks, 16]` — the chunk's blocks as little-endian
                message words, the trailing one zero-padded
    chunk_len : the chunk's byte count; static, and `nblocks` follows from it
    counter   : uint32 `[B, 2]` — the chunk index as (low, high)

    A chunk is a chain rather than a batch: every block feeds the next, so the
    parallelism lives across chunks and rows, never here.
    """
    if words.ndim != 3 or words.shape[2] != 16:
        raise ValueError(f"words must be [B, nblocks, 16], got {words.shape}")
    if not 0 <= chunk_len <= CHUNK_LEN:
        raise ValueError(f"chunk_len must be 0..{CHUNK_LEN}, got {chunk_len}")
    batch, nblocks, _ = words.shape
    expected = max(1, -(-chunk_len // BLOCK_LEN))
    if nblocks != expected:
        raise ValueError(
            f"{chunk_len} bytes is {expected} block(s), got {nblocks} — the "
            "trailing block is padded, so its length cannot be read back"
        )

    # Every chunk opens from the key words rather than from a neighbour's
    # chaining value (spec section 2.4), and the hash mode's key is the IV
    # (section 2.3). That independence is what makes chunks parallel.
    cv = fnp.broadcast_to(fnp.asarray(IV, dtype=U32), (batch, 8))
    full_block = fnp.full((batch,), BLOCK_LEN, dtype=U32)
    for i in range(nblocks - 1):  # static and at most 15
        flags = CHUNK_START if i == 0 else 0
        cv = compress(
            cv,
            words[:, i],
            counter,
            full_block,
            fnp.full((batch,), flags, dtype=U32),
        )[:, :8]

    last = nblocks - 1
    return Output(
        chaining_value=cv,
        block=words[:, last],
        counter=counter,
        # The trailing block's own byte count; zero only for an empty chunk,
        # which is the one case in which a block is empty at all.
        block_len=fnp.full((batch,), chunk_len - BLOCK_LEN * last, dtype=U32),
        flags=fnp.full(
            (batch,),
            CHUNK_END | (CHUNK_START if last == 0 else 0),
            dtype=U32,
        ),
    )


def root_words(output: Output) -> Array:
    """Finish a root node's compression: uint32 `[B, 16]`.

    `ROOT` rides on this one call and no other (spec section 2.4). The full
    sixteen words are the node's extendable output; the first eight are the
    256-bit chaining value that the 32-byte digest encodes.
    """
    return compress(
        output.chaining_value,
        output.block,
        output.counter,
        output.block_len,
        output.flags | U32(ROOT),
    )


def _block_words(msg: Array) -> Array:
    """uint8 `[B, L]` -> uint32 `[B, nblocks, 16]` little-endian message words.

    Spec section 2.4: the trailing block is zero-padded to 64 bytes and its true
    byte count reaches the compression as `block_len` instead. So the padding is
    a host constant built from the static length rather than something written
    into the message, which keeps `msg` an operand and lets it be a tracer —
    `sha256._padding_tail` holds the same property for the same reason.
    """
    batch, length = msg.shape
    nblocks = max(1, -(-length // BLOCK_LEN))
    pad = nblocks * BLOCK_LEN - length
    if pad:
        msg = fnp.concatenate([msg, fnp.zeros((batch, pad), dtype=fnp.uint8)], axis=-1)
    w = msg.reshape(batch, nblocks, 16, 4).astype(U32)
    return (
        w[..., 0]
        | (w[..., 1] << U32(8))
        | (w[..., 2] << U32(16))
        | (w[..., 3] << U32(24))
    )


def _le_bytes(words: Array) -> Array:
    """uint32 `[B, n]` -> uint8 `[B, 4n]`, little-endian — BLAKE3's output order.

    Little-endian throughout, unlike `sha256.serialize_digest`; the two hashes
    disagree on byte order, so neither packer is shared.
    """
    batch = words.shape[0]
    out = fnp.stack(
        [
            words & U32(0xFF),
            (words >> U32(8)) & U32(0xFF),
            (words >> U32(16)) & U32(0xFF),
            (words >> U32(24)) & U32(0xFF),
        ],
        axis=-1,
    ).astype(fnp.uint8)
    return out.reshape(batch, -1)


def digest(msg: ArrayLike) -> Array:
    """BLAKE3 of a batch of equal-length messages: uint8 `[B, L]` -> `[B, 32]`.

    Byte-identical to the standard per message, for an `L` of at most one
    1024-byte chunk. `msg` may be a tracer, so a consumer can hash inside its own
    `@jit` or `vmap`.

    A one-chunk message is the root of its own tree, which is what makes it
    reachable without the tree: the chunk's counter is index 0 and its last
    compression is the root compression.
    """
    message = fnp.asarray(msg, dtype=fnp.uint8)
    if message.ndim != 2:
        raise ValueError(f"msg must be 2-D uint8 [B, L], got ndim={message.ndim}")
    batch, length = message.shape
    if length > CHUNK_LEN:
        raise ValueError(
            f"{length} bytes spans more than one {CHUNK_LEN}-byte chunk, which "
            "needs the parent-node tree"
        )

    output = chunk_output(
        _block_words(message), length, fnp.zeros((batch, 2), dtype=U32)
    )
    return _le_bytes(root_words(output)[:, :8])

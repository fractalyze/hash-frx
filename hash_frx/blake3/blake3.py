# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""BLAKE3's chunk and tree — a whole hash, at any length.

Spec sections 2.1, 2.4 and 2.5. A chunk is up to 1024 bytes split into 64-byte
blocks, each compressed into the next one's chaining value, with `CHUNK_START`
on the first block, `CHUNK_END` on the last, the chunk's index as the counter on
every one, and the trailing block's true byte count riding as `block_len` — so
the zeros that fill that block out to 64 bytes are never mistaken for message.
Above the chunks, a binary tree of parent compressions folds their chaining
values down to one, and the topmost node carries `ROOT`.

**A chunk is a chain; everything else is a batch.** Blocks inside a chunk each
feed the next, so there is nothing to parallelize there — but chunks do not
depend on each other at all (each opens from the key words rather than from its
neighbour's output), and nor do the nodes within a tree level. So the chunks are
one batched call and each tree level is one more, which is the property BLAKE3
was chosen for.

**The tree is built by level, and that is the spec's tree exactly.** Adjacent
nodes pair into parents from the bottom and an odd trailing node rides up
unpaired, so a message of `n` chunks costs `ceil(log2(n))` batched compressions
rather than a walk over `2n - 1` nodes. `tree_output` argues the equivalence.

**A node's final compression stays pending.** One node becomes a chaining value
under a tree, a digest at the root, and — by repeating that same compression
with an incrementing counter (spec section 2.6) — an output stream. Everything
below it is identical in all three and only that last call's flags differ, so
`chunk_output` and `parent_output` return the compression they have *not* run
and the caller finishes it with `chaining_value` or `root_words`. Finishing a
second way then costs one compression rather than the whole subtree again.

Message length is static, so the chunk count, the tree shape, the block count,
the flag schedule and every block length are compile-time constants: the loops
here are unrolled Python `for`/`while` over them and nothing reads a message
byte. That is what lets `digest` take a tracer, the property `sha256.digest`
holds for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.blake3.compress import (
    CHUNK_END,
    CHUNK_START,
    IV,
    PARENT,
    ROOT,
    compress,
)
from hash_frx.word import pack_le, split, unpack_le

U32 = fnp.uint32

BLOCK_LEN = 64
CHUNK_LEN = 1024


def _units(length: int, size: int) -> int:
    """`length` bytes as a count of `size`-byte units — empty still occupies one.

    Blocks within a chunk and chunks within a message are the same ceiling with
    the same floor, and the floor is the whole subtlety: an empty message is one
    empty chunk holding one empty block, not zero of either.
    """
    return max(1, -(-length // size))


def _key_words(batch: int) -> Array:
    """The key words a node opens from: uint32 `[batch, 8]`.

    Hash mode's key is the IV (spec section 2.3). Every chunk and every parent
    opens from it rather than from a neighbour's chaining value, which is what
    makes chunks independent and the tree parallel.
    """
    return fnp.broadcast_to(fnp.asarray(IV, dtype=U32), (batch, 8))


@dataclass(frozen=True)
class Output:
    """A node's final compression, assembled but not run — the five operands.

    Batched over a leading `[B, ...]` axis like `compress` itself. The flags are
    the node's own (`CHUNK_END` for a chunk); what the finishing call adds to
    them is the node's role in the tree, which the node does not know.

    `input_chaining_value` is what this compression reads, which for a parent is
    the key words rather than a child's output — it is not the node's own
    chaining value, which is what `chaining_value` computes *from* this. The
    reference implementation draws the same distinction with the same two names.
    """

    input_chaining_value: Array  # uint32 [B, 8]
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
    if words.dtype != U32:
        raise TypeError(f"words must be uint32, got {words.dtype}")
    if not 0 <= chunk_len <= CHUNK_LEN:
        raise ValueError(f"chunk_len must be 0..{CHUNK_LEN}, got {chunk_len}")
    batch, nblocks, _ = words.shape
    expected = _units(chunk_len, BLOCK_LEN)
    if nblocks != expected:
        raise ValueError(
            f"{chunk_len} bytes is {expected} block(s), got {nblocks} — the "
            "trailing block is padded, so its length cannot be read back"
        )

    cv = _key_words(batch)
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
        input_chaining_value=cv,
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


def chaining_value(output: Output) -> Array:
    """Finish a node that has a tree above it: uint32 `[B, 8]`.

    The same compression `root_words` runs, without `ROOT` — which is the only
    difference between a node's chaining value and the root's output, and the
    reason a node is assembled once and finished by whoever knows its role.
    """
    return compress(
        output.input_chaining_value,
        output.block,
        output.counter,
        output.block_len,
        output.flags,
    )[:, :8]


def parent_output(left: Array, right: Array) -> Output:
    """A parent node's compression, assembled but not run.

    left, right : uint32 `[B, 8]` — the two child chaining values

    Spec section 2.5. A parent reads the key words rather than a child's
    chaining value, its counter is always zero however deep the tree is, and its
    block is the two children end to end — so its whole message is 64 bytes and
    `block_len` is a full block.
    """
    for name, child in (("left", left), ("right", right)):
        # The one entry here a caller hands raw arrays to, so it is the one that
        # has to name the caller's mistake. Without this a `[B, 4]` child reaches
        # `compress` and is reported against `block`, an operand the caller never
        # passed.
        if child.ndim != 2 or child.shape[1] != 8:
            raise ValueError(f"{name} must be [B, 8], got {child.shape}")
        if child.dtype != U32:
            raise TypeError(f"{name} must be uint32, got {child.dtype}")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"left and right must agree on the batch, got {left.shape[0]} "
            f"and {right.shape[0]}"
        )

    batch = left.shape[0]
    return Output(
        input_chaining_value=_key_words(batch),
        block=fnp.concatenate([left, right], axis=1),
        counter=fnp.zeros((batch, 2), dtype=U32),
        block_len=fnp.full((batch,), BLOCK_LEN, dtype=U32),
        flags=fnp.full((batch,), PARENT, dtype=U32),
    )


def root_words(output: Output) -> Array:
    """Finish a root node's compression: uint32 `[B, 16]`.

    `ROOT` rides on this one call and no other (spec section 2.4). The full
    sixteen words are the node's extendable output; the first eight are the
    256-bit chaining value that the 32-byte digest encodes.
    """
    return compress(
        output.input_chaining_value,
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
    nblocks = _units(length, BLOCK_LEN)
    pad = nblocks * BLOCK_LEN - length
    if pad:
        msg = fnp.concatenate([msg, fnp.zeros((batch, pad), dtype=fnp.uint8)], axis=-1)
    return pack_le(msg.reshape(batch, nblocks, BLOCK_LEN))


def _counters(batch: int, first: int, count: int) -> Array:
    """Chunk indices `first .. first + count - 1`, tiled over `batch` messages.

    uint32 `[batch * count, 2]` as (low, high). The split is done on the host,
    where the index is a Python int, so no 64-bit value is ever materialised —
    the reason `compress` takes the counter in halves at all.
    """
    halves = np.array([split(i) for i in range(first, first + count)], dtype=np.uint32)
    return fnp.asarray(np.tile(halves, (batch, 1)))


def _chunk_chaining_values(message: Array, nchunks: int) -> Array:
    """Every chunk's chaining value: uint32 `[B, nchunks, 8]`.

    `nchunks` is at least two — `tree_output` handles a one-chunk message itself,
    because that chunk's last compression has to stay pending.

    Chunks of one length are one batched chain over `B * k` rows, which is the
    parallelism BLAKE3 exists for. Only the last chunk can be short, and a
    chunk's byte count fixes its *block count*, which is a shape — so a short
    last chunk is a second call at a shorter shape. A message ending on a chunk
    boundary has no short chunk and is one call.

    That split is not the only one possible, and it is not free: the short
    chunk's blocks repeat call sites the batched chain already has, so an
    unaligned message emits about twice the program of an aligned one — 33
    compressions at 2047 bytes against 17 at 2048. `block_len` and `flags` are
    already per-row operands, so the short chunk's leading blocks could ride in
    the batched chain and only its tail be handled apart. That is a larger change
    than the shape story above suggests, and it is not made here.
    """
    batch = message.shape[0]
    tail_len = message.shape[1] - CHUNK_LEN * (nchunks - 1)
    aligned = tail_len == CHUNK_LEN
    full = nchunks if aligned else nchunks - 1

    # The batched chunks arrive as `[B * k, 8]` with the chunk index varying
    # fastest, which is what `_counters` tiles to match, so the reshape lands
    # each message's chunks in order on its own row.
    head = message[:, : CHUNK_LEN * full].reshape(batch * full, CHUNK_LEN)
    values = chaining_value(
        chunk_output(_block_words(head), CHUNK_LEN, _counters(batch, 0, full))
    ).reshape(batch, full, 8)
    if aligned:
        return values

    tail = chaining_value(
        chunk_output(
            _block_words(message[:, CHUNK_LEN * full :]),
            tail_len,
            _counters(batch, full, 1),
        )
    ).reshape(batch, 1, 8)
    return fnp.concatenate([values, tail], axis=1)


def tree_output(msg: ArrayLike) -> Output:
    """The root node of a message's tree, assembled but not run.

    Spec section 2.1. The chunks are hashed in one or two batched calls and then
    reduced a level at a time: adjacent nodes pair into parents, and an odd
    trailing node rides up to the next level unpaired.

    **That level reduction is the spec's tree, not an approximation of it.** The
    spec splits a node by giving its left subtree the largest power of two
    strictly below the chunk count — a split that is not a halving, and that
    makes every left subtree perfect. Pairing from the bottom and carrying the
    odd node up produces exactly that shape for every chunk count, which is what
    lets the tree be `ceil(log2(n))` batched compressions rather than a walk.

    The root's own compression stays pending, for the reason `chunk_output`
    leaves a chunk's last one pending: the root is where `ROOT` and the
    extendable output live, and neither belongs to the tree that produced it.
    """
    message = fnp.asarray(msg, dtype=fnp.uint8)
    if message.ndim != 2:
        raise ValueError(f"msg must be 2-D uint8 [B, L], got ndim={message.ndim}")
    batch, length = message.shape

    nchunks = _units(length, CHUNK_LEN)
    if nchunks == 1:
        # A one-chunk message is the root of its own tree, with no parent above
        # it — so it is a chunk output, not a parent output.
        return chunk_output(
            _block_words(message), length, fnp.zeros((batch, 2), dtype=U32)
        )

    nodes = _chunk_chaining_values(message, nchunks)
    while nodes.shape[1] > 2:
        pairs, odd = divmod(nodes.shape[1], 2)
        paired = nodes[:, : 2 * pairs].reshape(batch * pairs, 2, 8)
        parents = chaining_value(parent_output(paired[:, 0], paired[:, 1])).reshape(
            batch, pairs, 8
        )
        # An odd node at the end has no sibling on this level and pairs with
        # whatever the levels above hand it, so it rides up untouched.
        nodes = fnp.concatenate([parents, nodes[:, -1:]], axis=1) if odd else parents

    return parent_output(nodes[:, 0], nodes[:, 1])


def digest(msg: ArrayLike) -> Array:
    """BLAKE3 of a batch of equal-length messages: uint8 `[B, L]` -> `[B, 32]`.

    Byte-identical to the standard per message, at any length. `msg` may be a
    tracer, so a consumer can hash inside its own `@jit` or `vmap`.
    """
    return unpack_le(root_words(tree_output(msg))[:, :8])

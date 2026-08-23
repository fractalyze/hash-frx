# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""BLAKE3's construction, as a function of which of its three modes is running.

Spec sections 2.1, 2.3, 2.4 and 2.5. A chunk is up to 1024 bytes split into
64-byte blocks, each compressed into the next one's chaining value, with
`CHUNK_START` on the first block, `CHUNK_END` on the last, the chunk's index as
the counter on every one, and the trailing block's true byte count riding as
`block_len` — so the zeros that fill that block out to 64 bytes are never
mistaken for message. Above the chunks, a binary tree of parent compressions
folds their chaining values down to one, and the topmost node carries `ROOT`.

**A mode is a key and a flag, and nothing else about BLAKE3 moves.** Hash mode
opens every node from the IV; keyed hashing puts a caller's 32 bytes there and
sets `KEYED_HASH` on every compression; key derivation is that construction run
twice, the first pass hashing the context string under `DERIVE_KEY_CONTEXT` to
produce the key the second opens from (spec section 2.3). So the chunk, the tree
and the root below are ONE implementation threaded with a `Mode` rather than
three that would drift, and only `Mode` names which of the three is running.
That is what this module is: the whole unmarked hash, parameterized on the mode,
with the named and marked surface over it in [`rows.py`](rows.py).

**A chunk is a chain; everything else is a batch.** Blocks inside a chunk each
feed the next, so there is nothing to parallelize there — but chunks do not
depend on each other at all (each opens from the key words rather than from its
neighbour's output), and nor do the nodes within a tree level. So the chunks are
one batched call and each tree level is one more, which is the property BLAKE3
was chosen for.

**The schedule is [`extension/tree.py`](../extension/tree.py) and the spelling is
here.** That module answers how many blocks a chunk holds, which nodes pair with
which on the way up, and how many compressions an output request spans; it emits
no operation, because a shared helper that takes an array fixes that array's
emission order for every caller. What ops those answers are carried out with —
the flag schedule, the counter halves, the word packing — is this module's, and
stays here for exactly that reason.

**A node's final compression stays pending.** One node becomes a chaining value
under a tree, a digest at the root, and — by repeating that same compression
with an incrementing counter (spec section 2.6) — an output stream. Everything
below it is identical in all three and only that last call's flags and counter
differ, so `chunk_output` and `parent_output` return the compression they have
*not* run and the caller finishes it with `chaining_value` or `root_words`.
Finishing a second way then costs one compression rather than the whole subtree
again, and reading further costs one more per 64 bytes.

Message length is static, so the chunk count, the tree shape, the block count,
the flag schedule and every block length are compile-time constants: the loops
here are unrolled Python `for`/`while` over them and nothing reads a message
byte. That is what lets `digest` take a tracer, the property `sha256.digest`
holds for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.blake3.compress import (
    CHUNK_END,
    CHUNK_START,
    DERIVE_KEY_CONTEXT,
    DERIVE_KEY_MATERIAL,
    IV_WORDS,
    KEYED_HASH,
    PARENT,
    ROOT,
    compress,
    compress_cv,
)
from hash_frx.byte_hash import device_message, padded_batch
from hash_frx.extension import tree
from hash_frx.word import pack_le, unpack_le

U32 = fnp.uint32

BLOCK_LEN = 64
CHUNK_LEN = 1024
DIGEST_LEN = 32
# The key both keyed modes open from is 256 bits (spec section 2.3), which is
# also what a derive-key context pass reads out — the two are the same width
# because the second pass's key *is* the first pass's digest.
KEY_LEN = 32

# The mode flags spec section 2.3 defines, and the whole of what a `Mode` may
# carry: plain hashing sets none, and the other three name a mode each. They are
# never combined — a compression runs in one mode — so this is a set of legal
# values rather than a mask.
_MODE_FLAGS = frozenset({0, KEYED_HASH, DERIVE_KEY_CONTEXT, DERIVE_KEY_MATERIAL})


@dataclass(frozen=True)
class Mode:
    """Which of BLAKE3's three modes is running — the whole of the difference.

    Spec section 2.3. `key_words` is what every node opens from in place of a
    neighbour's chaining value, and `flags` is what rides on every compression
    the mode makes, under the flags the node's own position already carries.
    Nothing else about the hash — the chunk size, the tree shape, the round
    function, the root — reads either field.

    `iv` is the third field and the one the spec does *not* vary: every mode
    opens every compression's third state row from the same table. It rides here
    because the record is what already travels to each compression, and because
    the value has to be threaded rather than built — a marked region passes its
    own operand (see `compress`), and a hop that forgot to forward it would grow
    the marker's operand list rather than fail. Threaded on the record, there is
    no hop to forget.

    **One key serves the whole batch.** It is a parameter of the hash, the way
    an output length is, rather than something a message carries: `[8]` words
    broadcast across the rows rather than `[B, 8]` tiled along them. A caller
    wanting a key *per* message maps over both — `frx.vmap` of a keyed call over
    `(key, msg)` — rather than hashing one message at a time.

    Comparing or hashing a `Mode` raises, and is meant to: the dataclass-derived
    `__eq__` is element-wise over an `Array` and the derived `__hash__` hashes
    one. A mode is an operand record rather than a parameter record, so it is
    not a jit static argument the way a `ByteHash` is — and the raise is what
    says so at the first call, rather than a per-instance cache miss that
    re-traces and bakes the key in on every one. `Output` is the same kind of
    record and answers the same way.
    """

    key_words: Array  # uint32 [8]
    flags: int
    # `default_factory` rather than a plain default: a dataclass refuses an
    # unhashable default, and an `Array` is one.
    iv: Array = field(default_factory=lambda: IV_WORDS)  # uint32 [8]

    def __post_init__(self) -> None:
        # Both fields fail silently unchecked, which is why this is here rather
        # than left to `compress`.
        #
        # A `[1, 8]` or `[B, 8]` key does not error at all: `_key_words`
        # broadcasts it and the result is well-formed: at `[1, 8]` byte-identical
        # to the `[8]` spelling, and at `[B, 8]` a per-row keying this does not
        # offer. Only a width that cannot broadcast errors, and it errors inside
        # `broadcast_to` rather than against anything a caller passed. A wrong
        # dtype is the one case `compress` catches, and it names `chaining_value`
        # there — an operand the caller never passed.
        if self.key_words.shape != (8,):
            raise ValueError(f"key_words must be [8], got {self.key_words.shape}")
        if self.key_words.dtype != U32:
            raise TypeError(f"key_words must be uint32, got {self.key_words.dtype}")
        if self.iv.shape != (8,):
            raise ValueError(f"iv must be [8], got {self.iv.shape}")
        if self.iv.dtype != U32:
            raise TypeError(f"iv must be uint32, got {self.iv.dtype}")
        # Flags never error anywhere: a wrong mode flag rides through every
        # compression and produces well-formed bytes under the wrong domain
        # separation. Spec section 2.3 names exactly these four, and a caller
        # assembling a `Mode` by hand is exactly the caller who picks one.
        if self.flags not in _MODE_FLAGS:
            raise ValueError(
                f"flags must be one of {sorted(_MODE_FLAGS)} (spec section 2.3), "
                f"got {self.flags}"
            )


def _key_words(mode: Mode, batch: int) -> Array:
    """The key words a node opens from: uint32 `[batch, 8]`.

    Every chunk and every parent opens from the mode's key rather than from a
    neighbour's chaining value, which is what makes chunks independent and the
    tree parallel. Hash mode's key is the IV (spec section 2.3); the two keyed
    modes put 32 bytes of the caller's there and change nothing else here.
    """
    return fnp.broadcast_to(mode.key_words, (batch, 8))


@dataclass(frozen=True)
class Output:
    """A node's final compression, assembled but not run — its six operands.

    Batched over a leading `[B, ...]` axis like `compress` itself. The flags are
    the node's own (`CHUNK_END` for a chunk); what the finishing call adds to
    them is the node's role in the tree, which the node does not know.

    `input_chaining_value` is what this compression reads, which for a parent is
    the key words rather than a child's output — it is not the node's own
    chaining value, which is what `chaining_value` computes *from* this. The
    reference implementation draws the same distinction with the same two names.

    `iv` is the node's mode's, carried here for the reason `Mode` carries it:
    whoever finishes the node has an `Output` and not the mode it came from, so
    without it the value would have to be passed alongside at every such call.
    """

    input_chaining_value: Array  # uint32 [B, 8]
    block: Array  # uint32 [B, 16]
    counter: Array  # uint32 [B, 2]
    block_len: Array  # uint32 [B]
    flags: Array  # uint32 [B]
    iv: Array  # uint32 [8]


def _chain(cv: Array, words: Array, counter: Array, first: int, mode: Mode) -> Array:
    """Compress `words` into `cv`, one block at a time: uint32 `[B, 8]`.

    words : uint32 `[B, k, 16]` — blocks `first .. first + k - 1` of a chunk,
            none of them the chunk's last, so each is a full 64 bytes
    first : where `words[:, 0]` sits in its chunk, which is the only thing that
            decides whether `CHUNK_START` rides on it

    The blocks before a chunk's last are the only part two chunks of *different*
    lengths have in common — all full 64 bytes, same flags, differing only in
    the counter — which is what lets a short chunk share a call with full ones.
    """
    batch = cv.shape[0]
    full_block = fnp.full((batch,), BLOCK_LEN, dtype=U32)

    def compress_block(running: Array, i: int) -> Array:
        flags = mode.flags | (CHUNK_START if first + i == 0 else 0)
        return compress_cv(
            running,
            words[:, i],
            counter,
            full_block,
            fnp.full((batch,), flags, dtype=U32),
            mode.iv,
        )

    # `full_block` is built before the loop rather than inside `compress_block`
    # deliberately: one block length serves every block of the chain, and the
    # schedule emits nothing of its own, so where the caller puts an op is where
    # it lands in the lowered module.
    return tree.chain(cv, count=words.shape[1], compress_block=compress_block)


def _chunk_from(
    cv: Array, words: Array, counter: Array, chunk_len: int, first: int, mode: Mode
) -> Output:
    """Run a chunk's blocks from `first` on, leaving its last compression unrun.

    words     : uint32 `[B, nblocks, 16]` — the chunk's blocks entire, of which
                only `first ..` run here
    chunk_len : the chunk's byte count, which the trailing block's own length is
                read off
    first     : where this picks the chunk up — 0 for a chunk run whole, and past
                a shared chain for one whose leading blocks already ran alongside
                chunks of a different length

    `CHUNK_START` rides on block 0 and `CHUNK_END` on the last, which is the
    whole of a chunk's flag schedule; a single-block chunk carries both.
    """
    last = words.shape[1] - 1
    return Output(
        input_chaining_value=_chain(cv, words[:, first:last], counter, first, mode),
        block=words[:, last],
        counter=counter,
        # The trailing block's own byte count; zero only for an empty chunk,
        # which is the one case in which a block is empty at all.
        block_len=fnp.full((cv.shape[0],), chunk_len - BLOCK_LEN * last, dtype=U32),
        flags=fnp.full(
            (cv.shape[0],),
            mode.flags | CHUNK_END | (CHUNK_START if last == 0 else 0),
            dtype=U32,
        ),
        iv=mode.iv,
    )


def _stacked(*outputs: Output) -> Output:
    """Several nodes' pending compressions as the rows of one call.

    They are the same operation on different operands, so finishing them apart
    would emit the compression body once per group for no reason — and the rows
    of a batch is exactly what `compress` already takes.
    """
    return Output(
        input_chaining_value=fnp.concatenate(
            [o.input_chaining_value for o in outputs], axis=0
        ),
        block=fnp.concatenate([o.block for o in outputs], axis=0),
        counter=fnp.concatenate([o.counter for o in outputs], axis=0),
        block_len=fnp.concatenate([o.block_len for o in outputs], axis=0),
        flags=fnp.concatenate([o.flags for o in outputs], axis=0),
        # Not concatenated: it is unbatched, and the rows of one call are one
        # mode's nodes, so every input carries the same table.
        iv=outputs[0].iv,
    )


def _output_blocks(output: Output, count: int) -> Output:
    """One node's pending compression as `count` rows, one per output block.

    The other way a batch arises here: `_stacked` gives several nodes a row
    each, this gives one node `count` of them, differing only in the counter
    that spec section 2.6 increments. The node's own counter is replaced rather
    than added to — a root chunk is chunk 0 and a parent's counter is zero, so
    what the field carries at the root is the output-block index and nothing
    else.

    Row `b * count + i` is message `b`'s block `i`: `repeat` holds each row for
    `count` places and `_counters` runs the indices inside them, so the two
    agree on the layout the trailing reshape splits back per message.
    """
    return Output(
        input_chaining_value=fnp.repeat(output.input_chaining_value, count, axis=0),
        block=fnp.repeat(output.block, count, axis=0),
        counter=_counters(output.block.shape[0], 0, count),
        block_len=fnp.repeat(output.block_len, count, axis=0),
        flags=fnp.repeat(output.flags, count, axis=0),
        iv=output.iv,
    )


def chunk_output(words: Array, chunk_len: int, counter: Array, mode: Mode) -> Output:
    """Chain one chunk's blocks up to, but not including, its last compression.

    words     : uint32 `[B, nblocks, 16]` — the chunk's blocks as little-endian
                message words, the trailing one zero-padded
    chunk_len : the chunk's byte count; static, and `nblocks` follows from it
    counter   : uint32 `[B, 2]` — the chunk index as (low, high)
    mode      : which of the three modes this chunk belongs to

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
    expected = tree.units(chunk_len, BLOCK_LEN)
    if nblocks != expected:
        raise ValueError(
            f"{chunk_len} bytes is {expected} block(s), got {nblocks} — the "
            "trailing block is padded, so its length cannot be read back"
        )

    return _chunk_from(_key_words(mode, batch), words, counter, chunk_len, 0, mode)


def chaining_value(output: Output) -> Array:
    """Finish a node that has a tree above it: uint32 `[B, 8]`.

    The same compression `root_words` runs, without `ROOT` — which is the only
    difference between a node's chaining value and the root's output, and the
    reason a node is assembled once and finished by whoever knows its role. The
    two now differ by that flag and the read alone, which `compress_cv` is what
    makes readable rather than asserted here.
    """
    return compress_cv(
        output.input_chaining_value,
        output.block,
        output.counter,
        output.block_len,
        output.flags,
        output.iv,
    )


def parent_output(left: Array, right: Array, mode: Mode) -> Output:
    """A parent node's compression, assembled but not run.

    left, right : uint32 `[B, 8]` — the two child chaining values
    mode        : which of the three modes this tree belongs to

    Spec section 2.5. A parent reads the key words rather than a child's
    chaining value, its counter is always zero however deep the tree is, and its
    block is the two children end to end — so its whole message is 64 bytes and
    `block_len` is a full block.
    """
    for name, child in (("left", left), ("right", right)):
        # A caller hands these in raw, so the mistake is named here. Without
        # this a `[B, 4]` child reaches `compress` and is reported against
        # `block`, an operand the caller never passed.
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
        input_chaining_value=_key_words(mode, batch),
        block=fnp.concatenate([left, right], axis=1),
        counter=fnp.zeros((batch, 2), dtype=U32),
        block_len=fnp.full((batch,), BLOCK_LEN, dtype=U32),
        flags=fnp.full((batch,), mode.flags | PARENT, dtype=U32),
        iv=mode.iv,
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
        output.iv,
    )


def root_bytes(output: Output, out_len: int) -> Array:
    """A root node's extendable output: uint8 `[B, out_len]`.

    Spec section 2.6. The root's compression is repeated with an output-block
    counter running 0, 1, 2, … and each run's sixteen words, little-endian, are
    64 bytes of the stream.

    **The blocks do not chain.** Every one reads the same root node and differs
    only in that counter, so they are `_output_blocks` rows of the *one* call
    `root_words` already is, rather than a sequential squeeze — the property a
    sponge XOF cannot offer. `out_len` is static, so how many rows that is is
    known at trace time.

    Only this compression is repeated: whatever chain produced the node already
    ran at the counter its role fixed, and is not re-run per output block. That
    falls out of the node arriving unrun.
    """
    if out_len < 1:
        raise ValueError(f"out_len must be at least 1, got {out_len}")
    batch = output.block.shape[0]
    blocks = tree.units(out_len, BLOCK_LEN)
    stream = root_words(_output_blocks(output, blocks))
    return unpack_le(stream).reshape(batch, blocks * BLOCK_LEN)[:, :out_len]


def _block_words(msg: Array) -> Array:
    """uint8 `[B, L]` -> uint32 `[B, nblocks, 16]` little-endian message words.

    Spec section 2.4: the trailing block is zero-padded to 64 bytes and its true
    byte count reaches the compression as `block_len` instead. So the padding is
    a host constant built from the static length rather than something written
    into the message, which keeps `msg` an operand and lets it be a tracer —
    `sha256._padding_tail` holds the same property for the same reason.
    """
    batch, length = msg.shape
    nblocks = tree.units(length, BLOCK_LEN)
    pad = nblocks * BLOCK_LEN - length
    if pad:
        msg = padded_batch(msg, fnp.zeros(pad, dtype=fnp.uint8))
    return pack_le(msg.reshape(batch, nblocks, BLOCK_LEN))


def _counters(batch: int, first: int, count: int) -> Array:
    """Counter values `first .. first + count - 1`, repeated per message.

    uint32 `[batch * count, 2]` as (low, high) — the halves `compress` takes,
    for the reason `keccak/lane.py` sets out: a 64-bit lane is not safely
    available here.

    What the index counts is the caller's: a chunk's position in the message
    for `_chunk_chaining_values`, an output block's position in the stream for
    `_output_blocks`. The counter field carries whichever the node's role fixes.

    **The repetition is a broadcast, not a host tile.** Every message numbers
    its blocks alike, so expanding the sequence on the host bakes `batch`
    identical copies of it into the program — a megabyte of constant for 128
    bytes of value at four thousand messages and sixteen output blocks. Held
    shaped, the program carries the sequence once whatever the batch is.

    **The sequence is counted on the device, not written on the host.** An
    `arange` is an index computation rather than a value the program carries, so
    there is no constant here for a `lax.composite` to lift into the marker's
    operand list — the rule and its two remedies are in
    `docs/reference/conventions.md`. Written as a host table instead, this alone
    would make the ABI's leading operands a function of the tree shape: three
    such tables at three chunks, one at one.

    The high half is zero, and exactly rather than approximately: `first + count`
    is a static shape's worth of chunks or output blocks, so reaching 2^32 of
    either would need a message no program can shape. The bound is asserted
    rather than assumed, because it is the one thing standing between this and a
    silently truncated counter.
    """
    if (first + count - 1) >> 32:
        raise ValueError(
            f"counter {first + count - 1} exceeds the 32-bit low half; the high "
            "half is only zero because a static shape cannot reach it"
        )
    pair = fnp.stack(
        [
            fnp.arange(first, first + count, dtype=U32),
            fnp.zeros((count,), dtype=U32),
        ],
        axis=1,
    )  # [count, 2]
    return fnp.broadcast_to(pair, (batch, count, 2)).reshape(batch * count, 2)


def _chunk_chaining_values(message: Array, nchunks: int, mode: Mode) -> Array:
    """Every chunk's chaining value: uint32 `[B, nchunks, 8]`.

    `nchunks` is at least two — `tree_output` handles a one-chunk message itself,
    because that chunk's last compression has to stay pending.

    Only the last chunk can be short. A chunk's byte count fixes its *block
    count*, which is a shape, so a short chunk cannot be a row of a call that
    runs sixteen blocks — but that binds only from its own last block onward.
    Every block before that is a full 64 bytes in both, carries the same flags,
    and differs only in the counter, which is already a per-row operand. So the
    short chunk rides the batched chain as far as it goes, leaves it at its own
    last block, and rejoins the full chunks for the finishing compression, which
    is the same operation on both and so one call over all the rows.

    The result is **sixteen compressions whatever the message length** — the
    same as if every chunk were full. Handled apart instead, a short chunk of
    `m` blocks costs `16 + m`, so a message one byte short of a chunk boundary
    would emit nearly twice the program of one exactly on it.
    """
    batch = message.shape[0]
    tail_len = message.shape[1] - CHUNK_LEN * (nchunks - 1)
    full = nchunks - 1

    head_words = _block_words(
        message[:, : CHUNK_LEN * full].reshape(batch * full, CHUNK_LEN)
    )
    tail_words = _block_words(message[:, CHUNK_LEN * full :])
    head_counter = _counters(batch, 0, full)
    tail_counter = _counters(batch, full, 1)

    # The blocks both kinds of chunk run, in one call over every row.
    leaves = tail_words.shape[1] - 1
    shared = _chain(
        _key_words(mode, batch * nchunks),
        fnp.concatenate([head_words[:, :leaves], tail_words[:, :leaves]], axis=0),
        fnp.concatenate([head_counter, tail_counter], axis=0),
        0,
        mode,
    )

    values = chaining_value(
        _stacked(
            _chunk_from(
                shared[: batch * full],
                head_words,
                head_counter,
                CHUNK_LEN,
                leaves,
                mode,
            ),
            _chunk_from(
                shared[batch * full :],
                tail_words,
                tail_counter,
                tail_len,
                leaves,
                mode,
            ),
        )
    )
    # The full chunks arrive as `[B * k, 8]` with the chunk index varying
    # fastest, which is what `_counters` repeats to match, so the reshape lands
    # each message's chunks in order on its own row, the short one last.
    return fnp.concatenate(
        [
            values[: batch * full].reshape(batch, full, 8),
            values[batch * full :].reshape(batch, 1, 8),
        ],
        axis=1,
    )


def tree_output(msg: ArrayLike, mode: Mode) -> Output:
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
    message = device_message(msg)
    batch, length = message.shape

    nchunks = tree.units(length, CHUNK_LEN)
    if nchunks == 1:
        # A one-chunk message is the root of its own tree, with no parent above
        # it — so it is a chunk output, not a parent output.
        return chunk_output(
            _block_words(message), length, fnp.zeros((batch, 2), dtype=U32), mode
        )

    nodes = _chunk_chaining_values(message, nchunks, mode)
    for pairs, odd in tree.levels(nchunks):
        paired = nodes[:, : 2 * pairs].reshape(batch * pairs, 2, 8)
        parents = chaining_value(
            parent_output(paired[:, 0], paired[:, 1], mode)
        ).reshape(batch, pairs, 8)
        # An odd node at the end has no sibling on this level and pairs with
        # whatever the levels above hand it, so it rides up untouched.
        nodes = fnp.concatenate([parents, nodes[:, -1:]], axis=1) if odd else parents

    return parent_output(nodes[:, 0], nodes[:, 1], mode)


def _bytes_array(data: ArrayLike | bytes) -> Array:
    """Bytes, or an array of them, as uint8 — so a literal needs no wrapping."""
    if isinstance(data, bytes):
        return fnp.asarray(np.frombuffer(data, dtype=np.uint8))
    return fnp.asarray(data, dtype=fnp.uint8)


def context_bytes(context: str | bytes) -> bytes:
    """A derive-key context as the bytes that are actually hashed.

    The standard names the context UTF-8, so a `str` and its encoding are one
    context rather than two. Exported because `Blake3DeriveKey` has to compare
    on the *same* answer this hashes on: a row whose value identity disagreed
    with what it derives would serve one context's trace under another's name,
    and neither side would error.
    """
    return context.encode() if isinstance(context, str) else bytes(context)


def hash_mode() -> Mode:
    """Plain hash mode: the IV as the key, and no mode flag at all.

    Spec section 2.3. The mode every consumer that just wants "the BLAKE3 of
    this" is asking for, and the one the published `hash` column pins.
    """
    return Mode(IV_WORDS, 0)


def keyed_mode(key: ArrayLike | bytes) -> Mode:
    """Keyed hashing: a 32-byte key in place of the IV, `KEYED_HASH` throughout.

    Spec section 2.3. This is the mode that makes BLAKE3 a PRF rather than a
    digest, and the key reaching *every* compression — not only the root's — is
    what stops a chaining value computed under one key being reusable under
    another.

    **The key is an operand, not a host constant**, so it may be a tracer: a
    caller keying per call compiles once and re-keys without re-tracing, and the
    key stays out of the compiled program's constant pool. `bytes` is accepted
    for the literal case and *does* bake in, which is the trade
    [`rows.py`](rows.py) makes to fit the `ByteHash` seam.
    """
    words = _bytes_array(key)
    if words.shape != (KEY_LEN,):
        raise ValueError(f"key must be {KEY_LEN} bytes, got {words.shape}")
    # Little-endian, like every other word BLAKE3 reads — the key is the eight
    # words a compression's chaining value would otherwise be.
    return Mode(pack_le(words), KEYED_HASH)


def derive_key_mode(context: str | bytes) -> Mode:
    """Key derivation's second pass: the context key as the key, and its flag.

    Spec section 2.3's two-pass mode. The context string is itself hashed —
    under `DERIVE_KEY_CONTEXT`, in hash mode's key — and its 32-byte output is
    the key the material pass opens from. Both passes are whole BLAKE3 hashes of
    the same construction, so this is `tree_output` twice rather than anything
    of its own.

    **The context pass runs on the device like any other hash**, even though the
    context is a host constant: hashing it on the host would mean a second
    BLAKE3 in shipped code, and `docs/reference/conventions.md` is explicit that
    two implementations of one standard agreeing is worth nothing.

    **It costs emitted program, and usually not arithmetic.** The pass is
    re-emitted per call site — every `derive_key` in a traced program carries
    its own copy, even at one shared context — which is one extra compression
    per site: a 64-byte message goes from 1 to 2 and a 1025-byte one from 17 to
    18, both countable with `blake3_test._compressions`. That part is exact.

    What happens to those compressions afterwards is XLA's, and it folds them:
    the optimized flop count comes out identical to plain `digest` at every
    context measured, from the one-block separator the standard asks for up to
    three chunks, each compiling in a few seconds. That is an optimizer
    heuristic rather than a property of this code, so it is worth stating as a
    measurement rather than a guarantee — a large context that failed to fold
    would ship the arithmetic instead, and only the flop count would say so.

    A consumer keeping the context to a domain separator, which is what the
    standard mandates, stays well inside the measured range. One tempted to
    hash something large as a context should read the flop count rather than
    assume.

    Memoizing the `Mode` on a caller's object is not the fix it looks like — if
    the first call happens under a trace, the cached `key_words` is a tracer
    that leaks out of it.

    **The context pass does not ride the marker, and is the one hash here that
    does not.** A marked call goes through `tree_hash`'s jit zone, so it compiles
    the whole unrolled body — which is affordable at a domain separator's one
    block, but it is paid at *every* `derive_key` call site with a fresh context
    length, and it buys nothing: the pass hashes a host constant, and the
    optimizer folds it away (the measurement below). Marked, it would also make
    `derive_key` two composites where the contract counts one per digest.

    The output words are taken as words rather than through bytes: a root's
    first eight words *are* its 32-byte digest little-endian, which is exactly
    the packing a key is read with.

    The context pass is a fourth `Mode` value, not a fourth mode: it is key
    derivation's own first half, which is why it is built here rather than
    exported beside the three.
    """
    data = _bytes_array(context_bytes(context))
    context_pass = replace(hash_mode(), flags=DERIVE_KEY_CONTEXT)
    words = root_words(tree_output(data.reshape(1, -1), context_pass))
    return Mode(words[0, :8], DERIVE_KEY_MATERIAL)


def unmarked_hash(msg: ArrayLike, mode: Mode, out_len: int) -> Array:
    """A whole BLAKE3 hash with no marker on it: uint8 `[B, L]` -> `[B, out_len]`.

    The tree and its root output, which is the entire hash — every mode, every
    length. `tree_hash` is this under the `hash_frx.digest.blake3` composite, and an
    unrecognized composite inlines to exactly this, so the two are one
    implementation rather than two spellings that could drift.

    Exported because the suite needs the unmarked side by name: a marked call
    compiles its whole unrolled body, which is affordable at a block and not at a
    chunk (`blake3_test` measures it), so the value tables read the hash here
    while the marker is asserted on the lowered module. Byte-exactness against
    the published vectors therefore still pins one implementation, not a test
    double — `docs/reference/conventions.md` forbids the latter, and rightly.
    """
    return root_bytes(tree_output(msg, mode), out_len)


def pair_bytes(pairs: ArrayLike) -> Array:
    """A parent's two children as the uint8 `[B, 2*32]` batch it compresses:
    left ‖ right chaining values, 32 bytes each.

    Checked here for the reason `byte_hash.device_message` checks a message — a
    caller holding the wrong width otherwise reads a reshape error against an
    intermediate it never wrote, and a `[B, 63]` operand would reshape-fail while
    a `[B, 128]` one would silently hash the wrong pair.

    **Public where a message's coercion is the seam's**, because two callers in
    two layers need one answer: `unmarked_parent_hash` here, and
    `rows.parent_digest` one layer up. `context_bytes` is public in this module
    for the same reason. It does NOT route
    through `device_message` despite the overlapping rank check — the width is
    BLAKE3's and so is the error a caller of `parent_digest` should read, and the
    seam's generic "msg must be 2-D uint8 [B, L]" would name an operand nobody
    passed.
    """
    block = fnp.asarray(pairs, dtype=fnp.uint8)
    if block.ndim != 2:
        raise ValueError(
            f"pairs must be 2-D uint8 [B, {2 * DIGEST_LEN}], got ndim={block.ndim}"
        )
    if block.shape[1] != 2 * DIGEST_LEN:
        raise ValueError(
            f"pairs must be two {DIGEST_LEN}-byte chaining values end to end, "
            f"got {block.shape[1]} bytes"
        )
    return block


def unmarked_non_root_hash(msg: ArrayLike, mode: Mode) -> Array:
    """A whole BLAKE3 tree finished WITHOUT `ROOT`, no marker: uint8 `[B, L]`
    -> `[B, 32]`.

    `unmarked_hash`'s sibling for the non-root contract: the same tree, its
    final node finished by `chaining_value` instead of `root_bytes`. What
    `tree_hash(non_root=True)` inlines to when no emitter is wired.

    Named rather than inlined into that decomposition because it IS the
    decomposition's contract across the module boundary — `rows.py` reads it,
    and a marked region's body being a named function is what lets the marked
    and unmarked sides be one implementation rather than two spellings. Unlike
    its two siblings it has no test caller: `blake3_test` covers the non-root
    path through the marked `non_root_digest` and open-codes this body where it
    needs the unmarked side.
    """
    return unpack_le(chaining_value(tree_output(msg, mode)))


def unmarked_parent_hash(pairs: ArrayLike, mode: Mode) -> Array:
    """One non-root PARENT compression, no marker: uint8 `[B, 64]` -> `[B, 32]`.

    `unmarked_non_root_hash`'s sibling one level up: where that finishes the
    final node of a *message*'s tree, this compresses a pair of chaining values
    the way every internal level of a Merkle tree over BLAKE3's own semantics
    does. Exported by name for the same reason its siblings are — the suite
    reads the unmarked side while the marker is asserted on the lowered module.

    Not expressible as a hash of the 64 bytes: a message of that length sets
    `CHUNK_START|CHUNK_END` and a real counter, where a parent sets `PARENT`
    with a zero counter and opens from the key words (spec section 2.5). The
    two produce different bytes from the same input, which is why the message
    entry cannot be pointed at a parent level.
    """
    block = pair_bytes(pairs)
    words = pack_le(block.reshape(block.shape[0], 2, DIGEST_LEN))
    return unpack_le(chaining_value(parent_output(words[:, 0], words[:, 1], mode)))

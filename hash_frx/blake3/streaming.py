# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Incremental BLAKE3 — the reference `Hasher` as a fixed-shape pytree.

`blake3.digest` hashes a whole message whose length is static, which is not what
a caller holding a *prefix* has: a byte Fiat-Shamir transcript absorbs a round's
framing at a time, and how much it has absorbed by round `k` is a runtime value
inside the loop. So this keeps BLAKE3's hasher between calls, the way
`ShakeAbsorb` keeps the sponge and `Sha256State` the Merkle-Damgård midstate.

**Fixed shapes, not a growing buffer.** A state that kept the absorbed bytes and
hashed them on demand cannot be a loop carry: the used length grows with every
absorb, so inside a `while` it is traced, and a one-shot digest needs a static
message shape. Every field here is shaped at `blake3_stream_init` and stays that
shape, so absorbing a megabyte leaves the treedef and every leaf exactly as they
were — `streaming_test.test_state_shape_is_absorb_invariant` is that property,
and it is the whole reason the module exists.

Runtime lengths still work because BLAKE3 takes a partial block's length as an
operand (spec section 2.4): the trailing block is zero-padded to 64 bytes and
its true byte count rides in `block_len`, so "how many bytes are real" is a
value rather than a shape.

Structure follows the reference implementation's `Hasher` (spec section 5.1): a
chunk state — the running chaining value, the block being filled, how many
blocks of this chunk are already compressed — over a stack of completed subtree
chaining values, merged whenever the completed-chunk count turns even.

`CHUNK_END` is why a full block is kept rather than compressed eagerly: a
chunk's last compression carries a different flag from its others, and which
block is last is only known once more input arrives. The reference makes the
same choice for the same reason.

**Two counts are traced, and both are `while_loop`s.** How many subtree merges
a finished chunk triggers, and how many stack entries a finalize folds, are
functions of the chunk count — absorbed data rather than a shape. `ShakeAbsorb`
meets the same problem in its block schedule and answers it by running to a
static upper bound and selecting; that answer does not carry here, because the
bound is the stack's whole 54 levels against an amortized *one* merge per chunk,
so the select would pay 54 compressions for one. Either way the stack's shape
does not move — only `stack_len` does, which is what the state's invariance
needs.

**The marked region here is the compression, not the hash.** The
`hash_frx.digest.blake3` composite spans a whole hash — chunks, tree and root output —
which is exactly what a resumable state does not hold, so it cannot serve this
path. What a resumable state does repeat is the compression, the way a sponge
repeats its permutation and a streaming SHAKE rides that marker on every block.
So `_compress1` carries `hash_frx.compress.blake3`, and the three hops that
finish one node ride it: the absorb path's block, the subtree merge, and
finalize's stack fold. The two traced counts above stay outside it.

The root read does not. `blake3.root_bytes` repeats one node's compression at an
output-block counter running 0, 1, 2 …, which is a batch of rows rather than a
node — the one place BLAKE3's own `[B, ...]` primitive is already the right
shape, and re-spelling it here would fork the extendable-output logic that lives
with the tree. One unmarked compression per finalize, against one per absorbed
block and one per merge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import frx
import frx.numpy as fnp
from frx import Array, lax
from frx.tree_util import register_dataclass

from hash_frx.blake3 import blake3
from hash_frx.blake3.compress import CHUNK_END, CHUNK_START, PARENT, compress
from hash_frx.fusion import fused_region
from hash_frx.word import pack_le

U32 = fnp.uint32

BLOCK_LEN = blake3.BLOCK_LEN  # 64
# A chaining value is the low half of a compression's sixteen output words.
CV_WORDS = 8
BLOCKS_PER_CHUNK = blake3.CHUNK_LEN // BLOCK_LEN  # 16
# The reference's bound: a 2^64-chunk input needs 54 levels of subtree stack.
MAX_STACK = 54

# Hash mode (spec section 2.3), held once for the module: the key words every
# chunk and every parent opens from are the IV, and the mode contributes no flag.
# A keyed state would differ in exactly these two values and nothing else here,
# which is why they are named rather than inlined.
_MODE = blake3.hash_mode()
_KEY_WORDS = _MODE.key_words
_MODE_FLAGS = U32(_MODE.flags)


def _compress1(
    cv: Array, block_words: Array, counter: Array, block_len: Array, flags: Array
) -> Array:
    """`compress` on a single row, unbatched in and out, as one generic marked
    region.

    The batch axis is where BLAKE3's parallelism lives, and a resumable state has
    none of it — one chunk, one block, one stack entry at a time — so every call
    here is the `[1, ...]` spelling of the batched primitive rather than a second
    compression function.

    Marked here rather than in `compress`: a stream repeats this compression the
    way a sponge repeats its permutation, so this is the region a streaming
    consumer needs, while marking `compress` itself would nest a composite
    inside `hash_frx.digest.blake3` and `hash_frx.compress.blake3_parent`,
    whose emitters read a plain body.

    Name-routed, not generic. The generic rewriter declines a body with no
    live-width operand — it would otherwise route to a LoopFusion whose indexed
    per-element subgraph re-executes shared producers per output element, which
    for a compression (every output word depends on the whole state) is the
    worst case — so a generic marker here lowers and then silently inlines. An
    unrecognized *name* inlines too, which is why this is safe to emit before
    the plugin ships its arm: bytes stay right and only the fusion waits.

    `iv` is passed rather than defaulted: `compress`'s ABI note says a caller
    under a marked region hands in the region's operand, since a captured
    constant would be lifted ahead of the explicit ones. `compress` is the
    decomposition directly — it already takes the region's operands in the
    region's order, and a wrapper closure would be re-traced per call site.
    """
    # The region's ABI is the batched one — `[1, ...]` here — not this call
    # site's unbatched row. An emitter for it is a row kernel like every other
    # BLAKE3 arm, so a degenerate leading axis costs nothing and keeps one
    # kernel able to serve a batched caller; the alternative would be a
    # rank-1-and-scalar special case that only a stream can use.
    return fused_region(
        compress,
        cv[None, :],
        block_words[None, :],
        counter[None, :],
        block_len[None],
        flags[None],
        _MODE.iv,
        name=blake3.BLAKE3_COMPRESS_MARKER,
        version=blake3.BLAKE3_COMPRESS_MARKER_VERSION,
    )[0]


def _counter_inc(counter: Array) -> Array:
    """The 64-bit chunk counter held as (low, high) uint32, plus one."""
    lo = counter[0] + U32(1)
    carry = (lo == U32(0)).astype(U32)
    return fnp.stack([lo, counter[1] + carry])


def _bit(counter: Array, i: Array) -> Array:
    """Bit `i` of the 64-bit `counter` held as (low, high) uint32."""
    lo, hi = counter[0], counter[1]
    word = fnp.where(i < U32(32), lo, hi)
    return (word >> (i & U32(31))) & U32(1)


def _output(
    icv: Array, blk: Array, ctr: Array, blen: Array, flags: Array
) -> blake3.Output:
    """A one-row unrun node. `iv` is the mode's and constant, so it rides here
    rather than through the merge carry."""
    return blake3.Output(
        icv[None, :],
        blk[None, :],
        ctr[None, :],
        blen[None],
        flags[None],
        _MODE.iv,
    )


def _push_chunk_cv(state: Blake3Stream, cv: Array) -> Blake3Stream:
    """Add a finished chunk's chaining value, merging while the completed-chunk
    count is even (spec section 5.1.2).

    The merge count is data-dependent, so it is a `while_loop` — the stack's
    shape does not move, only `stack_len`. The stack rides as a closure rather
    than in the carry, as `finalize`'s merge does: the body only reads it, and
    threading `[MAX_STACK, 8]` through the loop state says otherwise.
    """
    total = _counter_inc(state.counter)  # chunks completed after this one

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        _, slen, shift = carry
        # Merge while the completed-chunk count is even at this level — and
        # never below an empty stack, which `total` alone does not rule out.
        return (slen > U32(0)) & (_bit(total, shift) == U32(0))

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        cv_, slen, shift = carry
        slen = slen - U32(1)
        left = lax.dynamic_index_in_dim(state.cv_stack, slen, axis=0, keepdims=False)
        # `parent_output` assembled and `chaining_value` run, spelled through
        # the marked compression instead: a parent opens from the key words,
        # its counter is zero however deep the tree is, its block is the two
        # children end to end, and PARENT rides over the mode (spec 2.5). The
        # same operands, so the bytes are `parent_output`'s by construction.
        merged = _compress1(
            _KEY_WORDS,
            fnp.concatenate([left, cv_]),
            fnp.zeros((2,), U32),
            U32(BLOCK_LEN),
            _MODE_FLAGS | U32(PARENT),
        )[:CV_WORDS]
        return merged, slen, shift + U32(1)

    # The reference tests bit 0 of the completed-chunk count first, then shifts;
    # starting at 1 would skip the first merge and corrupt every even chunk.
    cv, stack_len, _ = lax.while_loop(cond, body, (cv, state.stack_len, U32(0)))
    stack = lax.dynamic_update_index_in_dim(state.cv_stack, cv, stack_len, axis=0)
    return replace(
        state,
        cv_stack=stack,
        stack_len=stack_len + U32(1),
        counter=total,
        chunk_cv=_KEY_WORDS,
        compressed=U32(0),
    )


def _absorb_block(state: Blake3Stream, block: Array) -> Blake3Stream:
    """Compress one full 64-byte block into the current chunk.

    The chunk's 16th block is its last, so it carries `CHUNK_END` and its result
    is the chunk's chaining value rather than a running one.
    """
    last = state.compressed == U32(BLOCKS_PER_CHUNK - 1)
    flags = (
        _MODE_FLAGS
        | fnp.where(state.compressed == U32(0), U32(CHUNK_START), U32(0))
        | fnp.where(last, U32(CHUNK_END), U32(0))
    )
    cv = _compress1(
        state.chunk_cv, pack_le(block), state.counter, U32(BLOCK_LEN), flags
    )[:8]
    advanced = replace(state, chunk_cv=cv, compressed=state.compressed + U32(1))
    return lax.cond(last, lambda s: _push_chunk_cv(s, cv), lambda s: s, advanced)


@register_dataclass
@dataclass(frozen=True)
class Blake3Stream:
    """A resumable BLAKE3 hash state. Every field's shape is absorb-invariant.

    cv_stack   uint32 [MAX_STACK, 8]  completed subtree chaining values
    stack_len  uint32 []              how many of them are live
    chunk_cv   uint32 [8]             the current chunk's running chaining value
    counter    uint32 [2]             the current chunk's index, (low, high)
    block      uint8  [BLOCK_LEN]     the block being filled, zero-padded
    block_len  uint32 []              how much of `block` is real
    compressed uint32 []              blocks of this chunk already compressed

    Nothing rides as a `meta_field`. `ShakeAbsorb` keeps its rate as static aux
    because a sponge is parameterised by one; BLAKE3 fixes every bound it has —
    the 64-byte block, the 16 blocks of a chunk, the 54-level stack — so there is
    no parameterisation to hold out of the leaves, and one treedef serves every
    state.
    """

    cv_stack: Array
    stack_len: Array
    chunk_cv: Array
    counter: Array
    block: Array
    block_len: Array
    compressed: Array

    @frx.jit
    def absorb(self, data: Array) -> Blake3Stream:
        """Absorb `data` (uint8 `[L]`, `L` static), returning the new state.

        Any split of a message absorbs to the same state as absorbing it whole,
        which is the only property a streaming hash has to have.

        **A module-level jit zone rather than an inlined body**, so the schedule
        is emitted once and every call site shares it. It is 1,783 StableHLO ops,
        and the same 1,783 at 3 bytes as at 5,000 — the block loop is traced once
        — so a caller absorbing three times per round would re-emit all of it
        three times, under a round loop unrolled on top of that.
        `blake3.tree_hash` is a zone for the same reason and gives the opposite
        `inline=` answer: that one has to splice back in to keep one composite
        per hash, and there is no composite here to keep.
        """
        msg = fnp.asarray(data)
        # Coercing instead would hash a truncation of what the caller passed and
        # return a well-formed digest of the wrong message — an int32 payload
        # silently narrows mod 256, a [B, L] one silently flattens into a single
        # stream. Same reason `keccak.streaming.ShakeAbsorb.absorb` rejects both.
        if msg.ndim != 1:
            raise ValueError(f"data must be 1-D uint8 [L], got ndim={msg.ndim}")
        if msg.dtype != fnp.uint8:
            raise TypeError(f"data must be uint8, got {msg.dtype}")
        length = msg.shape[0]
        if length == 0:
            return self

        # Lay the buffered partial block and the new bytes out contiguously. The
        # write offset is a runtime value; the shapes are not. The trailing
        # block of headroom is what keeps the tail read below in bounds: the
        # last block starts at most one block short of the written end, so its
        # 64-byte window can run past it. Reserving it here rather than
        # concatenating it on at the read costs nothing and copies nothing.
        work = fnp.zeros((2 * BLOCK_LEN + length,), fnp.uint8)
        work = lax.dynamic_update_slice(work, self.block, (0,))
        work = lax.dynamic_update_slice(work, msg, (self.block_len,))

        total = self.block_len + U32(length)
        # Keep the final block un-compressed: which one is a chunk's last is only
        # known once more input arrives.
        nblocks = (total - U32(1)) // U32(BLOCK_LEN)
        max_blocks = (BLOCK_LEN + length - 1) // BLOCK_LEN

        def step(i: Array, carry: Blake3Stream) -> Blake3Stream:
            return lax.cond(
                U32(i) < nblocks,
                lambda s: _absorb_block(
                    s,
                    lax.dynamic_slice(work, (U32(i) * U32(BLOCK_LEN),), (BLOCK_LEN,)),
                ),
                lambda s: s,
                carry,
            )

        state = lax.fori_loop(0, max_blocks, step, self)

        # What is left over is the tail after the compressed blocks.
        rest = total - nblocks * U32(BLOCK_LEN)
        tail = lax.dynamic_slice(work, (nblocks * U32(BLOCK_LEN),), (BLOCK_LEN,))
        keep = fnp.arange(BLOCK_LEN, dtype=U32) < rest
        return replace(state, block=fnp.where(keep, tail, fnp.uint8(0)), block_len=rest)

    @partial(frx.jit, static_argnums=(1,))
    def finalize(self, out_len: int) -> Array:
        """Read `out_len` bytes of the root's extendable output: uint8 `[out_len]`.

        Non-mutating — the state is unchanged, matching the reference, so a
        caller can read the hash of what it has absorbed and go on absorbing.
        That is what `ShakeSqueeze` cannot offer and does not pretend to: FIPS
        202 forbids an absorb after a squeeze begins, where the BLAKE3 root is a
        compression of the tree rather than a state the read consumes. Shared
        across call sites for the reason `absorb` is.
        """
        root_flags = (
            _MODE_FLAGS
            | fnp.where(self.compressed == U32(0), U32(CHUNK_START), U32(0))
            | U32(CHUNK_END)
        )
        # `Output` is an operand record, not a pytree (its `__eq__`/`__hash__`
        # are element-wise over Arrays and raise), so the merge carries its five
        # fields as plain arrays and rebuilds the record at the end.
        carry = (
            self.chunk_cv,
            pack_le(self.block),
            self.counter,
            self.block_len,
            root_flags,
            self.stack_len,
        )

        def cond(state: tuple[Array, ...]) -> Array:
            return state[-1] > U32(0)

        def body(state: tuple[Array, ...]) -> tuple[Array, ...]:
            icv, blk, ctr, blen, flags, slen = state
            slen = slen - U32(1)
            left = lax.dynamic_index_in_dim(self.cv_stack, slen, axis=0, keepdims=False)
            cv = _compress1(icv, blk, ctr, blen, flags)[:CV_WORDS]
            parent = blake3.parent_output(left[None, :], cv[None, :], _MODE)
            return (
                parent.input_chaining_value[0],
                parent.block[0],
                parent.counter[0],
                parent.block_len[0],
                parent.flags[0],
                slen,
            )

        icv, blk, ctr, blen, flags, _ = lax.while_loop(cond, body, carry)
        return blake3.root_bytes(_output(icv, blk, ctr, blen, flags), out_len)[0]


def blake3_stream_init() -> Blake3Stream:
    """A fresh incremental BLAKE3 in hash mode (no bytes absorbed)."""
    return Blake3Stream(
        cv_stack=fnp.zeros((MAX_STACK, 8), U32),
        stack_len=U32(0),
        chunk_cv=_KEY_WORDS,
        counter=fnp.zeros((2,), U32),
        block=fnp.zeros((BLOCK_LEN,), fnp.uint8),
        block_len=U32(0),
        compressed=U32(0),
    )

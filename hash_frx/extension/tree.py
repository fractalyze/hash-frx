# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Tree: the schedule that folds a message's chunks down to one root node.

The third of the model's three extensions, beside Merkle-Damgard and the sponge,
and the one with a single implementation: BLAKE3 is this package's only tree
hash. So the generality here is exactly what BLAKE3 needs and no more — the rule
`extension/md.py` and `extension/sponge.py` were held to, and the one #169 gives
for it: a module written against one implementation and *shaped* for an imagined
second is shaped wrong, because the second one re-reads it anyway.

**Nothing here emits an op, and that is a constraint rather than an
observation.** `extension/sponge.py` states why and it is worth restating,
because it is the whole reason a tree schedule can be factored out at all:
Python evaluates a call's arguments before its body, so a helper that *takes* an
array has already fixed where that array's ops sit relative to its own. Two call
sites that build the same value in opposite orders cannot both route through one
such helper — #229 measured that on Keccak and Ascon, where one shared merge
helper reordered all 32 of Ascon's lowered cases without changing a single
operation. A schedule that emits nothing has no order to impose, so every caller
keeps its own spelling.

So the functions below are host arithmetic and control flow: they answer *how
many* and *which pairs with which*, and the caller answers *with what ops*. That
is also what makes them testable without a device, which is the layer these rules
actually live on — a case written over BLAKE3 alone passes with the schedule
wrong and BLAKE3's own spelling right.

**`compress_block` is a plain callback, and not a `CompressionFunction`.** The
seam was proposed for this module and is declined here, with the three
signatures that would have to fit it on the table. #228 added one, removed it
again inside the same branch, and recorded why — it had no implementors and the
shape had been chosen before the families were read. The honest candidate set is
the three that take a per-block counter and flag at all, so those three were read
before deciding:

- BLAKE3 `compress(cv[B,8], block[B,16], counter[B,2], block_len[B], flags[B],
  iv[8]) -> [B,16]`
- BLAKE2b `_compress(state[B,16], iv_lo[8], iv_hi[8], w32[B,32], t: int, f: bool)
  -> [B,16]`
- BLAKE2s `_compress(state[B,8], iv_a[4], iv_b[4], w32[B,16], t: int, f: bool)
  -> [B,8]`

They disagree on every axis a seam would have to fix. BLAKE3's counter and flags
are **device operands**, because chunks batch and the counter varies per row;
BLAKE2's `t` and `f` are **host values** folded into the working vector as
scalar-literal XORs, because the message length is static. That is the same
concept on opposite sides of the host/device boundary, and a seam spanning it has
to move one of them across: pushing BLAKE2's scalars onto the device adds
operands to a marked region, which is an ABI change, and pulling BLAKE3's onto the
host is not possible at all. The IV arrives as one array, two halves, or two
4-lane rows respectively — pre-split by the caller in BLAKE2's case, deliberately.
BLAKE3 carries a `block_len` operand for the partial trailing block (spec section
2.4) that neither BLAKE2 has, and returns **double** its state width where both
BLAKE2s return their own.

A seam is worth its cost when it lets one consumer name several implementations.
This one would name three families that cannot be called through it without
changing one of their wire ABIs, in a package that already spells `Compression`
for an unrelated construction (`hash_frx/compression.py`, truncated-permutation
n-to-1). So the schedule takes the callback, and the seam lands if and when a
second tree hash gives it an implementor to be designed against.

**The three rules, and they are the whole module.**

- A chunk is a *chain*: every block feeds the next, so there is no parallelism
  inside one and `chain` is a plain loop over a static count.
- A tree level is a *batch*: nodes on one level do not depend on each other, so
  `levels` plans one batched call per level and an odd trailing node rides up
  unpaired. Pairing from the bottom that way is the spec's own tree shape for
  every chunk count, which `levels` argues.
- A node's final compression stays *pending*: one node becomes a chaining value
  under a tree and a digest at the root, and only that last call differs. The
  schedule therefore never runs it — `levels` stops with two nodes left, and what
  the caller does with them is the root's business.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeVar

# One node, in whatever shape the caller carries it — a `[B, 8]` chaining value
# for BLAKE3. `chain` only threads it between the caller's own callbacks, so it
# never needs to know which.
N = TypeVar("N")


def units(length: int, size: int) -> int:
    """`length` bytes as a count of `size`-byte units — empty still occupies one.

    Blocks within a chunk and chunks within a message are the same ceiling with
    the same floor, and the floor is the whole subtlety: an empty message is one
    empty chunk holding one empty block, not zero of either. A plain ceiling
    gives a tree with no nodes and a chunk with nothing to compress, and the
    empty digest is a published vector — so getting this wrong fails at exactly
    one length and nowhere else.

    **This is the module's counting rule, and it answers two questions**: how
    many blocks a chunk holds, and how many compressions the root's output
    stream spans at 64 bytes each (spec section 2.6). A separate floor-less
    spelling for the second was tried and removed — `blake3.modes.root_bytes`
    rejects a zero-byte request before asking, so the floor is unreachable
    there and the two answers never differ. That guard lives with the caller
    and this module cannot check it, which is the price of being dep-free.

    **`sponge.squeeze_blocks` is the same ceiling WITHOUT the floor**, and that
    is why the two do not merge — not the dependency, which would only bite in
    the direction nobody would take (`sponge` already carries `frx` and could
    depend on this target for free). Folding them would move
    `squeeze_blocks(0, rate)` from 0 to 1. A caller that wants an output count
    rather than a unit count therefore has to mean the floor, not inherit it.
    """
    return max(1, -(-length // size))


def chain(node: N, *, count: int, compress_block: Callable[[N, int], N]) -> N:
    """Fold `count` blocks into `node` one at a time, each feeding the next.

    `compress_block(node, i)` compresses block `i` and returns the running value.
    Where that block comes from, and which flags its position carries, are the
    caller's: the schedule knows only that the blocks are sequential and that
    there is no parallelism to be had inside one chunk.

    The loop is Python-unrolled because the count is static — a chunk holds a
    fixed number of blocks — and a `lax` loop would be a control-flow boundary
    (`docs/reference/conventions.md`) around a body with no runtime bound to
    justify one.

    **This is not `md.chain`, though it is the same shape.** That walks a whole
    *message*'s blocks from one midstate and serializes; this walks the blocks of
    *one chunk*, of which a message has many and which are independent of each
    other. That independence is what makes the tree parallel at all, so the two
    stay apart rather than collapsing into one body with a flag — the same call
    `sponge.py` records for its two loop forms.
    """
    for i in range(count):  # static, and at most a chunk's worth
        node = compress_block(node, i)
    return node


def levels(nodes: int) -> Iterator[tuple[int, bool]]:
    """The reduction plan for `nodes` bottom nodes, a level at a time from the
    bottom up: `(pairs, odd)` — how many parent compressions this level is, and
    whether a trailing node rides up to the next one unpaired.

    Stops with **two** nodes left rather than one, because the last pair is the
    root and a root's compression is the caller's to run: it is where `ROOT` and
    the extendable output live, and neither belongs to the tree that produced it.
    `nodes` must therefore be at least two — a one-chunk message has no parent
    above it at all and is its own root, which is a case the caller answers
    before reaching here.

    **Pairing from the bottom is the spec's tree, not an approximation of it.**
    The spec splits a node by giving its left subtree the largest power of two
    strictly below the chunk count — a split that is not a halving, and that makes
    every left subtree perfect. Pairing adjacent nodes from the bottom and
    carrying an odd trailing node up produces exactly that shape at every count:
    a perfect left subtree is a power of two, so it consumes whole levels without
    ever leaving a remainder, and every odd node that does arise therefore belongs
    to the right subtree — which is where the spec puts it.

    What that buys is why this construction was chosen at all: `ceil(log2(n))`
    batched compressions rather than a walk over `2n - 1` nodes.

    The odd node rides up **untouched** — not paired with a copy of itself, not
    padded against a zero node. It has no sibling on this level and pairs with
    whatever the levels above hand it; compressing it early is a different tree
    and a different digest.
    """
    if nodes < 2:
        raise ValueError(
            f"a tree reduction needs at least two nodes, got {nodes} — a single "
            "node is its own root and has no parent to assemble"
        )
    while nodes > 2:
        pairs, odd = divmod(nodes, 2)
        yield pairs, bool(odd)
        nodes = pairs + odd

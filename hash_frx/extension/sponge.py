# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Sponge: the schedules that feed a permutation a message.

Two of them, and the split is the measurement rather than the intention. Both
run the same construction — merge a block into the rate, permute, repeat; then
read the rate, permute, read again until enough output is out — but they differ
on the one axis that does not merge, the **loop form**:

- `absorb_squeeze` walks a STATIC block count with a Python `for`. Both byte
  sponges take it: `ByteHash` fixes the message length, so the block and squeeze
  counts are known at trace time and the message is only ever an operand.
- `field_absorb` walks a runtime counter with a `lax.while_loop`. The field
  sponge takes it, because it reads its bound at runtime so a concrete and a
  symbolic `n` lower the same way.

A body parameterized on which of those two it is would be two bodies behind one
name, so they are two functions sharing one vocabulary — `pad.SpongePad`,
`squeeze_blocks`, and the squeeze rule below. The parent epic's XLA spike
reached the same split from the other side: byte and field absorbs differ only
in lane load and merge, but field trip counts are runtime operands where byte
ones are static, so the envelopes keep both forms.

**The squeeze rule is the part worth reading, and it is why this is shared at
all.** The absorb's final permutation has already run when the squeeze starts,
so the first output block is available before any further permutation — and a
permutation after the LAST read would be work whose result nothing reads. Keccak
spelled that by guarding the permute (`if i + 1 < squeezes`) and Ascon by peeling
the first read out of the loop; the two spellings are one rule, and getting it
wrong shifts every output block by one permutation. That is a wrong digest, not a
slow one, and it is the same class of error as the duplex squeeze fix that
`duplex_sponge.py` carries.

`absorb_squeeze` emits no op of its own: every array operation is the caller's,
reached through `absorb` / `permute` / `read`, which is what lets one schedule
serve a flat `[B, 50]` uint32 state and a pair of `[B, 5]` word grids without
either family's lowering moving. `field_absorb` owns its `while_loop` and the
carry around it — precisely the axis that does not merge — and nothing else.

**The merge is deliberately not here.** Both byte sponges combine a block into
the rate with the same slice-and-concatenate, but they emit it in opposite order:
Keccak packs its block before merging, so the block's ops precede the state
slice, while Ascon slices S0 first. A helper takes its block as an argument, so
the block is always emitted first — one order, and only one of the two can have
it. The shared form therefore folds the two *Keccak* spellings and lives with
them (`keccak.sponge._xor_into_rate`); Ascon keeps its own two lines. What
generalizes here is the schedule, which emits nothing and so has no order to
impose.

**The squeeze trim is the caller's**, for the same reason: the schedule reads
whole rate blocks, so a request that is not a multiple of the rate overshoots
and the caller slices. Keccak slices unconditionally, Ascon only when the
request is not a multiple of 8 — because Ascon-Hash256's 32 bytes are four exact
blocks and an unconditional slice would put an op in its region that was never
there.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import frx.numpy as fnp
from frx import Array, lax

# The sponge state, whatever the family carries it as: a flat lane array for
# Keccak, a (lo, hi) pair of word grids for Ascon, a rate+capacity vector for the
# field sponge. The schedules only thread it between the caller's own callbacks,
# so they never need to know which.
S = TypeVar("S")
# One squeeze block, in whatever shape the caller reads it out as.
R = TypeVar("R")


def squeeze_blocks(output_size: int, rate: int) -> int:
    """Permutations' worth of output a `output_size`-byte request spans, at
    `rate` bytes each — rounded up.

    No floor, unlike `tree.units`: a zero-byte squeeze is zero permutations, and
    every row here asks for a fixed positive length so the case is unreachable
    rather than handled. That difference is why the two ceilings do not merge.
    """
    return -(-output_size // rate)


def absorb_squeeze(
    state: S,
    *,
    blocks: int,
    squeezes: int,
    absorb: Callable[[S, int], S],
    permute: Callable[[S], S],
    read: Callable[[S], R],
) -> list[R]:
    """The byte sponges' schedule over STATIC counts: absorb `blocks` blocks,
    then read `squeezes` output blocks. Returns the blocks read, in order.

    `absorb(state, i)` merges block `i` and does nothing else — the permutation
    that follows it is this schedule's, not the caller's, because "every absorbed
    block is followed by a permutation" is the construction rather than the
    family. `read(state)` takes one output block out of the rate in whatever
    shape the caller assembles; the schedule never looks inside it.

    Both loops are Python-unrolled: the counts are static, and a `lax` loop would
    be a control-flow boundary (`docs/reference/conventions.md`). The permute
    between reads is guarded rather than unconditional — see the module
    docstring for why that guard is the whole point.
    """
    for i in range(blocks):
        state = permute(absorb(state, i))

    out: list[R] = []
    for i in range(squeezes):
        out.append(read(state))
        # No permutation after the last read: its result would be work nothing
        # reads, and running it before the FIRST read would shift every output
        # block by one permutation.
        if i + 1 < squeezes:
            state = permute(state)
    return out


def field_absorb(
    state: Array,
    *,
    blocks: Array | int,
    absorb: Callable[[Array, Array], Array],
    permute: Callable[[Array], Array],
) -> Array:
    """The field sponge's schedule over a RUNTIME count: `blocks` rounds of
    `permute(absorb(state, i))`, as a `lax.while_loop`.

    `absorb(state, i)` takes the traced block index rather than a Python one,
    which is the whole difference from `absorb_squeeze` above: the bound is read
    at runtime, so a concrete and a symbolic block count lower the same way.

    No squeeze half, because the construction this serves does not have one: the
    field sponge reads its digest by truncating the final state, which is one
    read with no permutation around it. A row that wanted an iterated squeeze
    over a runtime count would grow one here; none does.
    """

    def cond(carry: tuple[Array, Array]) -> Array:
        return carry[1] < blocks

    def body(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        s, i = carry
        return permute(absorb(s, i)), i + 1

    final, _ = lax.while_loop(cond, body, (state, fnp.int32(0)))
    return final

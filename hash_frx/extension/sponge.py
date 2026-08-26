# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Sponge: the schedules that feed a permutation a message.

Three of them, and the split is the measurement rather than the intention. All
run the same construction — merge a block into the rate, permute, repeat; then
read the rate, permute, read again until enough output is out — but they differ
on the one axis that does not merge, the **loop form**:

- `absorb_squeeze` walks a STATIC block count with a Python `for`. Both byte
  sponges take it: `ByteHash` fixes the message length, so the block and squeeze
  counts are known at trace time and the message is only ever an operand.
- `field_absorb` walks a runtime counter with a `lax.while_loop`. The field
  sponge takes it, because it reads its bound at runtime so a concrete and a
  symbolic `n` lower the same way.
- `scanned_absorb` walks a STATIC count with a `lax.scan`, for the case where
  that count is large enough that unrolling it costs more than the loop boundary
  saves. Only a caller whose marked region is the *permutation* may take it — one
  whose region is the whole hash cannot contain a `while` and must unroll.

A body parameterized on which of those three it is would be three bodies behind
one name, so they are separate functions sharing one vocabulary —
`pad.SpongePad`, `squeeze_blocks`, and the squeeze rule below, which
`absorb_squeeze` and a scanned caller both reach through `squeeze`. The parent
epic's XLA spike reached the first split from the other side: byte and field
absorbs differ only in lane load and merge, but field trip counts are runtime
operands where byte ones are static, so the envelopes keep both forms.

The third arrived later and from a consumer: "static" and "small" are two claims
and the contract's unrolling rule was written where both held. See
`scanned_absorb` for the measurement that separates them.

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

    return squeeze(state, squeezes=squeezes, permute=permute, read=read)


def squeeze(
    state: S,
    *,
    squeezes: int,
    permute: Callable[[S], S],
    read: Callable[[S], R],
) -> list[R]:
    """Read `squeezes` output blocks off `state`, permuting between them.

    The squeeze rule on its own, so that an absorb which is not
    `absorb_squeeze`'s — a scanned one, say — reaches it without restating the
    guard. The module docstring is where the rule is argued; this is the one
    place it is spelled.

    Assumes the absorb's final permutation has already run, which every schedule
    here guarantees.
    """
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


def scanned_absorb(
    state: S,
    blocks: Array,
    *,
    absorb: Callable[[S, Array], S],
    permute: Callable[[S], S],
    reverse: bool = False,
) -> S:
    """The schedule for a STATIC block count too large to unroll: `blocks` rounds
    of `permute(absorb(state, block))`, as a `lax.scan`.

    The third loop form, and it exists because "static" and "small" came apart.
    `absorb_squeeze` unrolls because the contract's rule says a round loop
    should — round counts are static *and* small, 24 forever. An absorb's count
    is `len(message) / rate`, which the caller chooses and nothing bounds, and
    past a few dozen blocks the unrolled graph costs more to compile than the
    loop boundary costs to keep: measured on `Sha3_256` at rate 136, one traced
    `digest` per length, the jaxpr grows ~22 eqns per block while the compile
    goes 5.9 s at 16 blocks, 84.6 s at 32 and 137.0 s at 64, with XLA's own
    slow-compile alarm firing past that.

    **This is not a replacement for `absorb_squeeze`, and the difference is where
    the marked region sits.** Where the whole hash is one composite, a
    `stablehlo.while` cannot live inside it and unrolling is the only option —
    that is the rule the contract protects. Where the marked region is the
    *permutation*, the loop is outside it and the scan costs nothing: the region
    moves into the while body rather than being destroyed, and there is no
    cross-iteration fusion to break because each permutation was already its own
    kernel. A caller that does not know which of the two it has should assume the
    first.

    That split is not new here: `sponge.py` already runs `field_absorb`'s
    `while_loop` on the generic path and reserves its whole-hash region for the
    dedicated one. This is the same arrangement for a count that is static.

    `blocks` carries the block *values* stacked on a leading axis, not an index:
    a scan feeds its body the row, and an index would put a `dynamic_slice` in
    the caller's graph where a static slice used to be. `reverse=True` walks the
    rows from the back, which is what a caller whose layout puts block 0 last
    wants — it costs no `reverse` op, unlike reversing `blocks` itself.

    **The `absorb`/`permute` pair must be stable across calls.** `lax.scan` keys
    its trace cache on the body's identity, and this builds the body from those
    two, so callables rebuilt per call recompile an identical graph every time —
    measured at 0.626 s against 0.001 s for a memoized pair, which is *worse*
    than the unrolled loop this replaces. Memoize them on whatever they close
    over. Inside a `@jit` zone none of this applies, since the jit cache absorbs
    it.
    """

    def step(carry: S, block: Array) -> tuple[S, None]:
        return permute(absorb(carry, block)), None

    final, _ = lax.scan(step, state, blocks, reverse=reverse)
    return final

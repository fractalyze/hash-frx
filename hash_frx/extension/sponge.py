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
- `scanned_absorb` walks a STATIC count with a `lax.scan`, taking its blocks as
  stacked *values* so the caller indexes nothing. It is what a caller reaches for
  when the count is large enough that unrolling costs more than a loop boundary.

So the loop form is what a caller's *count* picks, and the three differ again on
what each hands `absorb` — a Python `int`, a traced index, or the row itself.
That second axis is why these are three functions rather than one with the loop
as a parameter: a shared body would have to fix one block contract, and
`blocks/hash.md` already records why a helper that takes a block and one that
takes an index are not the same helper. They share one vocabulary instead —
`pad.SpongePad`, `squeeze_blocks`, and the squeeze rule below, which
`absorb_squeeze` and a scanned caller both reach through `squeeze`. The parent
epic's XLA spike reached the first split from the other side: byte and field
absorbs differ only in lane load and merge, but field trip counts are runtime
operands where byte ones are static, so the envelopes keep both forms.

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
from functools import lru_cache
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

    The absorb is Python-unrolled: the count is static, and a `lax` loop would be
    a control-flow boundary (`docs/reference/conventions.md`) — see
    `scanned_absorb` for when that stops being the right trade. The squeeze is
    `squeeze`'s.

    **What the unrolling costs a marked caller, since the paragraph above says
    only why it is done.** This body is what `lax.composite` traces to build a
    whole-hash region, so a caller inside one has an O(1) top-level graph — three
    equations, whatever the block count — and a compile that is nonetheless
    LINEAR in it, about 0.2 s per absorb block, paid again per distinct shape.
    The region's emitter then discards this body and rolls the block loop itself
    (one `scf.for`), so that cost buys nothing at run time; the body is here to
    be traced, and to be correct on a backend that declines the marker.

    **The mitigation is not a different route.** Declining the whole-hash marker
    to get the per-permutation path instead is worse on both axes, because that
    path unrolls too and its graph grows with the block count rather than staying
    at three equations: measured on `Sha3_256` at rate 136, 64 absorb blocks, the
    marked arm compiles in 13.0 s against 104.3 s and runs a batch of 128 in
    1.5 ms against 19.2 ms. The generic path shares a *kernel* across message
    lengths, which is the tempting part, but not a *module compile*. A caller
    that minds the per-shape cost wants a `jit` zone or bucketed lengths, not a
    marker it declines.
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

    **This is the TERMINATING squeeze**, which is why no permutation follows the
    last read. A RESUMABLE one ends with that permutation instead, because its
    carry has to point at the next block rather than at the one just returned —
    `keccak.streaming.ShakeBlockSqueeze.squeeze_blocks` is that shape, and it
    does not route through here. Worth naming because a trailing permute reads
    like the dead one this rule forbids, and a genuinely dead one was removed
    from the sibling schedule once already.
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

    **What separates this from `field_absorb` is the block contract, not the
    unrolling.** `field_absorb` takes a static `int` bound too, so it already
    walks a static count without unrolling — but it hands `absorb` a *traced
    index*, so the caller slices the message itself. This hands the row over as
    an operand and the caller indexes nothing. Measured on a 25-block, rate-15
    absorb, both flat at one top-level eqn: the index form lowers to 39 lines
    with two gathers, this to 32 with the scan's own `dynamic_slice`. The
    difference is small and the contract is the point — a caller holding its
    blocks stacked writes no index arithmetic to get here.

    **Which loop form a marked caller may take is the emitter's question.** A
    generic region admits no control flow at all (`CLAUDE.md`), so a caller
    inside one must unroll. A name-routed region admits what its emitter's ABI
    says: `hash_frx.digest.field_sponge` carries a `while` for its runtime absorb
    length, and `sponge.py` ships exactly that. A loop *around* a marked region
    is always fine — the region moves into the loop body rather than being
    destroyed, and each permutation was already its own kernel, so there is no
    cross-iteration fusion to break.

    `reverse=True` walks the rows from the back, which is what a caller whose
    layout puts block 0 last wants — it costs two foldable scalar subtracts
    against a `stablehlo.reverse` over the whole operand.

    The byte sponges do not take this yet, and the reason is structural rather
    than a missing measurement: `keccak.sponge._absorb_squeeze` serves both the
    whole-hash `keccak_sponge` region and the generic path from one body, so
    moving it would mean splitting on `fusion_path` *and* restacking
    `[B, nb*rate]` into `[nb, B, rate]` for the block contract above. Where the
    crossover sits is tracked on #278.

    **Reached eagerly, this is only as cheap as `absorb` and `permute` are
    reusable.** `lax.scan` keys its trace cache on the body's *identity*, and the
    body is built from those two — so a body built per call misses that cache and
    re-traces an identical graph every time: 0.0215 s against 0.0001 s at eight
    blocks, which is worse than the unrolled loop this replaces. `_scan_step`
    memoizes it on the pair rather than leaving a caller to hoist a body it
    cannot see. Two consequences: the memo is by *equality*, so a bound method
    works where raw identity would not (`obj.permute is obj.permute` is False,
    while the two compare equal); and it is held for the life of the process, so
    a callback should close over parameters rather than message data — which is
    what taking `blocks` as an argument is for. Inside a `@jit` zone none of it
    matters, since the jit cache absorbs the trace.
    """
    final, _ = lax.scan(_scan_step(absorb, permute), state, blocks, reverse=reverse)
    return final


@lru_cache(maxsize=None)
def _scan_step(
    absorb: Callable[[S, Array], S], permute: Callable[[S], S]
) -> Callable[[S, Array], tuple[S, None]]:
    """`scanned_absorb`'s body, memoized so `lax.scan`'s trace cache can hit it.

    One absorb and the permutation the construction puts after it — the same
    pairing `absorb_squeeze` states, kept here rather than handed to the caller
    so a scanned caller cannot spell the schedule differently from an unrolled
    one.
    """

    def step(carry: S, block: Array) -> tuple[S, None]:
        return permute(absorb(carry, block)), None

    return step

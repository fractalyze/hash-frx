# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Incremental SHAKE — absorb and squeeze as fixed-shape pytrees.

A lattice scheme does not call SHAKE once on a fixed message. ML-DSA and Falcon
absorb a seed plus domain-separation bytes, then squeeze an unbounded number of
blocks until enough candidate coefficients survive rejection sampling. One-shot
`KeccakSponge.hash` cannot express that, so this keeps the state between calls.

**Absorbing and squeezing are two types, not one with a direction flag.** FIPS
202 section 4 permits no absorb after a squeeze has begun, and a type that cannot
express the illegal call is cheaper than one that has to reject it at runtime —
where the check would have to be traced, since the phase would live in the state.
`ShakeAbsorb.finalize()` is the direction switch, and it is the only way to reach
a `ShakeSqueeze`.

Both are registered pytrees with fixed shapes, so either threads a `@jit`
boundary or a `lax.scan` carry. `rate` rides as a `meta_field` (static aux) on
both halves, and `suffix` on the absorbing one, which is what keeps the treedef
stable across calls at one parameterisation and distinct between SHAKE128 and
SHAKE256.

**Every loop bound is static; only the counters are traced.** How many complete
blocks an absorb finishes depends on the pending length, which is traced — so
the block count cannot index anything. The schedule instead runs the fold to its
static upper bound and selects the answer, the pattern `sha256_stream_absorb`
established. One improvement over that: the lower candidate is a *prefix* of the
upper chain rather than a second chain, so both come out of one pass instead of
running the permutation twice.

Note that "no traced-index gather" constrains the *marked body*, not this
schedule: `permute` is the marked region and stays straight-line, while the
glue around it lowers to ordinary `gather`/`dynamic_slice` exactly as
`sha256_stream_absorb` does.

**A block-aligned caller need not pay for the offset at all.**
`ShakeBlockSqueeze`, reached from `ShakeAbsorb.finalize_blocks`, drops the field
rather than leaving it at zero: with no offset there is no `dynamic_slice` on
the output and no chain-select on the carry, which at width 50 is ~98 selects a
call. A rejection sampler is that caller, and its `lax.scan` carry is a squeezer,
so the field is not merely unused there — it is in the carry.

**The upper bound costs permutations, and that is the price of a traced
counter.** An absorb runs the fold `ceil((rate - 1 + L) / rate)` times where a
known pending length would need `(pending + L) // rate`, and a squeeze of `n`
emits `ceil((rate - 1 + n) / rate)` blocks where a known offset would need
`ceil(n / rate)` — so squeezing 32 bytes at a 136-byte rate permutes twice, not
once. Both are off by at most one block, and neither can be tightened while the
counter is traced: the alternative is a `while_loop` over a data-dependent bound,
which is the control-flow boundary the whole schedule exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import frx.numpy as fnp
from frx import Array, lax
from frx.tree_util import register_dataclass

from hash_frx.extension.sponge import squeeze_blocks
from hash_frx.keccak.byte_hashes import (
    SHAKE128_RATE,
    SHAKE256_RATE,
    SHAKE_SUFFIX,
)
from hash_frx.keccak.permutation import KeccakF1600
from hash_frx.keccak.sponge import _xor_into_rate, validate_sponge_params
from hash_frx.word import BYTES_PER_WORD, pack_le, unpack_le

_PERMUTE = KeccakF1600().permute
_STATE_WIDTH = KeccakF1600.width


def _absorb_block(state: Array, block_bytes: Array) -> Array:
    """XOR one rate block of bytes into the state and permute.

    `sponge._xor_into_rate` is indexed on the trailing axis, so the unbatched
    state this carries and the `[B, 50]` one the one-shot sponge absorbs into go
    through one spelling of the slice-and-concatenate."""
    return _PERMUTE(_xor_into_rate(state, pack_le(block_bytes)))


def _rate_bytes(state: Array, rate: int) -> Array:
    """The state's rate prefix as bytes — one squeeze block."""
    return unpack_le(state[: rate // BYTES_PER_WORD])


@partial(
    register_dataclass,
    data_fields=["state", "pending", "pending_len"],
    meta_fields=["rate", "suffix"],
)
@dataclass(frozen=True)
class ShakeAbsorb:
    """The absorbing half of an incremental SHAKE.

    The pending length is the only counter carried. `Sha256State` tracks the
    total absorbed as well, because Merkle-Damgård writes the message length
    into its padding — FIPS 202's `pad10*1` writes no length, so a sponge has
    nothing to read one with. A consumer that wants the count has it on the host
    for free, every absorb length being static.
    """

    state: Array  # uint32[50] — the Keccak state
    pending: Array  # uint8[rate] — trailing partial block, valid prefix [:pending_len]
    pending_len: Array  # int32, 0..rate-1
    rate: int
    suffix: int

    def absorb(self, data: Array) -> "ShakeAbsorb":
        """Absorb `data` (uint8 `[L]`, `L` static), returning the new state."""
        if data.ndim != 1:
            raise ValueError(f"data must be 1-D uint8 [L], got ndim={data.ndim}")
        # Coercing instead would hash a truncation of what the caller passed and
        # return a well-formed digest of the wrong message.
        if data.dtype != fnp.uint8:
            raise TypeError(f"data must be uint8, got {data.dtype}")
        rate = self.rate
        length = data.shape[0]
        pending_len = self.pending_len

        # Splice pending and data into one stream, then squeeze out the pending
        # buffer's invalid gap [pending_len:rate]: for stream position j the
        # source is j while j < pending_len, and j shifted past the gap after.
        source = fnp.concatenate([self.pending, data])
        max_blocks = (rate - 1 + length) // rate  # static upper bound
        slots = (max_blocks + 1) * rate
        pos = fnp.arange(slots, dtype=fnp.int32)
        idx = pos + fnp.where(pos < pending_len, fnp.int32(0), rate - pending_len)
        stream = source[fnp.clip(idx, 0, source.shape[0] - 1)]

        new_len = pending_len + fnp.int32(length)
        active = new_len // rate
        min_blocks = length // rate  # the other static candidate

        # One pass to the upper bound. `active` is `min_blocks` or `max_blocks`
        # and nothing else, and the `min_blocks` answer is the state partway
        # through this chain — so snapshot it rather than folding twice.
        # `extension.md.MdStream.absorb` runs the same schedule over a
        # compression function and reaches the same result a different way
        # (chain the prefix, then continue); the two are worth reading together.
        state = self.state
        lower = state
        for i in range(max_blocks):
            if i == min_blocks:
                lower = state
            state = _absorb_block(state, stream[i * rate : (i + 1) * rate])
        upper = state
        merged = (
            upper
            if min_blocks == max_blocks
            else fnp.where(active == max_blocks, upper, lower)
        )

        tail_len = new_len - active * rate
        tail = lax.dynamic_slice(stream, (active * rate,), (rate,))
        pending = fnp.where(
            fnp.arange(rate, dtype=fnp.int32) < tail_len, tail, fnp.uint8(0)
        )
        return ShakeAbsorb(merged, pending, tail_len, rate, self.suffix)

    def _padded_state(self) -> Array:
        """The state after FIPS 202's pad10*1 and the final absorb.

        The pad position is traced — it is the pending length — so `suffix` is
        placed by comparison rather than by index, and the `0x80` terminator is
        XOR-ed at the static last byte. When the message ends one byte short of a
        block the two land on the same byte and combine, which is FIPS 202's
        single-byte pad rather than a special case here.
        """
        rate, pending_len = self.rate, self.pending_len
        slot = fnp.arange(rate, dtype=fnp.int32)
        block = fnp.where(slot < pending_len, self.pending, fnp.uint8(0))
        block = fnp.where(slot == pending_len, fnp.uint8(self.suffix), block)
        block = block ^ fnp.where(
            slot == rate - 1, fnp.uint8(0x80), fnp.uint8(0)
        ).astype(fnp.uint8)
        return _absorb_block(self.state, block)

    def finalize(self) -> "ShakeSqueeze":
        """Pad, absorb the final block, and hand over to the squeezing half.

        Returns the general squeezer, which takes arbitrary byte counts. A
        caller that only ever takes whole rate blocks — a rejection sampler —
        wants `finalize_blocks` instead: the offset this one carries is what
        costs it a `dynamic_slice` and a select over all 50 lanes per call.
        """
        return ShakeSqueeze(self._padded_state(), fnp.int32(0), self.rate)

    def finalize_blocks(self) -> "ShakeBlockSqueeze":
        """`finalize`, handing back the whole-block squeezer.

        The narrow type is reachable only from here, and that is the point: its
        soundness is the offset being zero, which holds by construction after a
        pad-and-absorb and cannot be checked once the offset is traced.
        """
        return ShakeBlockSqueeze(self._padded_state(), self.rate)


@partial(
    register_dataclass,
    data_fields=["state", "offset"],
    meta_fields=["rate"],
)
@dataclass(frozen=True)
class ShakeSqueeze:
    """The squeezing half — `state`'s rate prefix is the live output block.

    `offset` is how much of that block a previous squeeze already consumed, so a
    run of squeezes returns the same bytes a single squeeze of the total would.

    The domain suffix does not ride along. `finalize` has already absorbed it,
    so nothing here could read it, and leaving it off lets a squeeze loop traced
    for one domain serve another at the same rate rather than re-tracing per
    domain.
    """

    state: Array  # uint32[50]
    offset: Array  # int32, 0..rate-1 — bytes taken from the live block
    rate: int

    def squeeze(self, nbytes: int) -> tuple["ShakeSqueeze", Array]:
        """Take `nbytes` (static) of output: `(new_state, uint8[nbytes])`."""
        if nbytes < 1:
            raise ValueError(f"nbytes ({nbytes}) must be >= 1")
        rate = self.rate

        # `offset + nbytes` spans two static block counts at most, so emit the
        # BYTES to the upper one and slice. The permutation count is smaller:
        # block i's bytes need i permutations (block 0 is the live prefix), and
        # the carried state can only be `chain[k]` for
        # k <= (offset + nbytes) // rate <= (rate - 1 + nbytes) // rate — so a
        # permute after the last block is reachable only when `rate - 1 +
        # nbytes` divides evenly (nbytes ≡ 1 mod rate). Emitting the ceil
        # unconditionally ran one dead permutation on every other length —
        # notably the whole-block squeeze a rejection sampler loops on — that
        # the traced select kept XLA from removing.
        upper = squeeze_blocks(rate - 1 + nbytes, rate)
        nperms = (rate - 1 + nbytes) // rate
        state = self.state
        chain = [state]
        blocks = []
        for i in range(upper):
            blocks.append(_rate_bytes(state, rate))
            if i < nperms:
                state = _PERMUTE(state)
                chain.append(state)

        stream = blocks[0] if len(blocks) == 1 else fnp.concatenate(blocks)
        out = lax.dynamic_slice(stream, (self.offset,), (nbytes,))

        end = self.offset + fnp.int32(nbytes)
        consumed = end // rate
        carried = chain[0]
        for k in range(1, len(chain)):
            carried = fnp.where(consumed == k, chain[k], carried)
        return ShakeSqueeze(carried, end % rate, rate), out


@partial(
    register_dataclass,
    data_fields=["state"],
    meta_fields=["rate"],
)
@dataclass(frozen=True)
class ShakeBlockSqueeze:
    """The squeezing half for a caller that only takes WHOLE rate blocks.

    Same schedule as `ShakeSqueeze` with the offset removed, and removing it is
    the whole point rather than a simplification. `ShakeSqueeze` carries how much
    of the live block a previous call consumed, and because that counter is
    traced it costs two things per call that a block-aligned caller pays for
    nothing:

    - the emitted stream runs to the upper block count and the output is
      `dynamic_slice`d out of it, so a whole-block squeeze extracts and
      concatenates two rate blocks to return one;
    - the carried state is selected out of a chain rather than being the last
      permute.

    Measured at rate 168, the schedule around the permutation shrinks from 41
    jaxpr equations to 16 for a one-block squeeze (95 to 65 at four), and the
    lowered module loses the `dynamic_slice` and four `select`s. It does NOT
    change the permutation count, which #218 already made minimal, and the
    lowered op count moves ~2% because the permutation dominates it — the win is
    in what a caller re-traces per shape and carries per iteration, not in the
    kernel.

    A rejection sampler is exactly the block-aligned caller: FIPS 204's
    `RejNTTPoly` takes one rate block at a time. Its `lax.scan` carry is a
    squeezer, so the offset is not merely unused there — it is in the carry, and
    the select is what maintains it.

    **Reachable only from `ShakeAbsorb.finalize_blocks`.** The type's soundness
    is that the offset is zero, which holds by construction after a pad-and-
    absorb and is uncheckable once traced — so there is no narrowing conversion
    from `ShakeSqueeze`, only `widen` in the other direction. That is the same
    move the absorb/squeeze split makes one level up: a type that cannot express
    the illegal call is cheaper than one that rejects it at runtime.
    """

    state: Array  # uint32[50]
    rate: int

    def squeeze_blocks(self, nblocks: int) -> tuple["ShakeBlockSqueeze", Array]:
        """Take `nblocks` (static) whole blocks: `(next, uint8[nblocks * rate])`.

        The unit is BLOCKS, unlike `ShakeSqueeze.squeeze`'s bytes, and the name
        says so — two squeezers whose methods differed only in what the integer
        meant would be a trap worth more than the brevity.

        `nblocks` blocks cost `nblocks` permutations: the state's rate prefix is
        block 0, and the permutation after the last block is what leaves the
        carry pointing at the next one.
        """
        if nblocks < 1:
            raise ValueError(f"nblocks ({nblocks}) must be >= 1")
        state = self.state
        blocks = []
        for _ in range(nblocks):
            blocks.append(_rate_bytes(state, self.rate))
            state = _PERMUTE(state)
        out = blocks[0] if nblocks == 1 else fnp.concatenate(blocks)
        return ShakeBlockSqueeze(state, self.rate), out

    def widen(self) -> ShakeSqueeze:
        """The same position as a general squeezer, for arbitrary byte counts.

        One direction only. This state is block-aligned so the offset it gains is
        zero; the reverse needs an offset that is traced, and a narrowing that
        cannot check its own precondition is one that pushes a silent wrong
        answer onto the caller.
        """
        return ShakeSqueeze(self.state, fnp.int32(0), self.rate)


def shake_init(rate: int, suffix: int = SHAKE_SUFFIX) -> ShakeAbsorb:
    """A fresh incremental sponge at `rate` bytes with `suffix` domain bits."""
    validate_sponge_params(rate, suffix)
    return ShakeAbsorb(
        state=fnp.zeros(_STATE_WIDTH, dtype=fnp.uint32),
        pending=fnp.zeros(rate, dtype=fnp.uint8),
        pending_len=fnp.int32(0),
        rate=rate,
        suffix=suffix,
    )


def shake128_init() -> ShakeAbsorb:
    """Incremental SHAKE128 — rate 168 B, capacity 256 bits."""
    return shake_init(SHAKE128_RATE)


def shake256_init() -> ShakeAbsorb:
    """Incremental SHAKE256 — rate 136 B, capacity 512 bits."""
    return shake_init(SHAKE256_RATE)

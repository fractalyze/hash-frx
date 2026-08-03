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
boundary or a `lax.scan` carry. `rate` and `suffix` ride as `meta_fields`
(static aux), which is what keeps the treedef stable across calls at one
parameterisation and distinct between SHAKE128 and SHAKE256.

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

from hash_frx.keccak.byte_hashes import (
    SHAKE128_RATE,
    SHAKE256_RATE,
    SHAKE_SUFFIX,
)
from hash_frx.keccak.permutation import KeccakF1600
from hash_frx.keccak.sponge import _BYTES_PER_ELEMENT, _pack_bytes, _unpack_bytes

_PERMUTE = KeccakF1600().permute
_STATE_WIDTH = KeccakF1600.width


def _blocks_of(nbytes: int, rate: int) -> int:
    """Blocks a `nbytes` run spans, rounded up."""
    return -(-nbytes // rate)


def _xor_block(state: Array, block: Array) -> Array:
    """XOR a packed rate block into the state prefix — the unbatched sibling of
    `sponge._xor_into_rate`, kept here because the streaming state is one row."""
    n = block.shape[0]
    return fnp.concatenate([state[:n] ^ block, state[n:]])


def _absorb_block(state: Array, block_bytes: Array) -> Array:
    """XOR one rate block of bytes into the state and permute."""
    packed = _pack_bytes(block_bytes.reshape(1, -1))[0]
    return _PERMUTE(_xor_block(state, packed))


def _rate_bytes(state: Array, rate: int) -> Array:
    """The state's rate prefix as bytes — one squeeze block."""
    return _unpack_bytes(state[: rate // _BYTES_PER_ELEMENT].reshape(1, -1))[0]


@partial(
    register_dataclass,
    data_fields=["state", "pending", "counts"],
    meta_fields=["rate", "suffix"],
)
@dataclass(frozen=True)
class ShakeAbsorb:
    """The absorbing half of an incremental SHAKE.

    `counts` packs both counters into one `int32[2]` leaf rather than carrying
    two scalar fields, for the reason `Sha256State` records: as separate leaves
    each costs its own scalar kernel per state hand-off, once per iteration
    inside a `lax.scan` carry.
    """

    state: Array  # uint32[50] — the Keccak state
    pending: Array  # uint8[rate] — trailing partial block, valid prefix [:counts[0]]
    counts: Array  # int32[2] = [pending_len (0..rate-1), total bytes absorbed]
    rate: int
    suffix: int

    @property
    def pending_len(self) -> Array:
        return self.counts[0]

    @property
    def total_len(self) -> Array:
        return self.counts[1]

    def absorb(self, data: Array) -> "ShakeAbsorb":
        """Absorb `data` (uint8 `[L]`, `L` static), returning the new state."""
        if data.ndim != 1:
            raise ValueError(f"data must be 1-D uint8 [L], got ndim={data.ndim}")
        rate = self.rate
        length = data.shape[0]
        pending_len = self.pending_len

        # Splice pending and data into one stream, then squeeze out the pending
        # buffer's invalid gap [pending_len:rate]: for stream position j the
        # source is j while j < pending_len, and j shifted past the gap after.
        source = fnp.concatenate([self.pending, data.astype(fnp.uint8)])
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
        counts = fnp.stack([tail_len, self.counts[1] + fnp.int32(length)])
        return ShakeAbsorb(merged, pending, counts, rate, self.suffix)

    def finalize(self) -> "ShakeSqueeze":
        """Pad, absorb the final block, and hand over to the squeezing half.

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
        return ShakeSqueeze(
            _absorb_block(self.state, block), fnp.int32(0), rate, self.suffix
        )


@partial(
    register_dataclass,
    data_fields=["state", "offset"],
    meta_fields=["rate", "suffix"],
)
@dataclass(frozen=True)
class ShakeSqueeze:
    """The squeezing half — `state`'s rate prefix is the live output block.

    `offset` is how much of that block a previous squeeze already consumed, so a
    run of squeezes returns the same bytes a single squeeze of the total would.
    """

    state: Array  # uint32[50]
    offset: Array  # int32, 0..rate-1 — bytes taken from the live block
    rate: int
    suffix: int

    def squeeze(self, nbytes: int) -> tuple["ShakeSqueeze", Array]:
        """Take `nbytes` (static) of output: `(new_state, uint8[nbytes])`."""
        if nbytes < 1:
            raise ValueError(f"nbytes ({nbytes}) must be >= 1")
        rate = self.rate

        # `offset + nbytes` spans two static block counts at most, so emit to the
        # upper one and slice. The chain is walked once and every prefix kept,
        # because the state to carry forward is one of those prefixes.
        upper = _blocks_of(rate - 1 + nbytes, rate)
        state = self.state
        chain = [state]
        blocks = []
        for _ in range(upper):
            blocks.append(_rate_bytes(state, rate))
            state = _PERMUTE(state)
            chain.append(state)

        stream = blocks[0] if len(blocks) == 1 else fnp.concatenate(blocks)
        out = lax.dynamic_slice(stream, (self.offset,), (nbytes,))

        end = self.offset + fnp.int32(nbytes)
        consumed = end // rate
        carried = chain[0]
        for k in range(1, len(chain)):
            carried = fnp.where(consumed == k, chain[k], carried)
        return ShakeSqueeze(carried, end % rate, rate, self.suffix), out


def shake_init(rate: int, suffix: int = SHAKE_SUFFIX) -> ShakeAbsorb:
    """A fresh incremental sponge at `rate` bytes with `suffix` domain bits."""
    return ShakeAbsorb(
        state=fnp.zeros(_STATE_WIDTH, dtype=fnp.uint32),
        pending=fnp.zeros(rate, dtype=fnp.uint8),
        counts=fnp.zeros(2, dtype=fnp.int32),
        rate=rate,
        suffix=suffix,
    )


def shake128_init() -> ShakeAbsorb:
    """Incremental SHAKE128 — rate 168 B, capacity 256 bits."""
    return shake_init(SHAKE128_RATE)


def shake256_init() -> ShakeAbsorb:
    """Incremental SHAKE256 — rate 136 B, capacity 512 bits."""
    return shake_init(SHAKE256_RATE)

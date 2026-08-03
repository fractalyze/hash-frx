# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Keccak sponge — XOR absorb, multi-rate padding, iterated squeeze.

FIPS 202 section 4 over `KeccakF1600`. One sponge parameterized by
`(rate, suffix, output_size)`; SHA3-256, SHAKE128 and SHAKE256 are three rows of
that table rather than three constructions (`byte_hashes.py`), and Keccak-256 is a
fourth row rather than a code change.

**Keccak-f[1600]-local, deliberately.** The permutation is bound here rather than
taken as a `Permutation`, because nothing else in the package needs a
byte-oriented sponge and the seam would buy generality with no second consumer.
Keccak-p[1600, 12] — TurboSHAKE, KangarooTwelve — would only need the round count
to vary, so widening this to take a permutation is the change to make when one of
them arrives, not before.

**Not `hash_frx.sponge.Sponge`.** That one absorbs by *overwriting* the rate
lanes, pads not at all, and squeezes by truncating the final state, and it takes
1-D field elements where this takes a `[B, L]` byte batch. The absorb/pad/squeeze
differences read like mode flags; the input domain does not, and it is the
decisive one — a merged form would carry a byte-packing layer used by one row and
a batch axis the other row lacks. `duplex_sponge.py` already made this call in
writing for the same reason.

Every loop bound here is static. `ByteHash` fixes the message length `L`, and the
output length is a parameter, so the block count and the squeeze count are known
at trace time: both are unrolled Python `for`s and the message is only ever an
operand. That is what lets `digest` take a tracer.

The state is `KeccakF1600`'s: 25 lanes as 50 `uint32` halves, lane `i` at
elements `2i` (low) and `2i+1` (high). A rate of `r` bytes is therefore the
contiguous prefix `state[: r // 4]`, and packing is a plain little-endian read of
four bytes per element — the halves-interleaved layout is chosen so that no lane
reordering is needed here (`params.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.keccak.permutation import KeccakF1600

U32 = fnp.uint32

# Elements of the uint32-halves state per byte of rate: 4 bytes per element.
_BYTES_PER_ELEMENT = 4

# A readability hoist, not a cache: the trace-once property belongs entirely to
# `permutation._permute_body`'s jit zone, which is keyed on the state aval and
# does not care where this wrapper lives.
_PERMUTE_BATCH = frx.vmap(KeccakF1600().permute)


@lru_cache(maxsize=None)
def _padding_tail(length: int, rate: int, suffix: int) -> np.ndarray:
    """What FIPS 202 section 5.1 appends to a `length`-byte message: uint8 [P].

    `suffix ‖ 0x00* ‖ 0x80`, where `suffix` carries the domain separation of
    section 6 (its low bits) and the `10*1` padding opens with the next bit up —
    which is why the two arrive as one byte rather than two steps.

    A function of the length alone, so it is a host constant built *from* the
    length rather than written *into* the message. That is what keeps `msg` an
    operand and lets it be traced, exactly as `sha256._padding_tail` does.

    When the message ends one byte short of a block the two ends land on the same
    byte and it becomes `suffix | 0x80`, which is the standard's single-byte pad
    rather than a special case.
    """
    nblocks = length // rate + 1
    tail = np.zeros(nblocks * rate - length, dtype=np.uint8)
    tail[0] = suffix
    tail[-1] |= 0x80
    return tail


def _pack_bytes(block: Array) -> Array:
    """uint8 [B, r] -> uint32 [B, r // 4] little-endian state elements.

    FIPS 202 section 3.1.2 maps a byte string onto lanes little-endian, and the
    halves layout keeps that a straight read: element `j` is bytes `4j .. 4j+3`,
    which is lane `j // 2`'s low half for even `j` and its high half for odd `j`.

    Named for bytes because `permutation.py` has its own `_pack`/`_unpack` pair
    in this package meaning something else — the flat state to the lane grid.
    """
    b = block.shape[0]
    w = block.reshape(b, -1, _BYTES_PER_ELEMENT).astype(U32)
    return (
        w[..., 0]
        | (w[..., 1] << U32(8))
        | (w[..., 2] << U32(16))
        | (w[..., 3] << U32(24))
    )


def _unpack_bytes(elements: Array) -> Array:
    """uint32 [B, n] -> uint8 [B, 4n] little-endian bytes — `_pack_bytes` inverted."""
    b = elements.shape[0]
    out = fnp.stack(
        [
            elements & U32(0xFF),
            (elements >> U32(8)) & U32(0xFF),
            (elements >> U32(16)) & U32(0xFF),
            (elements >> U32(24)) & U32(0xFF),
        ],
        axis=-1,
    ).astype(fnp.uint8)
    return out.reshape(b, -1)


def _xor_into_rate(state: Array, block: Array) -> Array:
    """XOR a packed block into the rate prefix, leaving the capacity untouched.

    Written as two static slices and a concatenate rather than `state.at[:n].set`:
    an in-place update lowers to a scatter, which is a fusion boundary and — on a
    GPU — one XLA serializes (`docs/reference/conventions.md`).
    """
    n = block.shape[-1]
    return fnp.concatenate([state[:, :n] ^ block, state[:, n:]], axis=-1)


@dataclass(frozen=True)
class KeccakSponge:
    """A FIPS 202 sponge over Keccak-f[1600], parameterized by its three knobs.

    `rate` is in bytes and must be a multiple of 4 so it lands on an element
    boundary of the halves state (every FIPS 202 rate is a multiple of 8).
    `suffix` is the domain-separation byte of section 6; `output_size` is how many
    bytes `hash` squeezes.

    Frozen over three ints, so the derived `__eq__`/`__hash__` are value-based —
    what the seam needs of anything riding as pytree aux, and safe here precisely
    because no field is an `Array`.
    """

    rate: int
    suffix: int
    output_size: int

    def __post_init__(self) -> None:
        if self.rate <= 0 or self.rate % _BYTES_PER_ELEMENT:
            raise ValueError(
                f"rate ({self.rate}) must be a positive multiple of "
                f"{_BYTES_PER_ELEMENT} bytes"
            )
        if self.rate >= KeccakF1600.width * _BYTES_PER_ELEMENT:
            raise ValueError(
                f"rate ({self.rate} bytes) leaves no capacity in a "
                f"{KeccakF1600.width * _BYTES_PER_ELEMENT}-byte state"
            )
        if not 0 <= self.suffix <= 0xFF:
            raise ValueError(f"suffix ({self.suffix:#x}) must be one byte")
        if self.output_size < 1:
            raise ValueError(f"output_size ({self.output_size}) must be >= 1")

    def hash(self, msg: ArrayLike) -> Array:
        """Absorb a batch of equal-length messages and squeeze: uint8 `[B, L]` ->
        uint8 `[B, output_size]`.

        `msg` may be a tracer: the padding is a host constant built from `L`, and
        every loop bound below is static, so nothing here reads a message byte.
        """
        message = fnp.asarray(msg, dtype=fnp.uint8)
        if message.ndim != 2:
            raise ValueError(f"msg must be 2-D uint8 [B, L], got ndim={message.ndim}")

        batch, length = message.shape
        tail = _padding_tail(length, self.rate, self.suffix)
        padded = fnp.concatenate(
            [message, fnp.broadcast_to(fnp.asarray(tail), (batch, tail.shape[0]))],
            axis=-1,
        )

        n = self.rate // _BYTES_PER_ELEMENT
        state = fnp.zeros((batch, KeccakF1600.width), dtype=U32)
        for start in range(0, padded.shape[-1], self.rate):
            block = _pack_bytes(padded[:, start : start + self.rate])
            state = _PERMUTE_BATCH(_xor_into_rate(state, block))

        # FIPS 202 section 4: emit the rate prefix, permute, repeat. The final
        # permutation of the absorb has already run, so the first block is
        # available before any further one. Collected as elements and unpacked
        # once rather than per block.
        squeezes = -(-self.output_size // self.rate)
        blocks = []
        for i in range(squeezes):
            blocks.append(state[:, :n])
            if i + 1 < squeezes:
                state = _PERMUTE_BATCH(state)
        return _unpack_bytes(fnp.concatenate(blocks, axis=-1))[:, : self.output_size]


# Deliberately NOT wrapped in a module-level jit zone the way
# `sha256.sha256_merkle_damgard` is. Measured both ways: a zone makes a repeated
# eager call 20-31x faster warm, but its static key is the whole
# `(rate, suffix, output_size, B, L)` shape, so it compiles per shape at ~0.5 s a
# time — around 300 same-shape eager calls before it pays for itself, against
# SHA-256's zone which is keyed on the block aval alone. The consumer that
# motivates this hash traces its own path, where `inline=True` would make the
# zone exactly neutral, and the batched device caller it is written for measured
# 1.3x at B=1024. So the trade lands the wrong way here and the sponge stays a
# plain body; `_permute_body` remains the one zone, shared by every block.

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Keccak sponge — XOR absorb, multi-rate padding, iterated squeeze.

FIPS 202 section 4 over `KeccakF1600`. One sponge parameterized by
`(rate, suffix, output_size)`; SHA3-256, SHAKE128 and SHAKE256 are three rows of
that table rather than three constructions (`byte_hashes.py`), and Keccak-256 is a
fourth row rather than a code change.

**Keccak-f[1600]-local, for now.** The permutation is bound here rather than
taken as a `Permutation`, because when this was written nothing else in the
package needed a byte-oriented sponge and the seam would have bought generality
with no second consumer. Ascon is that second consumer now
([`ascon/ascon.py`](../ascon/ascon.py) re-implements this schedule), so widening
this to take a permutation is open work rather than a deferral; Keccak-p[1600,
12] — TurboSHAKE, KangarooTwelve — would ride the same widening with only the
round count to vary.

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

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import device_message
from hash_frx.fusion import fused_region_over
from hash_frx.keccak.permutation import KeccakF1600
from hash_frx.word import BYTES_PER_WORD, pack_le, unpack_le

U32 = fnp.uint32

# The whole padded absorb + squeeze as one region a dedicated emitter expands.
# Not `hash_frx.digest.field_sponge`: that marker's `construction` attr switches between
# two sponges that share an input domain and an absorb, and this one shares
# neither (bytes rather than field elements, XOR rather than overwrite, an
# iterated squeeze rather than a truncation). Reusing it would re-merge what the
# module docstring keeps apart.
KECCAK_SPONGE_MARKER = "hash_frx.digest.keccak_sponge"
KECCAK_SPONGE_MARKER_VERSION = 1


def validate_sponge_params(rate: int, suffix: int) -> None:
    """Reject a rate or domain suffix the sponge cannot represent.

    Shared by the one-shot `KeccakSponge` and the incremental `shake_init`,
    which is not only de-duplication: the two spell the pad terminator
    differently (`|= 0x80` on the host tail here, `^ 0x80` on the traced block
    there), so a suffix with bit 7 set produced *different digests* from the
    two paths. That bit belongs to `pad10*1`'s trailing 1 — FIPS 202 section
    5.1 — and no standard suffix uses it (SHA-3 `0x06`, SHAKE `0x1F`, cSHAKE
    `0x04`, Keccak `0x01`), so it is rejected rather than defined.
    """
    if rate <= 0 or rate % BYTES_PER_WORD:
        raise ValueError(
            f"rate ({rate}) must be a positive multiple of " f"{BYTES_PER_WORD} bytes"
        )
    if rate >= KeccakF1600.width * BYTES_PER_WORD:
        raise ValueError(
            f"rate ({rate} bytes) leaves no capacity in a "
            f"{KeccakF1600.width * BYTES_PER_WORD}-byte state"
        )
    if not 0 <= suffix < 0x80:
        raise ValueError(
            f"suffix ({suffix:#x}) must be a byte below 0x80: bit 7 carries "
            "pad10*1's trailing 1 (FIPS 202 section 5.1)"
        )


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


def _xor_into_rate(state: Array, block: Array) -> Array:
    """XOR a packed block into the rate prefix, leaving the capacity untouched.

    Written as two static slices and a concatenate rather than `state.at[:n].set`:
    an in-place update lowers to a scatter, which is a fusion boundary and — on a
    GPU — one XLA serializes (`docs/reference/conventions.md`).
    """
    n = block.shape[-1]
    return fnp.concatenate([state[:, :n] ^ block, state[:, n:]], axis=-1)


def _absorb_squeeze(
    padded: Array,
    rate: int,
    output_size: int,
    permute: Callable[[Array], Array],
) -> Array:
    """FIPS 202 section 4 over an already-padded batch: XOR each rate block into
    the state and permute, then emit the rate prefix and permute again until
    `output_size` bytes are out.

    Takes `permute` batched over the leading axis rather than reaching for the
    permutation itself, so the same body serves the plain path and the marked
    one — where the permute is rebuilt from the region's ABI operands.

    The absorb's final permutation has already run when the squeeze starts, so
    the first block is available before any further one. Blocks are collected as
    elements and unpacked once rather than per block.
    """
    n = rate // BYTES_PER_WORD
    state = fnp.zeros((padded.shape[0], KeccakF1600.width), dtype=U32)
    for start in range(0, padded.shape[-1], rate):
        block = pack_le(padded[:, start : start + rate])
        state = permute(_xor_into_rate(state, block))

    squeezes = -(-output_size // rate)
    blocks = []
    for i in range(squeezes):
        blocks.append(state[:, :n])
        if i + 1 < squeezes:
            state = permute(state)
    return unpack_le(fnp.concatenate(blocks, axis=-1))[:, :output_size]


# A zone for the same reason `permutation._permute_body` is one, one level up:
# `lax.composite` re-traces its decomposition on every emission, and this
# marker's decomposition is the *whole* absorb and squeeze — every block's
# permute, each of which emits its own marker in turn. Without a zone an eager
# caller rebuilds all of it per call, which is 200-800x the generic path rather
# than the win the marker exists to be (#151). `inline=True` splices the cached
# jaxpr into the enclosing trace, so the emitted module is unchanged and a caller
# that traces its own path pays nothing for this.
#
# `perm` is a static key for the reason `_permute_body` gives — the marker it
# carries is not a function of its parameters, so a dedicated and a generic
# instance would otherwise collide here — and `rate` / `output_size` are the
# region's attributes and its loop bounds.
@partial(frx.jit, static_argnames=("perm", "rate", "output_size"), inline=True)
def _fused_hash(perm: KeccakF1600, padded: Array, rate: int, output_size: int) -> Array:
    """The padded absorb and the squeeze as ONE `hash_frx.digest.keccak_sponge` region
    over a dedicated-fusion permutation. Caller gates on
    `fusion_path.is_one_kernel`.

    The padding is applied before the region rather than inside it, so the
    emitter reads a rate-aligned byte matrix and never needs the domain suffix:
    `pad10*1` is a function of the message length alone, which is static, so it
    is a host constant either way and putting it inside would only widen the ABI.
    """

    def sponge(inp: Array, permute: Callable[[Array], Array]) -> Array:
        # The seam's `permute` is one state; this sponge carries a batch axis.
        return _absorb_squeeze(inp, rate, output_size, frx.vmap(permute))

    return fused_region_over(
        perm,
        padded,
        sponge,
        name=KECCAK_SPONGE_MARKER,
        version=KECCAK_SPONGE_MARKER_VERSION,
        rate=rate,
        output_size=output_size,
    )


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

    Keccak-bound rather than a generic byte sponge over any `Permutation`: a
    surface generalized from a single instance encodes that instance's
    accidents (the XOR absorb, the multi-rate padding split, the 4-byte lane
    packing) as if they were the family's, so the seam waited for a second
    implementation to shape it against. Ascon is that implementation, and
    `ascon.ascon_hash256_bytes` currently repeats this schedule rather than
    sharing it — the residue that stays family-specific is the pad rule, the
    initial state, and the state layout.
    """

    rate: int
    suffix: int
    output_size: int

    def __post_init__(self) -> None:
        validate_sponge_params(self.rate, self.suffix)
        if self.output_size < 1:
            raise ValueError(f"output_size ({self.output_size}) must be >= 1")

    def hash(self, msg: ArrayLike) -> Array:
        """Absorb a batch of equal-length messages and squeeze: uint8 `[B, L]` ->
        uint8 `[B, output_size]`.

        `msg` may be a tracer: the padding is a host constant built from `L`, and
        every loop bound below is static, so nothing here reads a message byte.

        A `DEDICATED`-path permutation lowers the whole padded absorb and
        squeeze to one `hash_frx.digest.keccak_sponge` region; otherwise each permute is
        its own marked region and the XOR glue between them stays outside.
        """
        message = device_message(msg)
        batch, length = message.shape
        tail = _padding_tail(length, self.rate, self.suffix)
        padded = fnp.concatenate(
            [message, fnp.broadcast_to(fnp.asarray(tail), (batch, tail.shape[0]))],
            axis=-1,
        )

        # Built per call rather than hoisted: the permutation reads the emitter
        # switch at construction, so a module-level instance would pin the
        # routing at import. The trace-once property belongs entirely to
        # `permutation._permute_body`'s jit zone, which does not care where the
        # instance comes from.
        perm = KeccakF1600()
        if perm.fusion_path.is_one_kernel:
            return _fused_hash(
                perm=perm,
                padded=padded,
                rate=self.rate,
                output_size=self.output_size,
            )
        return _absorb_squeeze(
            padded, self.rate, self.output_size, frx.vmap(perm.permute)
        )


# `hash` itself is deliberately NOT wrapped in a module-level jit zone the way
# `sha256.sha256_merkle_damgard` is, and the marked path above is, so the two
# decisions are worth separating.
#
# On the *generic* path a zone would make a repeated eager call 20-31x faster
# warm, but its static key is the whole `(rate, suffix, output_size, B, L)` shape,
# so it compiles per shape — around 300 same-shape eager calls before it pays for
# itself, against SHA-256's zone which is keyed on the block aval alone. The
# consumer that motivates this hash traces its own path, and the batched device
# caller it is written for measured 1.3x at B=1024. So the trade lands the wrong
# way there and that path stays a plain body over `_permute_body`'s zone.
#
# The *marked* path is not the same trade and was not covered by that reasoning.
# Its per-call cost is not the block loop, it is re-tracing one composite whose
# decomposition is the entire absorb and squeeze, which is three orders above the
# generic body rather than 20-31x below it. The same compile therefore pays back
# in about three calls instead of three hundred — the arithmetic did not change,
# the path it was applied to did — so the zone lives on `_fused_hash` and this
# note stays true of everything outside it.

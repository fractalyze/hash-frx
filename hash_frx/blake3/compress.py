# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The BLAKE3 compression function — 8-word CV and a 16-word block to 16 words.

The whole of BLAKE3's cryptography; everything above it is bookkeeping. Written
straight-line over `uint32` lanes and batched over a leading `[B, ...]` axis, so
a Merkle layer or a chunk's blocks drive it without a Python loop over the batch.

**The state is a `(4, 4)` grid, not sixteen words.** BLAKE3's round is four
column mixes and then four diagonal ones, and the four mixes within each half are
independent — so with the state held as four rows of four, one `G` call does all
four at once and the round is a handful of element-wise ops. Written as sixteen
scalars instead it is one HLO instruction per word per round, which is the shape
that made Keccak-f uncompilable (`docs/reference/conventions.md`, "Unrolled means
unrolled over rounds, not over lanes").

The diagonal half is the same `G` over rows rolled by 0/1/2/3 — the standard
ChaCha diagonalisation — so there is one mixing routine rather than two.

**The counter arrives as two `uint32` halves**, not one 64-bit word, for the
reason `keccak/lane.py` sets out at length: `uint64` is not safely available
here. It reaches the state as two words anyway (spec section 2.2 puts the low
half at index 12 and the high at 13), so the split costs nothing and the caller
never materialises a 64-bit value.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array

U32 = fnp.uint32

# Spec section 2.2, Table 1 — the SHA-2 IV.
# The IV is transcribed from the spec and its shape carries meaning — the two
# rows are the halves the feed-forward treats separately — so it is fenced from
# the formatter, which would flatten it one word per line.
# fmt: off
IV = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)
# fmt: on

# Domain-separation flags (spec section 2.2, Table 3). A caller ORs these into
# `flags`.
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3
KEYED_HASH = 1 << 4
DERIVE_KEY_CONTEXT = 1 << 5
DERIVE_KEY_MATERIAL = 1 << 6

# The message word schedule applied between rounds (spec section 2.2, Table 2).
MSG_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
ROUNDS = 7

# Where each round reads the message, composed on the host so every round slices
# the ORIGINAL block rather than a chain of six permuted copies. Round r's word i
# is `_SCHEDULE[r][i]` of the input.
_SCHEDULE: list[tuple[int, ...]] = [tuple(range(16))]
for _r in range(ROUNDS - 1):
    _SCHEDULE.append(tuple(_SCHEDULE[-1][i] for i in MSG_PERMUTATION))

# G's rotation amounts, in order (spec section 2.2).
_ROTATIONS = (16, 12, 8, 7)


def _rotr(x: Array, n: int) -> Array:
    """Rotate each `uint32` lane right by `n` (0 < n < 32, so no shift by 32)."""
    return (x >> U32(n)) | (x << U32(32 - n))


def _g(
    a: Array, b: Array, c: Array, d: Array, mx: Array, my: Array
) -> tuple[Array, Array, Array, Array]:
    """The G mixing function over four `[B, 4]` rows — four mixes at once."""
    n0, n1, n2, n3 = _ROTATIONS
    a = a + b + mx
    d = _rotr(d ^ a, n0)
    c = c + d
    b = _rotr(b ^ c, n1)
    a = a + b + my
    d = _rotr(d ^ a, n2)
    c = c + d
    b = _rotr(b ^ c, n3)
    return a, b, c, d


def _roll(x: Array, shift: int) -> Array:
    """`fnp.roll` along the column axis as the static slices it is — the wrapper
    carries an internal jit and would put a call in the body.

    `shift` is the diagonalisation's 1/2/3 and never a multiple of the row width,
    so the cut point is always interior.
    """
    cut = (-shift) % x.shape[1]
    return fnp.concatenate([x[:, cut:], x[:, :cut]], axis=1)


def _words(block: Array, indices: tuple[int, ...]) -> Array:
    """Message words at `indices` as one `[B, 4]` row — a static reorder."""
    return fnp.concatenate([block[:, i : i + 1] for i in indices], axis=1)


def _round(
    rows: tuple[Array, Array, Array, Array], block: Array, rnd: int
) -> tuple[Array, Array, Array, Array]:
    """One round: the column half, then the same G over rolled rows."""
    schedule = _SCHEDULE[rnd]
    r0, r1, r2, r3 = rows

    r0, r1, r2, r3 = _g(
        r0,
        r1,
        r2,
        r3,
        _words(block, schedule[0:8:2]),
        _words(block, schedule[1:8:2]),
    )

    # Diagonalise: mix (r0[k], r1[k+1], r2[k+2], r3[k+3]) by rolling each row into
    # a common column, running the column mix, and rolling back.
    r1, r2, r3 = _roll(r1, -1), _roll(r2, -2), _roll(r3, -3)
    r0, r1, r2, r3 = _g(
        r0,
        r1,
        r2,
        r3,
        _words(block, schedule[8:16:2]),
        _words(block, schedule[9:16:2]),
    )
    return r0, _roll(r1, 1), _roll(r2, 2), _roll(r3, 3)


def compress(
    chaining_value: Array,
    block: Array,
    counter: Array,
    block_len: Array,
    flags: Array,
) -> Array:
    """Compress one block per batch row: `[B, 16]` uint32 output.

    chaining_value : uint32 `[B, 8]`
    block          : uint32 `[B, 16]` — the message, little-endian words
    counter        : uint32 `[B, 2]` — the 64-bit chunk counter as (low, high)
    block_len      : uint32 `[B]`
    flags          : uint32 `[B]`

    The first eight output words are the new chaining value; a root node reads
    all sixteen as extendable output.
    """
    for name, arr, tail in (
        ("chaining_value", chaining_value, (8,)),
        ("block", block, (16,)),
        ("counter", counter, (2,)),  # (low, high)
        ("block_len", block_len, ()),
        ("flags", flags, ()),
    ):
        if arr.ndim == 0 or arr.shape[1:] != tail:
            raise ValueError(f"{name} must be [B, *{tail}], got {arr.shape}")
        # A narrower or signed operand does not error, it promotes: the result is
        # a correctly shaped uint32 of wrong words, since a signed `>>` is
        # arithmetic and a narrow lane never wraps at 32 bits. Raise rather than
        # coerce, so the caller's dtype bug surfaces here and not one layer out.
        if arr.dtype != U32:
            raise TypeError(f"{name} must be uint32, got {arr.dtype}")

    batch = chaining_value.shape[0]
    iv = fnp.broadcast_to(fnp.asarray(IV[:4], dtype=U32), (batch, 4))
    rows = (
        chaining_value[:, 0:4],
        chaining_value[:, 4:8],
        iv,
        fnp.stack([counter[:, 0], counter[:, 1], block_len, flags], axis=1),
    )

    for rnd in range(ROUNDS):  # static and small
        rows = _round(rows, block, rnd)

    # Feed-forward (spec section 2.2): the low half takes the high, and the high
    # half takes the input chaining value.
    r0, r1, r2, r3 = rows
    return fnp.concatenate(
        [
            r0 ^ r2,
            r1 ^ r3,
            r2 ^ chaining_value[:, 0:4],
            r3 ^ chaining_value[:, 4:8],
        ],
        axis=1,
    )

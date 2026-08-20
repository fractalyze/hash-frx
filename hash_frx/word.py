# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Word-level primitives the bit-oriented hashes share — rotate, roll, pack.

A bit-oriented hash computes in machine words rather than field elements, and
the handful of operations it needs below its own round function are the same
four everywhere: rotate a `uint32` lane, roll a row by a static amount, and read
bytes to words and back. `linear.py` is this module's counterpart for the field
families, where `unrolled_sum` plays the same role.

**The packers split on byte order, not on hash family.** FIPS 202 and BLAKE3
both read four bytes into a word little-endian, so they are the *same* function
and share `pack_le`; the SHA-2 family reads big-endian and shares `pack_be` the
same way (sha256 and sha512 are the two consumers). Reading that split as "each
hash owns its own packing" is what produces a fourth transcription of a shift
chain no reviewer can check by eye.

Each of these is a plain Python function inlined at trace time, so sharing them
costs nothing at the fusion contract: no `func.call` appears in a marked body,
which is the same reason `linear.unrolled_sum` is safe to share. Between them
they are also the reason a marked body needs no gather — `roll` is a neighbour
reference written as static slices, and `pack_le` is a reshape rather than an
indexed read (`docs/reference/conventions.md`).
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array

U32 = fnp.uint32

BYTES_PER_WORD = 4


def split(value: int) -> tuple[int, int]:
    """A 64-bit host integer as `(lo, hi)` 32-bit halves — exact, never narrowed.

    Where a 64-bit quantity that a lane-based hash needs — Keccak's round
    constants, BLAKE3's chunk counter — is taken apart. On the host, because
    `value` is a Python int there and no 64-bit value ever has to reach a device
    that may not have the width for it.
    """
    if not 0 <= value < (1 << 64):
        raise ValueError(f"{value:#x} does not fit in 64 bits")
    return value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF


def rotr(x: Array, n: int) -> Array:
    """Rotate each `uint32` lane right by `n`.

    `0 < n < 32`. A shift by 32 is undefined, and the single-expression form
    reaches it at `n = 0` — `keccak/lane.py` sets out the same hazard for the
    64-bit case, where it needs three branches rather than none.
    """
    return (x >> U32(n)) | (x << U32(32 - n))


def roll(x: Array, shift: int, axis: int = 0) -> Array:
    """`fnp.roll` along `axis` written as the two static slices it is.

    The wrapper carries an internal `jit`, so it lowers to a call inside the
    body and the single-kernel rewriter rejects it
    (`docs/reference/conventions.md`). The shift is a compile-time constant, so
    the cut point is known and this is a slice pair plus a concatenate.
    """
    n = x.shape[axis]
    cut = (-shift) % n
    if cut == 0:
        return x
    head: list[slice] = [slice(None)] * x.ndim
    tail: list[slice] = [slice(None)] * x.ndim
    head[axis] = slice(cut, None)
    tail[axis] = slice(None, cut)
    return fnp.concatenate([x[tuple(head)], x[tuple(tail)]], axis=axis)


def pack_le(data: Array) -> Array:
    """uint8 `[..., 4n]` -> uint32 `[..., n]`, four bytes per word little-endian.

    The trailing axis is the one packed, so a caller holding bytes already cut
    into blocks passes `[B, nblocks, 4n]` and gets its words in place.
    """
    if data.shape[-1] % BYTES_PER_WORD:
        raise ValueError(
            f"trailing axis must be a multiple of {BYTES_PER_WORD}, "
            f"got {data.shape[-1]}"
        )
    w = data.reshape(*data.shape[:-1], -1, BYTES_PER_WORD).astype(U32)
    return (
        w[..., 0]
        | (w[..., 1] << U32(8))
        | (w[..., 2] << U32(16))
        | (w[..., 3] << U32(24))
    )


def unpack_le(words: Array) -> Array:
    """uint32 `[..., n]` -> uint8 `[..., 4n]` — `pack_le` inverted."""
    out = fnp.stack(
        [
            words & U32(0xFF),
            (words >> U32(8)) & U32(0xFF),
            (words >> U32(16)) & U32(0xFF),
            (words >> U32(24)) & U32(0xFF),
        ],
        axis=-1,
    ).astype(fnp.uint8)
    return out.reshape(*words.shape[:-1], -1)


def pack_be(data: Array) -> Array:
    """uint8 `[..., 4n]` -> uint32 `[..., n]`, four bytes per word big-endian.

    `pack_le`'s other-byte-order twin, shared by the SHA-2 family the same way
    that one is shared by FIPS 202 and BLAKE3.
    """
    if data.shape[-1] % BYTES_PER_WORD:
        raise ValueError(
            f"trailing axis must be a multiple of {BYTES_PER_WORD}, "
            f"got {data.shape[-1]}"
        )
    w = data.reshape(*data.shape[:-1], -1, BYTES_PER_WORD).astype(U32)
    return (
        (w[..., 0] << U32(24))
        | (w[..., 1] << U32(16))
        | (w[..., 2] << U32(8))
        | w[..., 3]
    )


def unpack_be(words: Array) -> Array:
    """uint32 `[..., n]` -> uint8 `[..., 4n]` — `pack_be` inverted."""
    out = fnp.stack(
        [
            (words >> U32(24)) & U32(0xFF),
            (words >> U32(16)) & U32(0xFF),
            (words >> U32(8)) & U32(0xFF),
            words & U32(0xFF),
        ],
        axis=-1,
    ).astype(fnp.uint8)
    return out.reshape(*words.shape[:-1], -1)

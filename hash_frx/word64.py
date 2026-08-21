# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""64-bit arithmetic over `(lo, hi)` uint32 half pairs — the word-level layer
the 64-bit-word hashes share.

A SHA-512 or BLAKE2b word is 64 bits, and this toolchain cannot hold one
safely: with x64 off `uint64` truncates to `uint32`, and enabling x64 flips
the default dtypes process-wide (`keccak/lane.py` states the law). So a 64-bit
word rides as a pair of `uint32` halves — `(lo, hi)`, both halves the same
shape — and the operations a 64-bit ARX round needs beyond per-half bitwise
ops live here: rotate-right, add-with-carry, and the variadic XOR.

**The charter is `word.py`'s: only literally identical functions are shared.**
These helpers began module-local to `sha512` with exactly that bar written on
them — a second literal consumer would justify the lift — and BLAKE2b's G is
that consumer: the identical add/rotr/xor set at the rotation constants
(32, 24, 16, 63). Ascon's Σ layer carried a byte-identical rotate of its own
and adopts `rotr64` as a third reader. What is family-specific stays out:
`keccak/lane.py`'s `rotl` is the mirrored rotation direction (not the same
function), its `rho_tables` bulk machinery is Keccak's own, and sha512's σ
shift (`_shr64`) stays module-local on its single consumer.

The pair layout inside a packed state array is each module's convention, not
this one's: sha512 packs big-endian (high half at the even index), BLAKE2b
little-endian (low half even) — each produced by its standard's byte order
(`word.pack_be` / `pack_le`). A `Pair` here is already split into two arrays,
so nothing below depends on that choice. Splitting a host constant into
halves is `word.split`, on the host, where Python integers are exact.

The rotation is written as three static cases so no shift is ever by 32: a
`uint32` shift equal to the word width is undefined, and the single-expression
form reaches it whenever the in-half shift is zero (`keccak/lane.py` sets out
the hazard). Each helper is a plain Python function inlined at trace time, so
sharing them costs nothing at the fusion contract: no `func.call` appears in
a marked body, the same reason `word.py`'s packers are safe to share.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array

U32 = fnp.uint32

# A 64-bit value: (low 32 bits, high 32 bits), both halves the same shape.
Pair = tuple[Array, Array]


def rotr64(a: Pair, n: int) -> Pair:
    """Rotate the 64-bit value right by a single compile-time `n` — the
    rotate-right mirror of `keccak.lane.rotl`, three cases for the same
    reason: `n == 32` is a pure half swap (the case BLAKE2b's R1 lands on),
    and `n > 32` is that swap composed with the sub-32 case, which is why the
    swapped halves appear below."""
    n %= 64
    lo, hi = a
    if n == 0:
        return lo, hi
    if n == 32:
        return hi, lo
    if n < 32:
        s = U32(n)
        c = U32(32 - n)
        return (lo >> s) | (hi << c), (hi >> s) | (lo << c)
    m = n - 32
    s = U32(m)
    c = U32(32 - m)
    return (hi >> s) | (lo << c), (lo >> s) | (hi << c)


def add64(a: Pair, b: Pair) -> Pair:
    """64-bit addition mod 2^64: per-half uint32 adds with a comparison-based
    carry — `lo` wrapped iff it came out below an addend. The comparison is an
    ordinary element-wise op, and the digest markers this feeds are
    name-routed and exempt from the generic single-kernel whitelist regardless
    (`sha256.sha256_merkle_damgard` states the exemption)."""
    lo = a[0] + b[0]
    carry = (lo < a[0]).astype(U32)
    return lo, a[1] + b[1] + carry


def xor64(*terms: Pair) -> Pair:
    """XOR of two or more 64-bit values — variadic because the consumers'
    equations are two- and three-term (SHA-512's Σ/σ, BLAKE2b's feedforward),
    and the flat call reads as the standard's formula."""
    lo, hi = terms[0]
    for t in terms[1:]:
        lo, hi = lo ^ t[0], hi ^ t[1]
    return lo, hi

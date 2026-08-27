# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""64-bit Keccak lane arithmetic over `uint32` halves.

A Keccak-f[1600] lane is 64 bits, and this toolchain cannot hold one safely: with
x64 off `uint64` truncates to `uint32`, enabling x64 flips the default integer
and float dtypes to 64-bit process-wide, and under
`jax_explicit_x64_dtypes=ALLOW` a 64-bit literal silently loses its high half in
most spellings. A hash cannot be built on a representation that is wrong by
default, so a lane is a `(lo, hi)` pair of `uint32` — `lo` the low 32 bits.

Every operation here is straight-line and element-wise, which is what the
permutation's single-kernel rule requires. Rotation is the only step where the
halves interact, and it is written as three static cases so no shift is ever by
32 — a shift equal to the word width is undefined, and the "generic" spelling
`(x << n) | (x >> (32 - n))` hits exactly that at `n = 0`.

Splitting a constant happens on the host, where Python integers are exact
(`word.split`), never by materialising a 64-bit value and narrowing it.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array, lax

# A lane: (low 32 bits, high 32 bits). Both halves carry the same shape, so the
# whole 5x5 lane grid rides as one `Lane` of `(5, 5)` arrays.
Lane = tuple[Array, Array]


def rotl(a: Lane, n: int) -> Lane:
    """Rotate the 64-bit lane left by a single compile-time `n`.

    Three cases rather than one expression, because a `uint32` shift by 32 is
    undefined and the single-expression form reaches it whenever the in-half
    shift is zero. `n == 32` is a pure half swap, and `n > 32` is that swap
    composed with the sub-32 case, which is why the swapped halves appear below.
    """
    n %= 64
    lo, hi = a
    if n == 0:
        return lo, hi
    if n == 32:
        return hi, lo
    if n < 32:
        s = fnp.uint32(n)
        c = fnp.uint32(32 - n)
        return (lo << s) | (hi >> c), (hi << s) | (lo >> c)
    m = n - 32
    s = fnp.uint32(m)
    c = fnp.uint32(32 - m)
    return (hi << s) | (lo >> c), (lo << s) | (hi >> c)


# rho's per-lane case analysis, precomputed off the offset grid: `swap` folds
# the `n >= 32` case into a half exchange, `shift`/`counter_shift` are the
# in-half shift pair, and `keep` marks the lanes whose in-half shift is zero.
RhoTables = tuple[Array, Array, Array, Array]


def rho_tables(offsets: Array) -> RhoTables:
    """The per-lane rotation tables rho applies, derived from its offset grid.

    Derived on device from `offsets` rather than folded on the host, because the
    offsets are a marked region's operand: a host-folded table is an array the
    decomposition materialises, which `lax.composite` lifts into an unnamed
    operand ahead of the declared ones — one copy per round, and the ABI a
    dedicated emitter reads is no longer the one written down. Element-wise over
    a 5x5 grid and shared by all 24 rounds, so deriving it costs a handful of ops
    once.

    `m == 0` would shift by 32, which is undefined for a `uint32`, so those lanes
    take a dummy shift of 1 and are selected away by `keep`.

    The two reductions modulo a lane width are written as masks: `%` is an `fnp`
    wrapper that lowers to a call inside the body, which the single-kernel
    rewriter rejects, and both moduli are powers of two.
    """
    n = offsets & fnp.uint32(63)  # offset mod 64, the lane width
    m = n & fnp.uint32(31)  # in-half shift, mod the 32-bit half
    keep = m == fnp.uint32(0)
    m_safe = lax.select(keep, lax.full_like(m, 1), m)
    return n >= fnp.uint32(32), m_safe, fnp.uint32(32) - m_safe, keep


def rotl_each(a: Lane, tables: RhoTables) -> Lane:
    """Rotate every lane of an array-shaped `Lane` by its own offset.

    rho gives each of the 25 lanes a different rotation, and doing that lane by
    lane is what makes the graph explode: one scalar op per lane per round turns
    24 rounds into a quarter of a million HLO instructions. `rho_tables` turns
    the per-lane case analysis of `rotl` into grids instead of Python branches,
    so the whole step is a handful of element-wise ops on the lane grid.

    The selects survive into the graph — four per round. XLA folds a `select`
    only on a scalar constant predicate or when both arms agree, never on an
    element-varying mask like this one, so the tables being derived from
    constants does not make them free. They are kept because the alternatives are
    worse: removing the `swap` pair costs more shift/or work than it saves, and
    the `keep` pair can only go by making a shift-by-32 unreachable another way.

    `lax.select` rather than `fnp.where`: the wrapper carries an internal `jit`
    and lowers to a call inside the body, which the single-kernel rewriter
    rejects (`docs/reference/conventions.md`).
    """
    swap, s, c, keep = tables
    lo, hi = a
    base_lo = lax.select(swap, hi, lo)
    base_hi = lax.select(swap, lo, hi)

    rot_lo = (base_lo << s) | (base_hi >> c)
    rot_hi = (base_hi << s) | (base_lo >> c)
    return lax.select(keep, base_lo, rot_lo), lax.select(keep, base_hi, rot_hi)

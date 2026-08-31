# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Shared normal-form linear helpers — explicit field add/mul, no
dot/reduce/gather.

Every field-permutation linear layer is written as a fixed, unrolled sum of
column-scaled lanes so a round body stays straight-line element-wise and fuses to
one kernel: `fnp.dot`/`fnp.sum` lower to a reduction (the `kInput` fusion
boundary) and dynamic indexing to `gather`, either of which splits the kernel.

A layer must also keep the number of times it reads its **chained** input — the
state, or the scalar a partial round threads through every lane — from scaling
with the width. Per-lane scalar expressions each re-derive that value, so where
rounds chain, the cost compounds per round rather than adding: measured ~3.6x per
round on the CPU backend, enough that a 22-round Goldilocks permutation never
finished (https://github.com/fractalyze/zorch/issues/565). Read the input once as
an array, or hoist its lanes once and index the hoisted list.

`unrolled_sum` is that summation primitive; `apply_matrix` is the dense
`matrix @ state` built on it for a **field-array** matrix. The scheme-specific
layers (`apply_external_m4`, `apply_internal`, `apply_sparse_partial`,
Vision's `apply_linearized_affine`) live in the per-permutation `linear.py`
and build on `unrolled_sum` from here.
"""

from __future__ import annotations

from frx import Array


def unrolled_sum(terms: list[Array]) -> Array:
    """Sum `terms` as a balanced tree of binary field adds: `(t0 + t1) + (t2 + t3)`.

    Deliberately not `fnp.sum`/`sum()`: those lower to a reduction op — the
    `kInput` fusion boundary that would split the round body's kernel. The tree
    stays a chain of straight-line element-wise adds, so the whole layer fuses.

    Paired rather than folded left. Both spell the same number of adds, but a
    left fold makes each one wait for the last, so the layer's critical path is
    one add per term where pairing makes it one per doubling — and a wide state
    is summed once per round. It costs nothing to prefer: modular addition is
    associative, so the two spellings agree bit for bit.
    """
    if not terms:
        raise ValueError("unrolled_sum needs at least one term")
    level = list(terms)
    while len(level) > 1:
        paired = [level[i] + level[i + 1] for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            paired.append(level[-1])
        level = paired
    return level[0]


def apply_matrix(matrix: Array, state: Array) -> Array:
    """Dense `matrix @ state`, as the sum of each column scaled by its lane —
    `out[i] = sum_j matrix[i][j] * state[j]`.

    Written as an unrolled `unrolled_sum` rather than `matrix @ state`/`fnp.dot`
    on purpose: the matmul/dot lowers to a reduction (the `kInput` fusion
    boundary) that splits the round body's kernel; the column-scaled sum stays
    element-wise and fuses. Takes the matrix as a **field array**; poseidon2's
    `apply_external_m4` is the one integer-literal sibling (its M4 is a small
    structural matrix, not field data).
    """
    if state.ndim != 1:
        raise ValueError(f"state must be 1-D, got shape {state.shape}")
    w = state.shape[0]
    if matrix.shape != (w, w):
        raise ValueError(
            f"need a square matrix matching 1-D state, got matrix {matrix.shape}, "
            f"state {state.shape}"
        )
    return unrolled_sum([matrix[:, j] * state[j] for j in range(w)])

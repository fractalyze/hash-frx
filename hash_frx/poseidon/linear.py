"""Normal-form Poseidon linear layers — explicit field add/mul, no dot/reduce/gather.

Every linear layer is a fixed, unrolled sum of column-scaled lanes so a round
body stays straight-line element-wise and fuses to one kernel: `fnp.dot`/`fnp.sum`
lower to a reduction (the `kInput` fusion boundary) and dynamic indexing to
`gather`, either of which splits the kernel. The summation primitive
`unrolled_sum` and the dense layer `apply_matrix` are shared with poseidon2 in
`hash_frx.linear`; this module adds the Poseidon-specific sparse form.

Every matrix rides as a field array. Inside a marked region a closed-over
matrix stays a constant of the decomposition body — frx keeps composite consts
inline rather than lifting them to operands — so the same form serves the
generic `zorch.fused_region` marker and the name-routed dedicated markers,
whose structure rides separately as int64 marker attributes.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array

from hash_frx.linear import unrolled_sum


def apply_sparse_partial(
    dot_row: Array,
    col_vec: Array,
    active: Array,
    tail: Array,
) -> Array:
    """The optimized-sparse partial round's linear layer, in normal form.

    With `a` the post-S-box lane-0 value (`active`) and `tail = state[1:]`:

        out[0]   = a*dot_row[0] + sum_{j>=1} tail[j-1]*dot_row[j]   (dense lane-0 row)
        out[t]   = tail[t-1] + a*col_vec[t-1]     for t = 1 .. width-1

    i.e. lane 0 gathers a full dot over the state while lanes 1.. only add a
    rank-1 correction from lane 0. `dot_row` (width) and `col_vec` (width-1) are
    field-array constants; the unrolled sum keeps the layer reduction- and
    gather-free. Uses `concatenate`, not `.at[0].set`, so no scatter splits the
    kernel.
    """
    if dot_row.ndim != 1 or col_vec.ndim != 1:
        raise ValueError(
            f"dot_row and col_vec must be 1-D, got {dot_row.shape}, {col_vec.shape}"
        )
    w = dot_row.shape[0]
    if col_vec.shape[0] != w - 1:
        raise ValueError(
            f"col_vec must have width-1 entries, got {col_vec.shape[0]} for width {w}"
        )
    if active.ndim != 0:
        raise ValueError(
            f"active (post-S-box lane 0) must be a scalar, got shape {active.shape}"
        )
    if tail.shape != (w - 1,):
        raise ValueError(
            f"tail (state[1:]) must have width-1 entries, got {tail.shape} for "
            f"width {w}"
        )
    # Array-shaped so `active` is read twice rather than once per lane — the
    # chained-input rule in `hash_frx.linear`.
    prods = dot_row[1:] * tail
    out0 = unrolled_sum([dot_row[0] * active, *prods])
    out_rest = tail + col_vec * active
    return fnp.concatenate([out0[None], out_rest])

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Normal-form Vision linear layers — explicit field add/mul, no dot/reduce/gather.

Vision's linear vocabulary is the dense MDS multiply — `apply_matrix`, shared
from `hash_frx.linear` — plus the F2-linearized affine polynomial
`B(x) = b_const + sum_j c_j * x**(2**j)` that finishes each S-box
(https://eprint.iacr.org/2019/426, Section 4.1). The polynomial is element-wise
over the state, so one evaluation serves all lanes at once: the `x**(2**j)`
powers are a chain of explicit squarings (each one field multiply) and the
coefficient terms fold through `unrolled_sum`, keeping the layer a fixed,
straight-line sequence that fuses to one kernel — `fnp.power` with a traced
exponent, `fnp.dot`, or `fnp.sum` would each put a call, a dot, or a reduction
(the `kInput` fusion boundary) in the body.

The chained-input rule in `hash_frx.linear` holds here by construction: the
state is read exactly once (the base of the squaring chain); every later term
reads the previous power, not the state.
"""

from __future__ import annotations

from frx import Array

from hash_frx.linear import apply_matrix, unrolled_sum

__all__ = ["apply_matrix", "apply_linearized_affine"]


def apply_linearized_affine(coeffs: Array, state: Array) -> Array:
    """Evaluate the F2-linearized affine polynomial `coeffs` on every element.

    `coeffs` is constant-first, the `VisionParams` layout: `coeffs[0]` the
    affine constant, `coeffs[1 + j]` the coefficient of `x**(2**j)`. Works for
    any length >= 2, which covers both of Vision's polynomials — the degree-4
    `B` (4 entries) and its dense inverse `B^{-1}` (n + 1 entries).

    Scalar-indexing `coeffs` is a static `slice`, not a gather; the constant
    term broadcasts over the state. Squaring is spelled `power * power` — one
    element-wise field multiply — so the whole evaluation stays inside the
    fusion whitelist.
    """
    if coeffs.ndim != 1 or coeffs.shape[0] < 2:
        raise ValueError(
            f"coeffs must be a 1-D constant-plus-coefficients vector of length "
            f">= 2, got shape {coeffs.shape}"
        )
    n_linear = coeffs.shape[0] - 1
    power = state
    terms = [coeffs[1] * power]
    for j in range(1, n_linear):
        power = power * power
        terms.append(coeffs[1 + j] * power)
    return unrolled_sum(terms) + coeffs[0]

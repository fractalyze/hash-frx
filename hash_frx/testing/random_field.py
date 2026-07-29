# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Deterministic random field elements for tests.

Draws canonical integers in [0, 2**30) (< every supported prime) and casts to the
field dtype, which Montgomery-encodes the canonical integer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array


def _is_extension_dtype(dtype: Any) -> bool:
    """True for an algebraic extension field. `zk_dtypes.efinfo` is the extension
    registry: it resolves extension metadata and raises for a base field."""
    try:
        zk_dtypes.efinfo(dtype)
        return True
    except ValueError:
        return False


def rand_field(seed: int, shape: Sequence[int], dtype: Any) -> Array:
    # An extension element is k>1 base limbs, so casting a scalar canonical
    # integer embeds it into the base subfield (the k-1 higher coordinates stay
    # zero) — a degenerate draw, not a generic extension element. Forbid it so the
    # footgun is caught at the call site; a generic element needs its limbs drawn
    # in the base field and bitcast to the extension.
    if _is_extension_dtype(dtype):
        raise ValueError(
            f"{np.dtype(dtype).name} is an extension field; rand_field would embed "
            f"a canonical integer into the base subfield (higher coordinates zero). "
            f"Draw (*shape, k) base elements and bitcast instead."
        )
    ints = np.random.default_rng(seed).integers(0, 1 << 30, size=shape, dtype=np.int64)
    return fnp.array(ints, dtype=dtype)

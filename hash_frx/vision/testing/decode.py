# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Canonical-integer decode for field-dtype test arrays.

The `astype(object)` is the load-bearing step: the numpy object cast routes
each element through the dtype's own Python conversion, which yields the
canonical integer (decoding any internal representation, with no frx x64
mode needed). Every device -> oracle handoff in this package goes through
these two helpers so that subtlety is spelled once.
"""

from __future__ import annotations

import numpy as np


def ints(arr: object) -> list[int]:
    """1-D field array -> its canonical Python ints."""
    return [int(v) for v in np.asarray(arr).astype(object)]


def int_rows(arr: object) -> list[list[int]]:
    """2-D field array -> its canonical Python ints, row by row."""
    return [[int(v) for v in row] for row in np.asarray(arr).astype(object)]

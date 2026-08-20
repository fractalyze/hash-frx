# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""HKDF (RFC 5869) — extract-then-expand over `Hmac`.

The KDF every composition standard in reach names (HPKE's KDF slot, the
TLS-style key schedules) is HKDF, and RFC 5869 defines it over HMAC
specifically — every published vector is HMAC-based — so this module takes an
`Hmac` rather than a pluggable PRF. Two functions, matching the RFC's own
surface: consumers like HPKE call `extract` and `expand` separately
(LabeledExtract / LabeledExpand), so no combined convenience wrapper is
offered.

Batch-parallel like the construction it consumes: the batch axis rides `ikm` /
`prk` (and optionally `salt` / `info`, which otherwise broadcast), and every
length is static, so the `T(i)` chain unrolls at trace time with no
data-dependent shape anywhere.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.hmac import Hmac


def hkdf_extract(
    mac: Hmac, salt: ArrayLike | None, ikm: ArrayLike
) -> Array | np.ndarray:
    """`PRK = HMAC(salt, ikm)` — RFC 5869 §2.2.

    ikm : uint8 `[B, I]` input keying material.
    salt : uint8 `[S]` (shared) or `[B, S]` (per entry); `None` takes the
        RFC's stated default, a digest-length string of zeros.
    Returns uint8 `[B, digest_size]`.
    """
    if salt is None:  # RFC 5869 §2.2: "if not provided, ... HashLen zeros"
        salt = fnp.zeros(mac.digest_size, dtype=fnp.uint8)
    return mac.mac(salt, ikm)


def hkdf_expand(
    mac: Hmac, prk: ArrayLike, info: ArrayLike | None, length: int
) -> Array | np.ndarray:
    """`OKM = first `length` bytes of T(1) ‖ T(2) ‖ …` — RFC 5869 §2.3, with
    `T(i) = HMAC(prk, T(i-1) ‖ info ‖ i)`.

    prk : uint8 `[digest_size]` or `[B, digest_size]` (a shared PRK
        broadcasts over a batched `info`).
    info : uint8 `[I]` (shared) or `[B, I]` (per entry); `None` is the RFC's
        default empty string.
    length : bytes of output, at most `255 * digest_size` (§2.3).
    Returns uint8 `[B, length]`.

    `length` and every input width are static, so `n = ceil(length /
    digest_size)` unrolls as a Python loop — each `T(i)` is one `mac` call.
    """
    if not 1 <= length <= 255 * mac.digest_size:
        raise ValueError(
            f"length ({length}) must be in [1, 255 * digest_size = "
            f"{255 * mac.digest_size}] (RFC 5869 §2.3)"
        )
    prk = fnp.asarray(prk, dtype=fnp.uint8)
    if info is None:  # RFC 5869 §2.3: "optional ... (can be zero-length)"
        info = fnp.zeros(0, dtype=fnp.uint8)
    info = fnp.asarray(info, dtype=fnp.uint8)
    if prk.ndim == 2:
        batch = prk.shape[0]
    elif info.ndim == 2:
        batch = info.shape[0]
    else:
        batch = 1
    if info.ndim == 1:
        info = fnp.broadcast_to(info, (batch, info.shape[0]))

    n = -(-length // mac.digest_size)
    t = fnp.zeros((batch, 0), dtype=fnp.uint8)  # T(0) is the empty string
    blocks = []
    for i in range(1, n + 1):
        ctr = fnp.full((batch, 1), i, dtype=fnp.uint8)
        t = fnp.asarray(
            mac.mac(prk, fnp.concatenate([t, info, ctr], axis=1)), dtype=fnp.uint8
        )
        blocks.append(t)
    return fnp.concatenate(blocks, axis=1)[:, :length]

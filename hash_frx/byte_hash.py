# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The ByteHash seam — the byte sibling of `Permutation`.

A byte hash maps a batch of equal-length byte messages to fixed-size digests:
`digest(uint8[B, L]) -> uint8[B, digest_size]`, byte-identical to the hash's
standard (SHA-256 = FIPS 180-4). Consumers — a byte Fiat-Shamir transcript, byte
Merkle leaves, proof-of-work grinding — read `digest_size` and call `digest`;
they never name a concrete hash. `Sha256` is one implementation; any other byte
hash drops into the same seam unchanged, its internal construction hidden behind
`digest`: SHA-256 is Merkle-Damgard, BLAKE3 a Merkle tree, Keccak a sponge — the
seam abstracts over all three because `digest` is the only common surface. A
shared *streaming* interface would not generalize: the midstate shape differs per
construction.

This is the byte counterpart of `permutation.Permutation`: where a `Permutation`
backs the algebraic sponge / compression / duplex transcript over a field dtype,
a `ByteHash` backs byte hashing over raw bytes. The two `has_dedicated_fusion`
flags mean the same thing: whether the primitive lowers to a hash-dedicated
fusion marker (vs a host-eager path), so a consumer can gate device-fusion
wrapping without naming a concrete hash.

Implementations define value-based `__eq__`/`__hash__` over their full parameter
surface — the same rule `Permutation` carries. A host byte transcript is a
`bytes` buffer rather than a jit-traced pytree, so it does not depend on this;
but the moment a `ByteHash` is carried as pytree aux (e.g. a byte Merkle threaded
through `@jit`), identity equality would silently re-trace the enclosing zone on
every freshly built instance. Defining it is cheap and keeps the seam
re-trace-safe by construction (a param-free hash compares by type).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from frx import Array
from frx.typing import ArrayLike


@runtime_checkable
class ByteHash(Protocol):
    digest_size: int
    # Whether `digest` lowers to a hash-dedicated fusion marker (vs a host-eager
    # path). Mirrors `Permutation.has_dedicated_fusion`: a consumer gates
    # device-fusion wrapping on it without naming a concrete hash.
    has_dedicated_fusion: bool

    def digest(self, msg: ArrayLike) -> Array | np.ndarray:
        """Hash a batch of equal-length messages: uint8 `[B, L]` -> uint8
        `[B, digest_size]`, big-endian (the hash's standard output order). The
        result is a device `Array` (a marker hash) or a host `np.ndarray` (a
        host-library hash); consumers `np.asarray` it to bytes.

        One call is one function — the unit that lowers to one fused kernel when
        `has_dedicated_fusion`. `L` is static, so any padding is data-independent.
        Batch with the `B` axis: a dedicated-fusion hash lowers the whole batch
        through one shared decomposition (Merkle leaves, a PoW nonce window).
        """
        ...

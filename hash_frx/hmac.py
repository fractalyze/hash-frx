# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""HMAC (FIPS 198-1) over the `ByteHash` seam.

HMAC turns a byte hash into a keyed MAC:
`HMAC(K, m) = H((K0 ^ opad) ‖ H((K0 ^ ipad) ‖ m))`, with `K0` the key brought
to the hash's block size. FIPS 198-1 fixes the two pad bytes and the key
processing and leaves the hash and its block size `B` as the parameterization —
a generator over profiles rather than a list of named members — so, like
`Poseidon2(params)`, this is one value-compared class rather than a type per
profile: `Hmac(Sha256(), 64)` is HMAC-SHA-256.

`block_size` is a parameter of the *construction*, deliberately not added to
the `ByteHash` Protocol: the seam carries the intersection of byte hashes
(`digest` alone), and a block size only means something to block-keyed
constructions — BLAKE3, whose keyed mode is native, has no block size for HMAC
to read. The same line keeps `DuplexSponge`'s `+`-merge on the construction
rather than on `Permutation`.

Batch-parallel like the seam it consumes: `mac(key, msg)` takes uint8 `[B, L]`
messages with a shared `[K]` or per-message `[B, K]` key and returns uint8
`[B, digest_size]`. `K` and `L` are static, so the longer-than-block key
hash-down resolves at trace time and nothing branches on data. The construction
adds no marker of its own — it lowers to exactly its `digest` calls (two, plus
one more when the key hashes down), so the fusion story is the underlying
hash's; a dedicated HMAC marker would be a measured optimization, not a
correctness property.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import ByteHash

# The FIPS 198-1 §4 inner / outer pad bytes, repeated to the block. Python ints
# here — wrapped per use — so importing this module puts nothing on a backend.
_IPAD = 0x36
_OPAD = 0x5C


class Hmac:
    """HMAC over a `ByteHash` — FIPS 198-1, byte-identical to the standard.

    byte_hash : the underlying byte hash `H` (any `ByteHash`; a host row works
        eagerly, a device row also inside a consumer's `@jit`).
    block_size : `H`'s input block size in bytes — FIPS 198-1's `B`
        (SHA-256: 64). Must be at least `H`'s digest size, because §4 replaces
        a longer-than-block key by its digest and zero-pads the result to one
        block.
    """

    def __init__(self, byte_hash: ByteHash, block_size: int) -> None:
        if block_size < byte_hash.digest_size:
            raise ValueError(
                f"block_size ({block_size}) must be >= the hash's digest_size "
                f"({byte_hash.digest_size}): FIPS 198-1 §4 replaces a "
                "longer-than-block key by its digest, which must fit one block"
            )
        self.byte_hash = byte_hash
        self.block_size = block_size
        self.digest_size = byte_hash.digest_size

    def mac(self, key: ArrayLike, msg: ArrayLike) -> Array | np.ndarray:
        """`HMAC(key, msg)` per message: msg uint8 `[B, L]`, key uint8 `[K]`
        (shared by the batch) or `[B, K]` (per message) -> uint8
        `[B, digest_size]`, the underlying hash's output order.

        `K` is static, so which arm of the §4 key processing runs — hash down
        when longer than the block, then zero-pad to it — is decided at trace
        time; `msg` and `key` may both be tracers when the hash is a device
        row.
        """
        key = fnp.asarray(key, dtype=fnp.uint8)
        msg = fnp.asarray(msg, dtype=fnp.uint8)
        if msg.ndim != 2:
            raise ValueError(f"msg must be uint8 [B, L], got shape {msg.shape}")
        if key.ndim == 1:
            key = key[None, :]
        if key.ndim != 2:
            raise ValueError(f"key must be [K] or [B, K], got shape {key.shape}")
        # FIPS 198-1 §4: K0 = H(K) when K is longer than the block, then
        # zero-padded to exactly one block.
        if key.shape[1] > self.block_size:
            key = fnp.asarray(self.byte_hash.digest(key), dtype=fnp.uint8)
        if key.shape[1] < self.block_size:
            pad = fnp.zeros(
                (key.shape[0], self.block_size - key.shape[1]), dtype=fnp.uint8
            )
            key = fnp.concatenate([key, pad], axis=1)
        k0 = fnp.broadcast_to(key, (msg.shape[0], self.block_size))
        inner = self.byte_hash.digest(
            fnp.concatenate([k0 ^ fnp.uint8(_IPAD), msg], axis=1)
        )
        return self.byte_hash.digest(
            fnp.concatenate(
                [k0 ^ fnp.uint8(_OPAD), fnp.asarray(inner, dtype=fnp.uint8)], axis=1
            )
        )

    def _parameters(self) -> tuple[object, ...]:
        """Everything two instances compare on — the full parameter surface,
        so an `Hmac` riding as pytree aux stays re-trace-safe (the hash rows
        already compare by value)."""
        return (self.byte_hash, self.block_size)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._parameters() == other._parameters()

    def __hash__(self) -> int:
        return hash((type(self), self._parameters()))

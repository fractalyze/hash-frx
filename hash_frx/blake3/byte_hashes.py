# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""BLAKE3 as a `ByteHash` — the seam a consumer reaches it through.

`Blake3` is hash mode's row: the 32-byte digest of a whole message, over a batch
of equal-length ones. BLAKE3's other modes are rows of the same table rather than
separate constructions — keyed hashing and key derivation change the key words a
node opens from and a flag on every compression, extendable output changes how
the root is finished, and nothing below that moves. Keccak's file is laid out the
same way.

**`has_dedicated_fusion` is `False`, and the return type is what matters here.**
No BLAKE3 emitter exists, so `digest` carries no hash-dedicated marker. It is
still a device function that takes a tracer and returns an `Array`, which is what
lets a consumer hash inside its own `@jit` — the flag and that property came
apart at Keccak, and [`byte_hash.py`](../byte_hash.py) is where the rule lives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frx import Array
from frx.typing import ArrayLike

from hash_frx.blake3 import blake3

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash


class Blake3:
    """`ByteHash` for BLAKE3 in hash mode — the standard fixes the key words,
    the chunk size and the tree shape, so nothing is left for a caller to choose
    and equality is by type."""

    digest_size = 32
    has_dedicated_fusion = False

    def digest(self, msg: ArrayLike) -> Array:
        return blake3.digest(msg)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Blake3)

    def __hash__(self) -> int:
        return hash(Blake3)


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_blake3: type[ByteHash] = Blake3

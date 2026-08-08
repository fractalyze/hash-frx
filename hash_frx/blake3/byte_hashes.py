# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""BLAKE3 as a `ByteHash` — the seam a consumer reaches it through.

`Blake3` is hash mode's row: `output_size` bytes of a whole message's hash, over
a batch of equal-length ones. Keyed hashing and key derivation are rows of the
same table rather than separate constructions — they change the key words a node
opens from and a flag on every compression, and nothing below that moves.
Keccak's file is laid out the same way.

**The output length is a parameter, not a second class.** BLAKE3's root output
is a stream and the 32-byte digest is its head, so a caller asking for more bytes
is asking the same hash to be read further — every constant is identical. That is
the axis `_KeccakHash` also makes a parameter; what it splits into separate
*types* is a differing **constant** (`Shake128` and `Shake256` disagree on the
rate). Nothing about BLAKE3 differs that way, so one class carries the length in
the value surface `__eq__` covers, and `digest_size` stays the concrete integer
the seam promises — fixed per instance, never per call.

It takes a default where `Shake256` refuses one: the standard names 32 bytes as
BLAKE3's digest size, so there is a length to pick that is not the caller's to
choose. An extendable output with no such default has none.

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
    """`ByteHash` for BLAKE3 in hash mode, read out to `output_size` bytes.

    The standard fixes the key words, the chunk size and the tree shape, so the
    length is the only thing left for a caller to choose — and it is what two
    instances compare on.
    """

    has_dedicated_fusion = False

    def __init__(self, output_size: int = 32) -> None:
        if output_size < 1:
            raise ValueError(f"output_size must be at least 1, got {output_size}")
        self.digest_size = output_size

    def digest(self, msg: ArrayLike) -> Array:
        return blake3.xof(msg, self.digest_size)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.digest_size == other.digest_size

    def __hash__(self) -> int:
        return hash((type(self), self.digest_size))


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_blake3: type[ByteHash] = Blake3

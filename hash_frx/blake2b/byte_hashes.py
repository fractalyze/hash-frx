# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The BLAKE2b byte hash — the host row over `hashlib` (RFC 7693).

BLAKE2b is HAIFA: Merkle–Damgård plus a byte counter and a finalization flag
threaded into every compression, over a 12-round ChaCha-derived ARX permutation
on eight 64-bit words. None of that construction shows at the seam — a
`ByteHash` hides axis B behind `digest` — but it is why the family lands here
rather than as a `Permutation`: the counter and the flag make the compression
call site construction-bound, so there is no free-standing fixed-width
permutation for `Sponge`/`Compression` to drive.

**Host-first, and the ordering is the point.** BLAKE2b's real consumers —
the EIP-152 precompile, Zcash, Filecoin — are strictly-sequential byte callers
that read each digest back immediately, which is the exact profile the host
rows exist for (`byte_hash.py`), and `hashlib.blake2b` is a C implementation
the standard library already ships. So the host row is free and immediately
useful, while the device row (`hash_frx.digest.blake2b`, `GENERIC` until an
emitter lands) is the follow-up that earns its keep only once a batched
consumer exists.

**The output length is a parameter of *which hash this is*.** RFC 7693 defines
digests of 1..64 bytes, and `HostBlake2b(digest_size=32)` is a different hash
from `HostBlake2b(64)` — not one hash asked for fewer bytes — so the length
rides the value surface `__eq__`/`__hash__` cover, the same rule the SHAKE and
BLAKE3 rows state. The default is 64: unlike an XOF, BLAKE2b names a canonical
full-length form (BLAKE2b-512), which is what a caller means without a length.

Keyed hashing, salting and the tree parameters are RFC 7693 features this row
does not yet carry — each is a value-surface addition (the BLAKE3 keyed rows
are the template), added when a consumer needs it rather than speculatively.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from frx.typing import ArrayLike

from hash_frx.byte_hash import host_digest
from hash_frx.fusion import FusionPath

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from hash_frx.byte_hash import ByteHash

# RFC 7693 §2.1: BLAKE2b digests are 1..64 bytes.
MAX_DIGEST_SIZE = 64


class HostBlake2b:
    """`ByteHash` for host BLAKE2b — `hashlib.blake2b` looped per message.

    The loop it runs under is [`byte_hash.host_digest`](../byte_hash.py),
    shared with every other host row in the package.
    """

    fusion_path = FusionPath.HOST

    def __init__(self, digest_size: int = MAX_DIGEST_SIZE) -> None:
        # Range-checked here rather than left to `hashlib` at the first
        # `digest`, where the caller can no longer choose another length.
        if not 1 <= digest_size <= MAX_DIGEST_SIZE:
            raise ValueError(
                f"digest_size must be in 1..{MAX_DIGEST_SIZE} (RFC 7693), got "
                f"{digest_size}"
            )
        self.digest_size = digest_size

    def _hash_one(self, data: ReadableBuffer) -> bytes:
        return hashlib.blake2b(data, digest_size=self.digest_size).digest()

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(self._hash_one, self.digest_size, msg)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.digest_size == other.digest_size

    def __hash__(self) -> int:
        return hash((type(self), self.digest_size))


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_host_blake2b: type[ByteHash] = HostBlake2b

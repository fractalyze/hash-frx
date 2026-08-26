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
useful, while the device row that earns its keep under a batched consumer is
the sibling module (`blake2b.Blake2b` behind the `hash_frx.digest.blake2b`
marker, `GENERIC` until an emitter lands).

**The output length is a parameter of *which hash this is*.** RFC 7693 defines
digests of 1..64 bytes, and `HostBlake2b(digest_size=32)` is a different hash
from `HostBlake2b(64)` — not one hash asked for fewer bytes — so the length
rides the value surface `__eq__`/`__hash__` cover, the same rule the SHAKE and
BLAKE3 rows state. The default is 64: unlike an XOF, BLAKE2b names a canonical
full-length form (BLAKE2b-512), which is what a caller means without a length.

**Keyed hashing, salting and personalization are here**, and `hashlib` gives
them away: `blake2b(key=, salt=, person=)` takes all three, so each row is a
constructor's width check plus one `_hash_one`. Keying is a separate row
(`HostBlake2bKeyed`) rather than a keyword, which is `Blake3Keyed`'s precedent
and the reason the device sibling states — a key is not a setting, and keeping
`HostBlake2b(digest_size)` a one-argument constructor is what lets it satisfy
`adapter.Xof`.

The tree parameters are still not carried; `blake2_params` writes sequential
mode's constants and records the offsets a tree mode would need.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from frx.typing import ArrayLike

from hash_frx.blake2_params import BLAKE2B_WORD_BYTES, param_block
from hash_frx.byte_hash import HostRow, host_digest

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from hash_frx.byte_hash import ByteHash

# RFC 7693 §2.1: BLAKE2b digests are 1..64 bytes.
MAX_DIGEST_SIZE = 64


class HostBlake2b(HostRow):
    """`ByteHash` for host BLAKE2b — `hashlib.blake2b` looped per message.

    The loop it runs under is [`byte_hash.host_digest`](../byte_hash.py),
    shared with every other host row in the package.

    `salt` and `person` are part of which hash this is, not settings on one:
    RFC 7693 folds both into the initial state through the §2.8 parameter
    block, so `HostBlake2b(32, person=b"ZcashPH")` disagrees with
    `HostBlake2b(32)` on every input. Both ride the value surface alongside
    the output length.
    """

    def __init__(
        self,
        digest_size: int = MAX_DIGEST_SIZE,
        *,
        salt: bytes = b"",
        person: bytes = b"",
    ) -> None:
        # Range-checked here rather than left to `hashlib` at the first
        # `digest`, where the caller can no longer choose another length. The
        # salt and personalization widths go through the same door —
        # `blake2_params` owns them, and reaching for it here rather than for
        # the device module's `_initial_state` is what keeps this file free of
        # the import cycle it would otherwise close (`blake2b.py` imports
        # `MAX_DIGEST_SIZE` from here).
        if not 1 <= digest_size <= MAX_DIGEST_SIZE:
            raise ValueError(
                f"digest_size must be in 1..{MAX_DIGEST_SIZE} (RFC 7693), got "
                f"{digest_size}"
            )
        param_block(BLAKE2B_WORD_BYTES, digest_size, 0, salt, person)
        self.digest_size = digest_size
        self._salt = salt
        self._person = person

    def _parameters(self) -> tuple[object, ...]:
        return (self.digest_size, self._salt, self._person)

    def _hash_one(self, data: ReadableBuffer) -> bytes:
        return hashlib.blake2b(
            data,
            digest_size=self.digest_size,
            salt=self._salt,
            person=self._person,
        ).digest()

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(self._hash_one, self.digest_size, msg)


class HostBlake2bKeyed(HostRow):
    """`ByteHash` for host keyed BLAKE2b — `hashlib.blake2b(key=...)` looped
    per message. The sequential fast path for `blake2b.Blake2bKeyed`, and the
    row a caller porting from libsodium's `crypto_generichash` lands on.

    A separate row rather than a `key=` keyword, for the reason
    `blake2b.Blake2bKeyed` states. The key is secret material held in a plain
    attribute, `__hash__` is over the bytes, and nothing here erases it."""

    def __init__(
        self,
        key: bytes,
        digest_size: int = MAX_DIGEST_SIZE,
        *,
        salt: bytes = b"",
        person: bytes = b"",
    ) -> None:
        if not 1 <= digest_size <= MAX_DIGEST_SIZE:
            raise ValueError(
                f"digest_size must be in 1..{MAX_DIGEST_SIZE} (RFC 7693), got "
                f"{digest_size}"
            )
        # An empty key is a caller bug rather than a demotion to the unkeyed
        # row: `kk` reaches the initial state, so the two are different hashes
        # and silently returning the unkeyed digest would hide the mistake.
        if not key:
            raise ValueError("key must be non-empty; the unkeyed hash is `HostBlake2b`")
        param_block(BLAKE2B_WORD_BYTES, digest_size, len(key), salt, person)
        self.digest_size = digest_size
        self._key = key
        self._salt = salt
        self._person = person

    def _parameters(self) -> tuple[object, ...]:
        return (self.digest_size, self._key, self._salt, self._person)

    def _hash_one(self, data: ReadableBuffer) -> bytes:
        return hashlib.blake2b(
            data,
            digest_size=self.digest_size,
            key=self._key,
            salt=self._salt,
            person=self._person,
        ).digest()

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(self._hash_one, self.digest_size, msg)


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md).
    _bh_host_blake2b: type[ByteHash] = HostBlake2b
    _bh_host_blake2b_keyed: type[ByteHash] = HostBlake2bKeyed

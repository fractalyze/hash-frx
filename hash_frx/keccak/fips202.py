# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The FIPS 202 wrappers — SHA3-256, SHAKE128, SHAKE256 as `ByteHash`es.

Each is one `KeccakSponge` row: a rate, a domain-separation byte, and an output
length. FIPS 202 section 6 fixes the first two per function; the third is fixed
for SHA-3 and a caller's choice for the SHAKEs.

| | rate | suffix | capacity | output |
|---|---|---|---|---|
| `Sha3_256` | 136 B | `0x06` | 512 bits | 32 B |
| `Shake128` | 168 B | `0x1F` | 256 bits | caller's |
| `Shake256` | 136 B | `0x1F` | 512 bits | caller's |

**An XOF's output length is a constructor parameter, not a weakened
`digest_size`.** `Shake256(output_size=64)` is a different hash from
`Shake256(output_size=32)` — not one hash asked for more bytes — so the length
rides in the value surface that `__eq__`/`__hash__` cover, and `digest_size`
stays the concrete integer the seam promises. It has no default: an extendable
output has no natural size, and picking one for the caller is how a scheme ends
up silently truncating.

**`has_dedicated_fusion` is `False` here on device and host alike**, because
Keccak carries only the generic region marker until an emitter exists (#21). So
for these six the flag does not separate substrate the way it does for SHA-256,
and the return type is what does: a device hash returns an `Array` and accepts a
tracer, a host one returns `np.ndarray` and never can. That is a seam question
rather than a fact about these hashes, and it is stated where the rule lives —
[`byte_hash.py`](../byte_hash.py) and `docs/blocks/hash.md`.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.keccak.sponge import KeccakSponge

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

# FIPS 202 section 6.1: SHA3-256(M) = KECCAK[512](M ‖ 01, 256), and section B.2
# packs the `01` domain bits with the opening `1` of `pad10*1` into one byte.
SHA3_256_RATE = 136
SHA3_256_SUFFIX = 0x06
SHA3_256_DIGEST_SIZE = 32

# FIPS 202 section 6.2: SHAKE128(M, d) = KECCAK[256](M ‖ 1111, d), likewise
# packed with the padding's opening bit.
SHAKE128_RATE = 168
SHAKE256_RATE = 136
SHAKE_SUFFIX = 0x1F


class _KeccakHash:
    """The shared body of the device FIPS 202 hashes — one `KeccakSponge` row.

    A subclass supplies the row: `_rate` and `_suffix` from FIPS 202 section 6,
    plus a fixed output where the standard fixes one. Splitting by constants into
    separate *types* rather than taking them as arguments is deliberate — a
    consumer holding one of these in pytree aux compares by type, so `Shake128`
    and `Shake256` must not be one class with a knob.
    """

    _rate: int
    _suffix: int
    has_dedicated_fusion = False

    def __init__(self, output_size: int) -> None:
        self.digest_size = output_size
        self._sponge = KeccakSponge(
            rate=self._rate, suffix=self._suffix, output_size=output_size
        )

    def digest(self, msg: ArrayLike) -> Array:
        return self._sponge.hash(msg)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.digest_size == other.digest_size

    def __hash__(self) -> int:
        return hash((type(self), self.digest_size))


class Sha3_256(_KeccakHash):
    """`ByteHash` for device SHA3-256 — the standard fixes its output at 32 B."""

    _rate = SHA3_256_RATE
    _suffix = SHA3_256_SUFFIX

    def __init__(self) -> None:
        super().__init__(SHA3_256_DIGEST_SIZE)


class Shake128(_KeccakHash):
    """`ByteHash` for device SHAKE128 at a fixed output length."""

    _rate = SHAKE128_RATE
    _suffix = SHAKE_SUFFIX


class Shake256(_KeccakHash):
    """`ByteHash` for device SHAKE256 at a fixed output length.

    The one SLH-DSA's SHAKE parameter sets and ML-DSA's sampling reach for.
    """

    _rate = SHAKE256_RATE
    _suffix = SHAKE_SUFFIX


class _HostKeccak:
    """The shared body of the host siblings — `hashlib` per message.

    The differential partner for the device implementations, and the right choice
    for a strictly sequential caller that reads each digest back immediately: a
    device dispatch per short message costs more than `hashlib` does. It can never
    be called on a traced message, which is the return type saying so.
    """

    has_dedicated_fusion = False

    def __init__(self, digest_size: int) -> None:
        self.digest_size = digest_size

    def _hash_one(self, data: bytes) -> bytes:
        raise NotImplementedError

    def digest(self, msg: ArrayLike) -> np.ndarray:
        rows = np.ascontiguousarray(np.asarray(msg, dtype=np.uint8))  # [B, L]
        out = np.empty((rows.shape[0], self.digest_size), dtype=np.uint8)
        for i, row in enumerate(rows):
            out[i] = np.frombuffer(self._hash_one(row.tobytes()), dtype=np.uint8)
        return out

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.digest_size == other.digest_size

    def __hash__(self) -> int:
        return hash((type(self), self.digest_size))


class HostSha3_256(_HostKeccak):
    """`ByteHash` for host SHA3-256 over `hashlib.sha3_256`."""

    def __init__(self) -> None:
        super().__init__(SHA3_256_DIGEST_SIZE)

    def _hash_one(self, data: bytes) -> bytes:
        return hashlib.sha3_256(data).digest()


class HostShake128(_HostKeccak):
    """`ByteHash` for host SHAKE128 over `hashlib.shake_128`."""

    def _hash_one(self, data: bytes) -> bytes:
        return hashlib.shake_128(data).digest(self.digest_size)


class HostShake256(_HostKeccak):
    """`ByteHash` for host SHAKE256 over `hashlib.shake_256`."""

    def _hash_one(self, data: bytes) -> bytes:
        return hashlib.shake_256(data).digest(self.digest_size)


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md). Named individually
    # because mypy rejects re-annotating one name.
    _bh_sha3: type[ByteHash] = Sha3_256
    _bh_shake128: type[ByteHash] = Shake128
    _bh_shake256: type[ByteHash] = Shake256
    _bh_host_sha3: type[ByteHash] = HostSha3_256
    _bh_host_shake128: type[ByteHash] = HostShake128
    _bh_host_shake256: type[ByteHash] = HostShake256

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Keccak byte hashes — SHA3-256, SHA3-512, SHAKE128, SHAKE256, Keccak-256.

Each is one `KeccakSponge` row: a rate, a domain-separation byte, and an output
length. FIPS 202 section 6 fixes the first two for the four it standardises;
the third is fixed for SHA-3 and a caller's choice for the SHAKEs.

| | rate | suffix | capacity | output |
|---|---|---|---|---|
| `Sha3_256` | 136 B | `0x06` | 512 bits | 32 B |
| `Sha3_512` | 72 B | `0x06` | 1024 bits | 64 B |
| `Shake128` | 168 B | `0x1F` | 256 bits | caller's |
| `Shake256` | 136 B | `0x1F` | 512 bits | caller's |
| `Keccak256` | 136 B | `0x01` | 512 bits | 32 B |

**Four of the five are FIPS 202; `Keccak256` is not.** It is the original
Keccak submission, whose padding NIST changed on standardisation, so it and
`Sha3_256` differ in exactly one byte — `0x01` against `0x06` — and in nothing
else. That is why the module is named for what its contents *are* rather than
for one standard: a table of rows over a shared sponge, which TurboSHAKE and
KangarooTwelve would extend the same way.

**The rate falls as the security level rises**, which is the opposite of the
intuition that a bigger digest reads more per permutation. Capacity is twice the
digest length, and the rate is what is left of the 200-byte state — so SHA3-512
absorbs 72 bytes a block against SHA3-256's 136 and costs closer to twice as
many permutations per message, not the same number.

**An XOF's output length is a constructor parameter, not a weakened
`digest_size`.** `Shake256(output_size=64)` is a different hash from
`Shake256(output_size=32)` — not one hash asked for more bytes — so the length
rides in the value surface that `__eq__`/`__hash__` cover, and `digest_size`
stays the concrete integer the seam promises. That the rate is a type here and
the length a parameter is the family-wide rule, stated once in
[`docs/reference/conventions.md`](../../docs/reference/conventions.md); it is
also why the SHAKEs take no default where BLAKE3's rows do.

**`has_dedicated_fusion` is `False` on every hash here, device and host alike**, because
Keccak carries only the generic region marker until an emitter exists (#21). So
the flag does not separate substrate here the way it does for SHA-256,
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

from hash_frx.byte_hash import host_digest
from hash_frx.keccak.sponge import KeccakSponge

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from hash_frx.byte_hash import ByteHash

# FIPS 202 section 6.1: SHA3-256(M) = KECCAK[512](M ‖ 01, 256) and
# SHA3-512(M) = KECCAK[1024](M ‖ 01, 512), and section B.2 packs the `01` domain
# bits with the opening `1` of `pad10*1` into one byte. The two rows differ in
# capacity alone, so the suffix is shared exactly as the SHAKEs' is below.
SHA3_SUFFIX = 0x06
SHA3_256_RATE = 136
SHA3_256_DIGEST_SIZE = 32
SHA3_512_RATE = 72
SHA3_512_DIGEST_SIZE = 64

# FIPS 202 section 6.2: SHAKE128(M, d) = KECCAK[256](M ‖ 1111, d), likewise
# packed with the padding's opening bit.
SHAKE128_RATE = 168
SHAKE256_RATE = 136
SHAKE_SUFFIX = 0x1F

# Keccak reference (the original SHA-3 submission), `pad10*1` with no domain
# bits under it — so the byte is the padding's opening `1` alone, where FIPS 202
# section 6.1 puts two domain bits beneath it. Frozen by Ethereum before that
# change, which is why the variant outlived the submission.
KECCAK256_RATE = 136
KECCAK256_SUFFIX = 0x01
KECCAK256_DIGEST_SIZE = 32


class _KeccakHash:
    """The shared body of the device FIPS 202 hashes — one `KeccakSponge` row.

    A subclass supplies the row: `_rate` and `_suffix` from FIPS 202 section 6,
    plus a fixed output where the standard fixes one. Splitting by constants into
    separate *types* rather than taking them as arguments is the family-wide rule
    (`docs/reference/conventions.md`).
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
    _suffix = SHA3_SUFFIX

    def __init__(self) -> None:
        super().__init__(SHA3_256_DIGEST_SIZE)


class Sha3_512(_KeccakHash):
    """`ByteHash` for device SHA3-512 — the standard fixes its output at 64 B.

    ML-KEM's `G` and the SHA-3 SLH-DSA parameter sets' `H` and `T_l`. The
    72-byte rate is the narrowest in the family, so a message spans roughly
    twice the blocks it would under SHA3-256.
    """

    _rate = SHA3_512_RATE
    _suffix = SHA3_SUFFIX

    def __init__(self) -> None:
        super().__init__(SHA3_512_DIGEST_SIZE)


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


class Keccak256(_KeccakHash):
    """`ByteHash` for device Keccak-256 — the original submission, not SHA3-256.

    A separate type rather than `Sha3_256(legacy_padding=True)`: a flag reads as
    a robustness knob, and this is a choice between two standards that a caller
    has to make deliberately. The Ethereum address derivation and every EVM
    `KECCAK256` need this one; a FIPS 202 consumer needs the other; nothing wants
    to pick at runtime.
    """

    _rate = KECCAK256_RATE
    _suffix = KECCAK256_SUFFIX

    def __init__(self) -> None:
        super().__init__(KECCAK256_DIGEST_SIZE)


class _HostKeccak:
    """The shared body of the host siblings — `hashlib` per message.

    The differential partner for the device implementations, and the right choice
    for a strictly sequential caller that reads each digest back immediately: a
    device dispatch per short message costs more than `hashlib` does. It can never
    be called on a traced message, which is the return type saying so.

    A subclass supplies `_hash_one`, which is the whole row: which `hashlib`
    function, and — for the XOFs — how many bytes to read out of it. The loop it
    runs under is [`byte_hash.host_digest`](../byte_hash.py), shared with every
    other host row in the package.
    """

    has_dedicated_fusion = False

    def __init__(self, digest_size: int) -> None:
        self.digest_size = digest_size

    def _hash_one(self, data: ReadableBuffer) -> bytes:
        raise NotImplementedError

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(self._hash_one, self.digest_size, msg)

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

    def _hash_one(self, data: ReadableBuffer) -> bytes:
        return hashlib.sha3_256(data).digest()


class HostSha3_512(_HostKeccak):
    """`ByteHash` for host SHA3-512 over `hashlib.sha3_512`."""

    def __init__(self) -> None:
        super().__init__(SHA3_512_DIGEST_SIZE)

    def _hash_one(self, data: ReadableBuffer) -> bytes:
        return hashlib.sha3_512(data).digest()


class HostShake128(_HostKeccak):
    """`ByteHash` for host SHAKE128 over `hashlib.shake_128`."""

    def _hash_one(self, data: ReadableBuffer) -> bytes:
        return hashlib.shake_128(data).digest(self.digest_size)


class HostShake256(_HostKeccak):
    """`ByteHash` for host SHAKE256 over `hashlib.shake_256`."""

    def _hash_one(self, data: ReadableBuffer) -> bytes:
        return hashlib.shake_256(data).digest(self.digest_size)


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md). Named individually
    # because mypy rejects re-annotating one name.
    _bh_sha3_256: type[ByteHash] = Sha3_256
    _bh_sha3_512: type[ByteHash] = Sha3_512
    _bh_shake128: type[ByteHash] = Shake128
    _bh_shake256: type[ByteHash] = Shake256
    _bh_keccak256: type[ByteHash] = Keccak256
    _bh_host_sha3_256: type[ByteHash] = HostSha3_256
    _bh_host_sha3_512: type[ByteHash] = HostSha3_512
    _bh_host_shake128: type[ByteHash] = HostShake128
    _bh_host_shake256: type[ByteHash] = HostShake256

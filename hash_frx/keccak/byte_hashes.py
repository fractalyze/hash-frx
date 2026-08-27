# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Keccak byte hashes — the four SHA-3 rows, both SHAKEs, and Keccak-256.

Each is one `KeccakSponge` row: a rate, a domain-separation byte, and an output
length. FIPS 202 section 6 fixes the first two for the six it standardises;
the third is fixed for SHA-3 and a caller's choice for the SHAKEs.

| | rate | suffix | capacity | output |
|---|---|---|---|---|
| `Sha3_224` | 144 B | `0x06` | 448 bits | 28 B |
| `Sha3_256` | 136 B | `0x06` | 512 bits | 32 B |
| `Sha3_384` | 104 B | `0x06` | 768 bits | 48 B |
| `Sha3_512` | 72 B | `0x06` | 1024 bits | 64 B |
| `Shake128` | 168 B | `0x1F` | 256 bits | caller's |
| `Shake256` | 136 B | `0x1F` | 512 bits | caller's |
| `Keccak256` | 136 B | `0x01` | 512 bits | 32 B |

**All four fixed-output SHA-3 rows are here.** FIPS 202 defines exactly these
four, and a module claiming to implement it needs all of them whether or not a
consumer has asked — CAVP validation of a SHA-3 implementation requires the set.
SHA3-224's 144 bytes is the widest SHA-3 rate, second in the table only to
SHAKE128's 168, so it absorbs more message per permutation than any other row
that fixes its output.

**Six of the seven are FIPS 202; `Keccak256` is not.** It is the original
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

**`fusion_path` is `DEDICATED` or `GENERIC` per what the running backend can
reach**, because the
whole padded absorb and squeeze lowers to one `hash_frx.digest.keccak_sponge` kernel
only where that emitter exists. The switch is
`keccak.permutation._routes_to_dedicated_emitter`, which the rows read
through `KeccakF1600`; it asks both whether the pinned plugin carries the
emitters and whether this backend has them. Both legs carry them from the wheel
that first shipped the CPU sponge emitter, so the rows read `DEDICATED`
on cpu and gpu; a backend without an arm — Metal today — is still `GENERIC`.

`is_one_kernel` answers "does this lower to a dedicated kernel", which a pin or
a backend can change — so it is a routing fact rather than a property of these
hashes, and the rule lives where the seam is stated
([`byte_hash.py`](../byte_hash.py) and `docs/blocks/hash.md`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import DeviceRow
from hash_frx.keccak.permutation import KeccakF1600
from hash_frx.keccak.sponge import KeccakSponge

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

# FIPS 202 section 6.1: SHA3-n(M) = KECCAK[2n](M ‖ 01, n), and section B.2 packs
# the `01` domain bits with the opening `1` of `pad10*1` into one byte. The four
# rows differ in capacity alone — 2n, which fixes the rate at 200 - 2n/8 bytes —
# so the suffix is shared across them exactly as the SHAKEs' is below.
SHA3_SUFFIX = 0x06
SHA3_224_RATE = 144
SHA3_224_DIGEST_SIZE = 28
SHA3_256_RATE = 136
SHA3_256_DIGEST_SIZE = 32
SHA3_384_RATE = 104
SHA3_384_DIGEST_SIZE = 48
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


class _KeccakHash(DeviceRow):
    """The shared body of the device FIPS 202 hashes — one `KeccakSponge` row.

    A subclass supplies the row: `_rate` and `_suffix` from FIPS 202 section 6,
    plus a fixed output where the standard fixes one. Splitting by constants into
    separate *types* rather than taking them as arguments is the family-wide rule
    (`docs/reference/conventions.md`).

    `suffix` may instead be passed per instance, which is what an SP 800-185
    derived function needs: cSHAKE's domain byte is a function of its *arguments*
    rather than of its type, because empty `N` and `S` make it plain SHAKE
    (`cshake.py`). The rate stays a class attribute either way — that one really
    is the type — and the sponge and `fusion_path` wiring stays written once
    here, so a derived function cannot drift from the routing `KeccakSponge.hash`
    actually takes.
    """

    _rate: int
    _suffix: int

    def __init__(self, output_size: int, suffix: int | None = None) -> None:
        self.digest_size = output_size
        self._sponge = KeccakSponge(
            rate=self._rate,
            suffix=self._suffix if suffix is None else suffix,
            output_size=output_size,
        )
        # Derived rather than declared, so it cannot disagree with the routing
        # `KeccakSponge.hash` actually takes — the same arrangement
        # `poseidon2.Poseidon2` uses. `DeviceRow` takes the resolved path
        # precisely so a row that derives it from a delegate goes through the
        # same door as one that reads a pin-and-backend gate.
        super().__init__(KeccakF1600().fusion_path)

    def digest(self, msg: ArrayLike) -> Array:
        return self._sponge.hash(msg)


class Sha3_224(_KeccakHash):
    """`ByteHash` for device SHA3-224 — the standard fixes its output at 28 B.

    The widest SHA-3 rate at 144 bytes, because capacity is twice the digest and
    this is the shortest digest FIPS 202 defines. So it absorbs the most message
    per permutation of the four, and it is the only row here whose rate no other
    row shares.
    """

    _rate = SHA3_224_RATE
    _suffix = SHA3_SUFFIX

    def __init__(self) -> None:
        super().__init__(SHA3_224_DIGEST_SIZE)


class Sha3_256(_KeccakHash):
    """`ByteHash` for device SHA3-256 — the standard fixes its output at 32 B."""

    _rate = SHA3_256_RATE
    _suffix = SHA3_SUFFIX

    def __init__(self) -> None:
        super().__init__(SHA3_256_DIGEST_SIZE)


class Sha3_384(_KeccakHash):
    """`ByteHash` for device SHA3-384 — the standard fixes its output at 48 B.

    The row the X.509 and CMS object identifiers name, and the one an IETF
    profile asking for a 384-bit SHA-3 digest means. Its 104-byte rate sits
    between SHA3-512's 72 and SHA3-256's 136 and is shared with nothing.
    """

    _rate = SHA3_384_RATE
    _suffix = SHA3_SUFFIX

    def __init__(self) -> None:
        super().__init__(SHA3_384_DIGEST_SIZE)


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


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md). Named individually
    # because mypy rejects re-annotating one name.
    _bh_sha3_224: type[ByteHash] = Sha3_224
    _bh_sha3_256: type[ByteHash] = Sha3_256
    _bh_sha3_384: type[ByteHash] = Sha3_384
    _bh_sha3_512: type[ByteHash] = Sha3_512
    _bh_shake128: type[ByteHash] = Shake128
    _bh_shake256: type[ByteHash] = Shake256
    _bh_keccak256: type[ByteHash] = Keccak256

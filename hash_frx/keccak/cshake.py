# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""cSHAKE128 / cSHAKE256 — SHAKE with a function name and a customization string.

SP 800-185 section 3. cSHAKE is the customizable XOF the rest of the document is
built on: KMAC and TupleHash are both cSHAKE with a fixed `N` and a
construction-specific message, so this is the layer they inherit rather than a
sibling of theirs.

**It is the ordinary SHAKE absorb over a longer message.** `bytepad` pads the
encoded `N ‖ S` to a whole number of rate blocks, so the prefix is rate-aligned
at every admissible length and the message that follows starts a fresh block.
There is no prefix *stage* — no second schedule, no state to carry between two
loops — only a longer byte string handed to the sponge already in use. What the
customization buys is a buffer, not a mechanism, which is why this module is
about a hundred lines and touches neither `sponge.py` nor the permutation.

**The empty-N-and-S case is a branch, not an identity.** Section 3.3 defines
cSHAKE with both strings empty to *be* SHAKE — suffix `0x1F` and no prefix block
— rather than to be cSHAKE over an empty customization. The two genuinely
differ: absorbing `bytepad(encode_string("") ‖ encode_string(""), rate)` under
suffix `0x04` is a well-defined hash and it is not SHAKE's. So the branch is
taken here at construction, where it is one conditional over two constants, and
`testing/cshake_test.py` pins both sides — that the fallback matches `hashlib`'s
SHAKE, and that the path not taken would have disagreed.

The neighbouring family answers the same question the other way, which is worth
knowing before assuming either. Ascon-CXOF128 has no such fallback: its IV
differs from Ascon-XOF128's, so an empty customization is still a
*customization* and the two disagree at every input. Same question, opposite
answer, one document apart.

**Domain separation is the point, so the parameters are part of the type's
value.** `N` and `S` ride on the instance and are compared by
`_parameters`, because two cSHAKEs with different customization are two
different hashes — that is the entire purpose of the construction, and a jit
cache key that ignored them would serve one's compiled program for the other.
They are `bytes` for the reason `Blake3Keyed`'s key is: `digest(msg)` takes a
message and nothing else, so anything that is not the message is part of *which
hash this is*.

`N` is reserved. Section 3.4 says the function name is for NIST's own derived
functions and should only be set to values NIST defines — `"KMAC"`,
`"TupleHash"`, `"ParallelHash"`. A consumer doing its own domain separation
wants `S`, and the argument order here puts `S` first for that reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import device_message
from hash_frx.keccak.byte_hashes import (
    SHAKE128_RATE,
    SHAKE256_RATE,
    SHAKE_SUFFIX,
    _KeccakHash,
)
from hash_frx.keccak.encodings import bytepad, encode_string

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

# FIPS 202 section 6.3 packs cSHAKE's `00` domain bits with `pad10*1`'s opening
# `1`, exactly as SHA-3's `01` and SHAKE's `1111` are packed — so cSHAKE differs
# from SHAKE in this one byte and in the prefix block, and in nothing else.
# `validate_sponge_params` already admits it by name.
CSHAKE_SUFFIX = 0x04


def _prefix_block(name: bytes, customization: bytes, rate: int) -> bytes:
    """SP 800-185 section 3.3's `bytepad(encode_string(N) ‖ encode_string(S), rate)`
    — the whole number of rate blocks absorbed ahead of the message.

    Empty for the SHAKE fallback — which is the signal `__init__` reads to pick
    the domain byte, and `digest` to skip the prefix entirely and defer to the
    plain row.
    """
    if not name and not customization:
        return b""
    return bytepad(encode_string(name) + encode_string(customization), rate)


class _CShake(_KeccakHash):
    """The shared body of the two cSHAKEs — a `_KeccakHash` whose suffix and
    message prefix are decided from `(N, S)` at construction.

    A subclass supplies `_rate`, which is the rate of the SHAKE it customizes
    (168 for cSHAKE128, 136 for cSHAKE256) and the `w` its `bytepad` uses. Those
    are the same number for the same reason: the prefix exists to fill whole
    blocks of the sponge it is about to enter.

    `_suffix` is set per instance before `super().__init__`, where every FIPS 202
    row above sets it as a class attribute. That is the honest shape — cSHAKE's
    domain byte is a function of its *arguments*, not of its type — and it is
    also why the base takes no suffix parameter: one there would be a public knob
    on `Shake128` and `Shake256`, which inherit that constructor, turning a row
    into a different standard without entering `_parameters`.

    The prefix *bytes* are built once here rather than per `digest` call. The
    device array is still materialised per call, which costs ~15 us eagerly and
    nothing under `jit`, where it constant-folds — the same arrangement
    `Blake3Keyed` uses for its key and `KeccakSponge._pad` for its tail.

    `output_size` is keyword-only and required. An XOF names no output length, so
    the family refuses a default (`docs/reference/conventions.md`), and keyword
    -only additionally stops `CShake128(24)` — the shape every other
    variable-output row takes — from silently reading 24 as a customization of
    twenty-four NUL bytes.
    """

    def __init__(
        self,
        customization: bytes = b"",
        name: bytes = b"",
        *,
        output_size: int,
    ) -> None:
        self._name = bytes(name)
        self._customization = bytes(customization)
        self._prefix = _prefix_block(self._name, self._customization, self._rate)
        # Section 3.3: with both strings empty this *is* SHAKE, which is a
        # different domain byte and not merely a shorter message.
        self._suffix = CSHAKE_SUFFIX if self._prefix else SHAKE_SUFFIX
        super().__init__(output_size)

    def digest(self, msg: ArrayLike) -> Array:
        if not self._prefix:
            # Section 3.3's fallback is plain SHAKE, which is exactly what the
            # base row already computes — so it goes through that door rather
            # than converting the message a second time on the way.
            return super().digest(msg)
        message = device_message(msg)
        # The prefix is one row — it is a property of the hash, which every row
        # of a batch shares — so it is broadcast rather than built per row, the
        # arrangement `byte_hash.padded_batch` uses at the other end of the
        # message. Both concatenates stay outside the marked region, so the
        # whole hash is still the one `keccak_sponge` kernel the plain row is.
        head = fnp.broadcast_to(
            fnp.asarray(np.frombuffer(self._prefix, dtype=np.uint8)),
            (message.shape[0], len(self._prefix)),
        )
        return self._sponge.hash(fnp.concatenate([head, message], axis=-1))

    def _parameters(self) -> tuple[object, ...]:
        # The customization is the whole point of the construction, so it is the
        # jit cache key as much as the output length is. Omitting it here would
        # serve one customization's compiled program for another's, silently and
        # with the right shape.
        return (*super()._parameters(), self._name, self._customization)


class CShake128(_CShake):
    """`ByteHash` for device cSHAKE128 at a fixed output length.

    SHAKE128's 168-byte rate, customized. With `S` and `N` both empty this is
    `Shake128(output_size)` byte for byte, by section 3.3 rather than by
    accident.
    """

    _rate = SHAKE128_RATE


class CShake256(_CShake):
    """`ByteHash` for device cSHAKE256 at a fixed output length.

    SHAKE256's 136-byte rate, customized — the one a 256-bit security level
    reaches for, and the rate KMAC256 inherits.
    """

    _rate = SHAKE256_RATE


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md). Named individually
    # because mypy rejects re-annotating one name.
    _bh_cshake128: type[ByteHash] = CShake128
    _bh_cshake256: type[ByteHash] = CShake256

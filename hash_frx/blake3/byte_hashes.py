# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The BLAKE3 byte hashes — hash mode, keyed hashing, and key derivation.

Each is one row of spec section 2.3's table: a key the tree opens from, a flag
every compression carries, and an output length.

| | key the tree opens from | mode flag |
|---|---|---|
| `Blake3` | the IV | — |
| `Blake3Keyed` | the caller's 32 bytes | `KEYED_HASH` |
| `Blake3DeriveKey` | the hashed context string | `DERIVE_KEY_MATERIAL` |

What the standard fixes per row is the mode flag and *where* the key comes from
— not the key's value, which is the caller's on two of the three. That is why
the rows are types and the key, context and length are parameters;
[`docs/reference/conventions.md`](../../docs/reference/conventions.md) states the
rule for the family, including why every row takes a 32-byte default where
`Shake256` refuses one.

Keccak's file is the same table one layer down: its rows differ by data on the
class (`_rate`, `_suffix`) where these differ by which mode function `_read`
calls. The hook is a method here because a row routes through `blake3`'s own
`xof` / `keyed_xof` / `derive_key`, so the seam cannot drift from the functional
API — the same bytes by construction rather than by a second assembly of the
same `Mode`.

**What the message is differs per row, and only the name says so.** `Blake3` and
`Blake3Keyed` hash a message; `Blake3DeriveKey` hashes *key material*, with the
context riding as the instance's parameter. The seam cannot express that — it has
one `digest(msg)` — so a consumer reaching a `ByteHash` generically gets whichever
reading the row it was handed carries.

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


class _Blake3Hash:
    """The shared body of the three modes — everything but which mode it is.

    A subclass supplies the row: `_read`, which of `blake3`'s mode functions
    reads a message, and `_parameters`, what the mode's own parameters are.
    `digest` stays here and forwards, so the seam's name and signature are
    written once however many rows there are.

    `_parameters` is not bookkeeping — `__eq__` covers whatever it returns, so a
    row that adds a key and forgets to name it there compares two different keys
    equal, and serves one key's trace for another as pytree aux. It never errors.
    """

    has_dedicated_fusion = False

    def __init__(self, output_size: int = blake3.DIGEST_LEN) -> None:
        if output_size < 1:
            raise ValueError(f"output_size must be at least 1, got {output_size}")
        self.digest_size = output_size

    def _read(self, msg: ArrayLike) -> Array:
        raise NotImplementedError

    def _parameters(self) -> tuple[object, ...]:
        """Everything two instances of this row compare on."""
        return (self.digest_size,)

    def digest(self, msg: ArrayLike) -> Array:
        return self._read(msg)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._parameters() == other._parameters()

    def __hash__(self) -> int:
        return hash((type(self), self._parameters()))


class Blake3(_Blake3Hash):
    """`ByteHash` for BLAKE3 in hash mode, read out to `output_size` bytes.

    The standard fixes the key words, the chunk size and the tree shape, so the
    length is the only thing left for a caller to choose — and it is what two
    instances compare on.
    """

    def _read(self, msg: ArrayLike) -> Array:
        return blake3.xof(msg, self.digest_size)


class Blake3Keyed(_Blake3Hash):
    """`ByteHash` for keyed BLAKE3 — the mode a PRF consumer reaches for.

    The key is a 32-byte `bytes` rather than an array, because the seam has
    nowhere to put a per-call one: `digest(msg)` takes a message and nothing
    else, so the key is part of *which hash this is*, and pytree aux compares it
    by value. Two consequences a caller should choose deliberately rather than
    discover:

    - **A new key is a new trace**, and the key rides in the compiled program's
      constant pool. For a per-call key — a fresh signing key per signature —
      call `blake3.keyed_xof` directly, where the key is an operand and one
      compiled program serves every key.
    - **It is secret material held in a plain attribute.** Nothing here erases
      it, and `__hash__` is over the bytes.
    """

    def __init__(self, key: bytes, output_size: int = blake3.DIGEST_LEN) -> None:
        if len(key) != blake3.KEY_LEN:
            raise ValueError(f"key must be {blake3.KEY_LEN} bytes, got {len(key)}")
        super().__init__(output_size)
        self._key = bytes(key)

    def _read(self, msg: ArrayLike) -> Array:
        return blake3.keyed_xof(self._key, msg, self.digest_size)

    def _parameters(self) -> tuple[object, ...]:
        return (*super()._parameters(), self._key)


class Blake3DeriveKey(_Blake3Hash):
    """`ByteHash` for BLAKE3's KDF — `digest` derives from *key material*.

    The context is the instance's parameter and the message is the secret being
    derived from, which is the inverse of what the argument order of a KDF
    usually suggests. The standard asks for a hardcoded, globally unique UTF-8
    context — application name, date, purpose — so a constant on the hash is
    where it belongs; a context that varied per call would be domain separation
    that separates nothing.

    A `str` context and its UTF-8 bytes are the same hash and compare equal:
    they derive identical bytes, so treating them as two would make one of them
    a second jit cache key for no computation.
    """

    def __init__(
        self, context: str | bytes, output_size: int = blake3.DIGEST_LEN
    ) -> None:
        super().__init__(output_size)
        self._context = blake3.context_bytes(context)

    def _read(self, msg: ArrayLike) -> Array:
        return blake3.derive_key(self._context, msg, self.digest_size)

    def _parameters(self) -> tuple[object, ...]:
        return (*super()._parameters(), self._context)


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md). Named individually
    # because mypy rejects re-annotating one name.
    _bh_blake3: type[ByteHash] = Blake3
    _bh_blake3_keyed: type[ByteHash] = Blake3Keyed
    _bh_blake3_derive_key: type[ByteHash] = Blake3DeriveKey

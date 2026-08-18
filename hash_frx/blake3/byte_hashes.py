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

**`fusion_path` derives from the pin and the backend, like Keccak's.** Every
row routes through [`blake3.tree_hash`](blake3.py), the name-routed
`hash_frx.blake3` composite, and the pinned plugin recognizes that name on the
CPU and GPU compilers (fractalyze/xla#499, #507) — so the rows read `DEDICATED`
on both legs, and hardcoding the answer instead is exactly how the flag went
stale for two pins after the emitter shipped. A backend without the arm —
Metal today — inlines the marker: same bytes, no kernel, `GENERIC`, yet still
a device function a consumer may call inside its own `@jit`. Keeping that
state distinct from `HOST` is what the seam's three-state `FusionPath` is for
([`byte_hash.py`](../byte_hash.py)); `blake3.testing.emitter` reads the same
switch rather than spelling its own, so the caps it gates lift with the pin.

**Each row has a host sibling** — `HostBlake3`, `HostBlake3Keyed`,
`HostBlake3DeriveKey` — over the BLAKE3 team's own Rust binding, mirroring
`HostSha256` and the Keccak host rows. They are the right choice for a
strictly-sequential caller that reads each digest back immediately, where a
device dispatch per short message costs more than a native hash does, and they
are the differential partner the published vectors cannot be: agreement with the
reference implementation at a random length is a check no table of 35 lengths
performs. They read `HOST` everywhere — a host loop cannot stop being one, so
this is the one row class attribute in the taxonomy — and their `np.ndarray`
return type stays the authority on why they may never see a tracer.

The narrowing worth knowing before reaching for one: the binding takes a
derive-key context as a `str`, so `HostBlake3DeriveKey` refuses a context that is
not valid UTF-8 where the device row would hash it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# The BLAKE3 team's Rust binding (the `blake3` distribution on PyPI), aliased
# because the unqualified name is this package's own `blake3` module below.
import blake3 as blake3_py
import frx
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.blake3 import blake3
from hash_frx.byte_hash import host_digest
from hash_frx.fusion import FusionPath

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from hash_frx.byte_hash import ByteHash

# Whether the pinned Fractalyze XLA plugin ships the BLAKE3 emitter, and on
# which backends. fractalyze/xla#499 registers the recognizer and rewriter on
# the GPU compiler and #507 on the CPU one, so unlike Keccak's GPU-only tuple
# this carries both legs; the pin floor in `pyproject.toml` is already above
# the wheel that first shipped them. Family-wide rationale for the two-flag
# shape in `keccak.permutation`; a backend absent from the tuple — Metal today
# — still emits the marker (an unrecognized name inlines byte-neutrally, and
# there is no per-node routing alternative for a whole-tree digest).
_DEDICATED_EMITTER_AVAILABLE = True
_EMITTER_BACKENDS = ("cpu", "gpu")


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry the BLAKE3 emitter. Read
    per construction so importing does not initialize a backend; the lookup
    behind `frx.default_backend()` is memoized."""
    return _DEDICATED_EMITTER_AVAILABLE and frx.default_backend() in _EMITTER_BACKENDS


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

    def __init__(self, output_size: int = blake3.DIGEST_LEN) -> None:
        if output_size < 1:
            raise ValueError(f"output_size must be at least 1, got {output_size}")
        self.digest_size = output_size
        # Per instance rather than on the class: the emitter switch is a
        # property of the pin and the backend, and a value read at import would
        # pin the answer before anything could vary it (`_KeccakHash` states
        # the same rule).
        self.fusion_path = FusionPath.from_routing(_routes_to_dedicated_emitter())

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


class _HostBlake3Hash:
    """The shared body of the three host siblings — `blake3` per message.

    The same `_read` / `_parameters` split as `_Blake3Hash`, and for the same
    reason: a row that adds a key and forgets to name it in `_parameters` compares
    two different keys equal and hands one key's trace to the other's caller. The
    Keccak host base compares on `digest_size` alone, which is why it is not the
    shape these rows copy — two of the three carry a parameter beyond the length.

    The loop `_read` runs under is [`byte_hash.host_digest`](../byte_hash.py),
    shared with every other host row in the package.
    """

    fusion_path = FusionPath.HOST

    def __init__(self, output_size: int = blake3.DIGEST_LEN) -> None:
        if output_size < 1:
            raise ValueError(f"output_size must be at least 1, got {output_size}")
        self.digest_size = output_size

    def _read(self, msg: ReadableBuffer) -> bytes:
        raise NotImplementedError

    def _parameters(self) -> tuple[object, ...]:
        """Everything two instances of this row compare on."""
        return (self.digest_size,)

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(self._read, self.digest_size, msg)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._parameters() == other._parameters()

    def __hash__(self) -> int:
        return hash((type(self), self._parameters()))


class HostBlake3(_HostBlake3Hash):
    """`ByteHash` for host BLAKE3 in hash mode, read out to `output_size` bytes."""

    def _read(self, msg: ReadableBuffer) -> bytes:
        return blake3_py.blake3(msg).digest(self.digest_size)


class HostBlake3Keyed(_HostBlake3Hash):
    """`ByteHash` for host keyed BLAKE3 — `Blake3Keyed`'s sibling.

    The key is a 32-byte `bytes` for the reason it is on the device row: the seam
    has nowhere to put a per-call one, so it is part of *which hash this is*. The
    two caveats stated there hold here bar the tracing one — it is secret material
    in a plain attribute, and `__hash__` is over the bytes.
    """

    def __init__(self, key: bytes, output_size: int = blake3.DIGEST_LEN) -> None:
        if len(key) != blake3.KEY_LEN:
            raise ValueError(f"key must be {blake3.KEY_LEN} bytes, got {len(key)}")
        super().__init__(output_size)
        self._key = bytes(key)

    def _read(self, msg: ReadableBuffer) -> bytes:
        return blake3_py.blake3(msg, key=self._key).digest(self.digest_size)

    def _parameters(self) -> tuple[object, ...]:
        return (*super()._parameters(), self._key)


class HostBlake3DeriveKey(_HostBlake3Hash):
    """`ByteHash` for host BLAKE3 key derivation — `Blake3DeriveKey`'s sibling.

    As on the device row, the context is the instance's parameter and `digest`
    reads *key material*.

    **The context is held as a `str` here and as `bytes` there, and that is a real
    narrowing rather than a style difference.** `blake3.blake3(...,
    derive_key_context=...)` takes only a `str`, while the device row hashes
    whatever bytes `blake3.context_bytes` produced — so a context that is not
    valid UTF-8 is a hash the device row can compute and this one cannot. It is
    refused at construction, where the caller can still choose another context,
    rather than at the first `digest`. The standard asks for a UTF-8 context
    string, so nothing that follows it can hit this.

    A `str` context and its UTF-8 bytes remain one hash, the same as on the device
    row: both normalize to the same `str` and so compare equal.
    """

    def __init__(
        self, context: str | bytes, output_size: int = blake3.DIGEST_LEN
    ) -> None:
        super().__init__(output_size)
        try:
            self._context = blake3.context_bytes(context).decode()
        except UnicodeDecodeError as e:
            raise ValueError(
                "context must be valid UTF-8 for the host row; the standard names "
                f"the derive-key context a UTF-8 string, got {context!r}"
            ) from e

    def _read(self, msg: ReadableBuffer) -> bytes:
        return blake3_py.blake3(msg, derive_key_context=self._context).digest(
            self.digest_size
        )

    def _parameters(self) -> tuple[object, ...]:
        return (*super()._parameters(), self._context)


if TYPE_CHECKING:
    # Seam-conformance pins (docs/reference/conventions.md). Named individually
    # because mypy rejects re-annotating one name.
    _bh_blake3: type[ByteHash] = Blake3
    _bh_blake3_keyed: type[ByteHash] = Blake3Keyed
    _bh_blake3_derive_key: type[ByteHash] = Blake3DeriveKey
    _bh_host_blake3: type[ByteHash] = HostBlake3
    _bh_host_blake3_keyed: type[ByteHash] = HostBlake3Keyed
    _bh_host_blake3_derive_key: type[ByteHash] = HostBlake3DeriveKey

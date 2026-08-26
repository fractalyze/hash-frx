# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The BLAKE3 rows — the named, marked, public surface over the three modes.

[`modes.py`](modes.py) is the whole construction as a function of a `Mode`; this
is where it gets a name, a marker and an entry point. Each row of spec section
2.3's table appears twice here, and deliberately so: once functionally
(`digest` / `keyed_digest` / `derive_key` and their extendable-output siblings)
and once as a `ByteHash` (`Blake3` / `Blake3Keyed` / `Blake3DeriveKey`), with the
seam rows routing through the functional ones rather than assembling a `Mode` of
their own. Two spellings of one row cannot drift when one of them calls the other.

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

**What the message is differs per row, and only the name says so.** `Blake3` and
`Blake3Keyed` hash a message; `Blake3DeriveKey` hashes *key material*, with the
context riding as the instance's parameter. The seam cannot express that — it has
one `digest(msg)` — so a consumer reaching a `ByteHash` generically gets whichever
reading the row it was handed carries.

**`fusion_path` derives from the pin and the backend, like Keccak's.** Every row
routes through `tree_hash` below, the name-routed `hash_frx.digest.blake3`
composite, and the pinned plugin recognizes that name on the CPU and GPU
compilers (fractalyze/xla#499, #507) — so the rows read `DEDICATED` on both legs,
and hardcoding the answer instead is exactly how the flag went stale for two pins
after the emitter shipped. A backend without the arm — Metal today — inlines the
marker: same bytes, no kernel, `GENERIC`, yet still a device function a consumer
may call inside its own `@jit`. Keeping that state distinct from `HOST` is what
the seam's three-state `FusionPath` is for ([`byte_hash.py`](../byte_hash.py));
`blake3.testing.emitter` reads the same switch rather than spelling its own, so
the caps it gates lift with the pin.

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

from functools import partial
from typing import TYPE_CHECKING

# The BLAKE3 team's Rust binding (the `blake3` distribution on PyPI), aliased
# because the unqualified name is this package's own `blake3` package below.
import blake3 as blake3_py
import frx
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.blake3.compress import IV_WORDS
from hash_frx.blake3.modes import (
    DIGEST_LEN,
    KEY_LEN,
    Mode,
    context_bytes,
    derive_key_mode,
    hash_mode,
    keyed_mode,
    pair_bytes,
    unmarked_hash,
    unmarked_non_root_hash,
    unmarked_parent_hash,
)
from hash_frx.byte_hash import DeviceRow, HostRow, device_message, host_digest
from hash_frx.fusion import FusionPath, fused_region, routing

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from hash_frx.byte_hash import ByteHash

BLAKE3_MARKER = "hash_frx.digest.blake3"
# Marker revision riding as `composite.version`, the way `hash_frx.digest.sha256`
# carries one: a contract change stages through it rather than through a rename,
# which the recognizer would not accept and which would silently lose fusion.
BLAKE3_MARKER_VERSION = 1

# A Merkle parent is a different construction, so it takes a name of its own
# rather than an attribute on the one above.
#
# The pull to reuse `hash_frx.digest.blake3` is real — the operands have the same
# shapes, and `non_root` is precedent for selecting a variant by attribute. It
# is wrong, and silently: a recognizer matches by NAME, so a shipped emitter
# that predates the attribute recognizes the marker anyway, ignores what it
# does not know, and runs the message-hash construction on a pair of chaining
# values. Measured on frx 0.10.2.dev20260813075049 — a parent marked that way
# returned the 64-byte message hash rather than the parent compression, with
# the right decomposition sitting unused in the module. An unrecognized NAME
# has no such failure: the composite inlines and the bytes stay right while
# only fusion is lost (see `fusion.py`), so a new name degrades safely where a
# new attribute degrades wrongly.
BLAKE3_PARENT_MARKER = "hash_frx.compress.blake3_parent"
BLAKE3_PARENT_MARKER_VERSION = 1


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and a Merkle commit emits a digest per leaf and per internal level —
# so the uncached re-trace of a 16-compression body would dominate the
# first-trace floor (cf. `sha256.sha256_merkle_damgard`). `inline=True` splices
# the cached jaxpr into the enclosing trace, so the emitted module — one
# composite per hash — is unchanged. `out_len`, `flags` and `non_root` are
# static: each fixes the shape or the finalization of the emitted program, and
# each rides the marker as an attribute.
@partial(frx.jit, inline=True, static_argnames=("out_len", "flags", "non_root"))
def tree_hash(
    msg: Array, key_words: Array, *, out_len: int, flags: int, non_root: bool = False
) -> Array:
    """A whole BLAKE3 hash — chunks, tree and root output — as the name-routed
    `hash_frx.digest.blake3` composite: uint8 `[B, L]` -> uint8 `[B, out_len]`.

    All three modes route through here, because all three are this one
    construction under a different key and flag (spec section 2.3), so a
    recognizing emitter implements the hash once rather than once per mode.
    BLAKE3 is a tree of chained compressions rather than a straight-line body, so
    it takes a name-routed marker (exempt from the generic single-kernel rule,
    the way `hash_frx.perm.poseidon2` and `hash_frx.digest.sha256` are); with no emitter
    wired the composite inlines its decomposition and the bytes are unchanged.

    **The operand ABI**, positional, and the whole of what an emitter reads:

    0. `msg`       — uint8 `[B, L]`, `B` equal-length messages. `L` is static, so
                     the chunk count, the tree shape, the block count and every
                     block length are shape constants rather than data a kernel
                     reads. The trailing block of a chunk is zero-padded to 64
                     bytes and its true byte count reaches the compression as
                     `block_len` (spec section 2.4).
    1. `key_words` — uint32 `[8]`, little-endian: what every chunk and every
                     parent opens from in place of a child's chaining value. The
                     IV in hash mode, the caller's 32 bytes under `KEYED_HASH`,
                     the context pass's digest under `DERIVE_KEY_MATERIAL`. One
                     key serves the whole batch, broadcast across the rows.
    2. `iv`        — uint32 `[8]`, the spec IV (section 2.2, Table 1), whose
                     first four words open every compression's third state row in
                     every mode. An operand rather than a constant the body
                     builds, because a `lax.composite` lifts such a constant into
                     an operand *ahead* of the explicit ones, one per call site —
                     which would make the ABI a function of the message length.

    **The attributes.** `out_len`, the output byte count (32 for a digest), and
    `flags`, the mode flag: one of `0`, `KEYED_HASH`, `DERIVE_KEY_CONTEXT`,
    `DERIVE_KEY_MATERIAL`, never a mask. The flags a node's own position carries
    — `CHUNK_START`, `CHUNK_END`, `PARENT`, `ROOT` — belong to the hash rather
    than the caller, so the emitter derives them and they never appear here.
    Under `non_root=True` a third attribute `non_root = 1` rides.

    **The result.** uint8 `[B, out_len]`: the root node's extendable output read
    from the start (spec section 2.6), which at `out_len = 32` is the standard
    digest. Under `non_root=True` it is instead the final node's 32-byte
    chaining value — the same compression without `ROOT`, which is what a
    Merkle tree built on BLAKE3's own tree semantics (leaf =
    `finalize_non_root`) commits to. A chaining value is one 256-bit value, not
    a stream, so `out_len` must be 32: extending output is the `ROOT`
    compression's mechanism, which non-root finalization by definition never
    runs.

    Nothing else varies. The counter is the chunk index on a chunk's blocks and
    zero on a parent, with a zero high half (a static shape cannot reach 2^32 of
    either); on the root it is the output-block index. The tree pairs adjacent
    nodes from the bottom and carries an odd trailing node up unpaired, which is
    the spec's recursion for every chunk count — `tree_output` argues that.
    """

    if non_root:
        if out_len != DIGEST_LEN:
            raise ValueError(
                f"a chaining value is exactly {DIGEST_LEN} bytes, got out_len={out_len}"
            )

        def cv_decomposition(
            message: Array, key: Array, iv: Array, **_attrs: object
        ) -> Array:
            return unmarked_non_root_hash(message, Mode(key, flags, iv))

        return fused_region(
            cv_decomposition,
            msg,
            key_words,
            IV_WORDS,
            name=BLAKE3_MARKER,
            version=BLAKE3_MARKER_VERSION,
            out_len=out_len,
            flags=flags,
            non_root=1,
        )

    def decomposition(message: Array, key: Array, iv: Array, **_attrs: object) -> Array:
        return unmarked_hash(message, Mode(key, flags, iv), out_len)

    return fused_region(
        decomposition,
        msg,
        key_words,
        IV_WORDS,
        name=BLAKE3_MARKER,
        version=BLAKE3_MARKER_VERSION,
        out_len=out_len,
        flags=flags,
    )


@partial(frx.jit, inline=True, static_argnames=("flags",))
def parent_hash(pairs: Array, key_words: Array, *, flags: int) -> Array:
    """One non-root `PARENT` compression as the `hash_frx.compress.blake3_parent`
    composite: uint8 `[B, 64]` -> uint8 `[B, 32]`.

    A Merkle tree built on BLAKE3's tree semantics hashes its leaves through
    `tree_hash(non_root=True)` and then compresses pairs up the levels through
    here, so an emitter that recognizes both fuses a whole level either side of
    the leaf boundary. Its own marker rather than an attribute on that one, for
    the reason `BLAKE3_PARENT_MARKER` states. The jit zone is here for
    `tree_hash`'s reason: a commit emits one of these per internal node, and an
    uncached re-trace per emission would dominate the first-trace floor.

    **The operand ABI**, positional, and the whole of what an emitter reads:

    0. `pairs`     — uint8 `[B, 64]`, `B` parent nodes, each the left child's
                     32-byte chaining value followed by the right child's
    1. `key_words` — uint32 `[8]`, what a parent opens from in place of a
                     child's chaining value. Same operand, same meaning, and
                     same broadcast across the batch as on a message hash.
    2. `iv`        — uint32 `[8]`, the spec IV, an operand for the reason
                     `tree_hash` states.

    **The attributes.** `non_root = 1` selects the finalization, which is
    `chaining_value` rather than `root_bytes` — carried rather than implied
    because the root of a Merkle tree is the same construction finished WITH
    `ROOT`, which no consumer needs yet but which this marker would express by
    dropping the attribute. `out_len` is 32 and `flags` is the mode, read
    exactly as on a message hash so an emitter shares one attribute reader. The
    positional flags a node's own role carries — `PARENT` here — stay the
    emitter's to derive, so `flags` remains a mode and never a mask.

    **The result.** uint8 `[B, 32]`: the parent's chaining value, what it
    contributes to the level above.
    """

    def decomposition(block: Array, key: Array, iv: Array, **_attrs: object) -> Array:
        return unmarked_parent_hash(block, Mode(key, flags, iv))

    return fused_region(
        decomposition,
        pairs,
        key_words,
        IV_WORDS,
        name=BLAKE3_PARENT_MARKER,
        version=BLAKE3_PARENT_MARKER_VERSION,
        out_len=DIGEST_LEN,
        flags=flags,
        non_root=1,
    )


def _marked_hash(
    mode: Mode, msg: ArrayLike, out_len: int, non_root: bool = False
) -> Array:
    """`tree_hash` from a `Mode`: the one place a mode is taken apart for the
    marker, so which of its fields is an operand and which is an attribute is
    stated once rather than at each of the entry points."""
    return tree_hash(
        device_message(msg),
        mode.key_words,
        out_len=out_len,
        flags=mode.flags,
        non_root=non_root,
    )


def digest(msg: ArrayLike) -> Array:
    """BLAKE3 of a batch of equal-length messages: uint8 `[B, L]` -> `[B, 32]`.

    Byte-identical to the standard per message, at any length. `msg` may be a
    tracer, so a consumer can hash inside its own `@jit` or `vmap`.

    The 32 bytes are the head of the root's extendable output rather than a
    different computation, which is the standard's own construction and why this
    is `root_bytes` at one block rather than a path of its own.
    """
    return xof(msg, DIGEST_LEN)


def xof(msg: ArrayLike, out_len: int) -> Array:
    """BLAKE3 read out to `out_len` bytes: uint8 `[B, L]` -> `[B, out_len]`.

    Byte-identical to the standard per message, at any input length and any
    output length. `digest` is this at 32, which is the standard's own
    construction rather than a shortcut — the digest is the head of the stream.
    """
    return _marked_hash(hash_mode(), msg, out_len)


def non_root_digest(msg: ArrayLike) -> Array:
    """The chaining value of a batch of messages: uint8 `[B, L]` -> `[B, 32]`.

    The final node of `msg`'s tree finished WITHOUT `ROOT` — BLAKE3's
    `finalize_non_root`, the value a node contributes when a tree continues
    ABOVE it. A Merkle tree built on BLAKE3's own tree semantics (flock-
    challenge's, `blake3::hazmat::merge_subtrees_non_root`'s) commits to leaf
    chaining values, and this entry is that leaf hash as one marked region, so
    an emitter can fuse a whole leaf level.

    Hash mode only: the known consumers key nothing, and the seam grows a mode
    when a consumer has to make the choice and cannot.
    """
    return _marked_hash(hash_mode(), msg, DIGEST_LEN, non_root=True)


def parent_digest(pairs: ArrayLike) -> Array:
    """A Merkle parent's chaining value: uint8 `[B, 64]` -> `[B, 32]`.

    `non_root_digest`'s partner one level up. That entry hashes the leaves of a
    Merkle tree built on BLAKE3's own semantics; this compresses each pair of
    child chaining values into the node above it, which is BLAKE3's
    `merge_subtrees_non_root` — so a whole parent level is one marked region
    rather than a traced compression per node.

    Hash mode only, for `non_root_digest`'s reason: the known consumers key
    nothing, and the seam grows a mode when a consumer has to make the choice
    and cannot.
    """
    mode = hash_mode()
    return parent_hash(pair_bytes(pairs), mode.key_words, flags=mode.flags)


def keyed_digest(key: ArrayLike | bytes, msg: ArrayLike) -> Array:
    """Keyed BLAKE3: uint8 `[B, L]` under a 32-byte key -> `[B, 32]`.

    Byte-identical to the standard's `keyed_hash` per message. `digest` is this
    with the IV as the key and the mode flag dropped, which is the only
    difference between the two — a keyed hash of a message shares no compression
    with the unkeyed one.
    """
    return keyed_xof(key, msg, DIGEST_LEN)


def keyed_xof(key: ArrayLike | bytes, msg: ArrayLike, out_len: int) -> Array:
    """Keyed BLAKE3 read out to `out_len` bytes: uint8 `[B, L]` -> `[B, out_len]`.

    The stream is extendable in keyed mode for the reason it is in hash mode —
    the mode reaches the node, and reading it further is the root's own
    compression repeated (spec section 2.6).
    """
    return _marked_hash(keyed_mode(key), msg, out_len)


def derive_key(
    context: str | bytes, key_material: ArrayLike, out_len: int = DIGEST_LEN
) -> Array:
    """BLAKE3's KDF: `out_len` bytes derived from `key_material` under `context`.

    context      : the domain separator, hashed once into the key the material
                   pass opens from. The standard asks for a hardcoded, globally
                   unique UTF-8 string — application name, date, purpose — so
                   that two uses of one key material cannot collide.
    key_material : uint8 `[B, L]`, the secret being derived from; it may be a
                   tracer, and it is the *message*, never the key

    Byte-identical to the standard's `derive_key` per row. Which of the two
    arguments is hashed as what is the whole of this mode's fragility: the
    context is the domain and the material is the message, and swapping them
    derives well-formed bytes of the wrong thing.

    The default length is here rather than in a `derive_key_digest` sibling
    because there is nothing for such a name to distinguish: `digest`/`xof` and
    `keyed_digest`/`keyed_xof` are pairs only so that each mode has one spelling
    of "32 bytes", and this mode's is the default.
    """
    return _marked_hash(derive_key_mode(context), key_material, out_len)


# Whether the pinned Fractalyze XLA plugin ships the BLAKE3 emitter, and on
# which backends. fractalyze/xla#499 registers the recognizer and rewriter on
# the GPU compiler and #507 on the CPU one, so this carries both legs, unlike
# `poseidon.sparse`'s GPU-only tuple; the pin floor in `pyproject.toml` is already above
# the wheel that first shipped them. Family-wide rationale for the two-flag
# shape in `keccak.permutation`; a backend absent from the tuple — Metal today
# — still emits the marker (an unrecognized name inlines byte-neutrally, and
# there is no per-node routing alternative for a whole-tree digest).
_DEDICATED_EMITTER_AVAILABLE = True
_EMITTER_BACKENDS = ("cpu", "gpu")


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry this emitter
    (`fusion.routing`, which carries the rationale)."""
    return routing(_DEDICATED_EMITTER_AVAILABLE, _EMITTER_BACKENDS)


class _Blake3Hash(DeviceRow):
    """The shared body of the three modes — everything but which mode it is.

    A subclass supplies the row: `_read`, which of this module's mode functions
    reads a message, and `_parameters`, what the mode's own parameters are.
    `digest` stays here and forwards, so the seam's name and signature are
    written once however many rows there are.

    `_parameters` is not bookkeeping — `__eq__` covers whatever it returns, so a
    row that adds a key and forgets to name it there compares two different keys
    equal, and serves one key's trace for another as pytree aux. It never errors.
    """

    def __init__(self, output_size: int = DIGEST_LEN) -> None:
        if output_size < 1:
            raise ValueError(f"output_size must be at least 1, got {output_size}")
        self.digest_size = output_size
        super().__init__(FusionPath.from_routing(_routes_to_dedicated_emitter()))

    def _read(self, msg: ArrayLike) -> Array:
        raise NotImplementedError

    def digest(self, msg: ArrayLike) -> Array:
        return self._read(msg)


class Blake3(_Blake3Hash):
    """`ByteHash` for BLAKE3 in hash mode, read out to `output_size` bytes.

    The standard fixes the key words, the chunk size and the tree shape, so the
    length is the only thing left for a caller to choose — and it is what two
    instances compare on.
    """

    def _read(self, msg: ArrayLike) -> Array:
        return xof(msg, self.digest_size)


class Blake3Keyed(_Blake3Hash):
    """`ByteHash` for keyed BLAKE3 — the mode a PRF consumer reaches for.

    The key is a 32-byte `bytes` rather than an array, because the seam has
    nowhere to put a per-call one: `digest(msg)` takes a message and nothing
    else, so the key is part of *which hash this is*, and pytree aux compares it
    by value. Two consequences a caller should choose deliberately rather than
    discover:

    - **A new key is a new trace**, and the key rides in the compiled program's
      constant pool. For a per-call key — a fresh signing key per signature —
      call `keyed_xof` directly, where the key is an operand and one
      compiled program serves every key.
    - **It is secret material held in a plain attribute.** Nothing here erases
      it, and `__hash__` is over the bytes.
    """

    def __init__(self, key: bytes, output_size: int = DIGEST_LEN) -> None:
        if len(key) != KEY_LEN:
            raise ValueError(f"key must be {KEY_LEN} bytes, got {len(key)}")
        super().__init__(output_size)
        self._key = bytes(key)

    def _read(self, msg: ArrayLike) -> Array:
        return keyed_xof(self._key, msg, self.digest_size)

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

    def __init__(self, context: str | bytes, output_size: int = DIGEST_LEN) -> None:
        super().__init__(output_size)
        self._context = context_bytes(context)

    def _read(self, msg: ArrayLike) -> Array:
        return derive_key(self._context, msg, self.digest_size)

    def _parameters(self) -> tuple[object, ...]:
        return (*super()._parameters(), self._context)


class _HostBlake3Hash(HostRow):
    """The shared body of the three host siblings — `blake3` per message.

    The same `_read` / `_parameters` split as `_Blake3Hash`, and for the same
    reason: a row that adds a key and forgets to name it in `_parameters` compares
    two different keys equal and hands one key's trace to the other's caller. The
    Keccak host base compares on `digest_size` alone, which is why it is not the
    shape these rows copy — two of the three carry a parameter beyond the length.

    The loop `_read` runs under is [`byte_hash.host_digest`](../byte_hash.py),
    shared with every other host row in the package.
    """

    def __init__(self, output_size: int = DIGEST_LEN) -> None:
        if output_size < 1:
            raise ValueError(f"output_size must be at least 1, got {output_size}")
        self.digest_size = output_size

    def _read(self, msg: ReadableBuffer) -> bytes:
        raise NotImplementedError

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(self._read, self.digest_size, msg)


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

    def __init__(self, key: bytes, output_size: int = DIGEST_LEN) -> None:
        if len(key) != KEY_LEN:
            raise ValueError(f"key must be {KEY_LEN} bytes, got {len(key)}")
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
    narrowing rather than a style difference.** `blake3_py.blake3(...,
    derive_key_context=...)` takes only a `str`, while the device row hashes
    whatever bytes `context_bytes` produced — so a context that is not
    valid UTF-8 is a hash the device row can compute and this one cannot. It is
    refused at construction, where the caller can still choose another context,
    rather than at the first `digest`. The standard asks for a UTF-8 context
    string, so nothing that follows it can hit this.

    A `str` context and its UTF-8 bytes remain one hash, the same as on the device
    row: both normalize to the same `str` and so compare equal.
    """

    def __init__(self, context: str | bytes, output_size: int = DIGEST_LEN) -> None:
        super().__init__(output_size)
        try:
            self._context = context_bytes(context).decode()
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

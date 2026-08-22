# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The ByteHash seam — the byte sibling of `Permutation`.

A byte hash maps a batch of equal-length byte messages to fixed-size digests:
`digest(uint8[B, L]) -> uint8[B, digest_size]`, byte-identical to the hash's
standard (SHA-256 = FIPS 180-4). Consumers — a byte Fiat-Shamir transcript, byte
Merkle leaves, proof-of-work grinding — read `digest_size` and call `digest`;
they never name a concrete hash. `Sha256` is one implementation; any other byte
hash drops into the same seam unchanged, its internal construction hidden behind
`digest`: SHA-256 is Merkle-Damgard, BLAKE3 a Merkle tree, Keccak a sponge — the
seam abstracts over all three because `digest` is the only common surface. A
shared *streaming* interface would not generalize: the midstate shape differs per
construction.

This is the byte counterpart of `permutation.Permutation`: where a `Permutation`
backs the algebraic sponge / compression / duplex transcript over a field dtype,
a `ByteHash` backs byte hashing over raw bytes. The two `fusion_path` attributes
mean the same thing — how one call lowers on this backend — with one difference:
a `ByteHash` may be `HOST` (a native library looped per message), a state a
`Permutation` never has, so this is the seam where all three states of
`hash_frx.fusion.FusionPath` are live.

Implementations define value-based `__eq__`/`__hash__` over their full parameter
surface — the same rule `Permutation` carries. A host byte transcript is a
`bytes` buffer rather than a jit-traced pytree, so it does not depend on this;
but the moment a `ByteHash` is carried as pytree aux (e.g. a byte Merkle threaded
through `@jit`), identity equality would silently re-trace the enclosing zone on
every freshly built instance. Defining it is cheap and keeps the seam
re-trace-safe by construction (a param-free hash compares by type).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.fusion import FusionPath

if TYPE_CHECKING:
    # Typeshed-only: what every host hash accepts, and what numpy's stub declines
    # to say `ndarray` is. TYPE_CHECKING-guarded, so it is a name mypy resolves
    # and never an import at runtime.
    from _typeshed import ReadableBuffer


@runtime_checkable
class ByteHash(Protocol):
    digest_size: int
    # How `digest` lowers on the backend this instance was built for.
    # `DEDICATED` — a hash-dedicated marker the pinned plugin routes to one
    # kernel; consumers gate device-fusion wrapping on
    # `fusion_path.is_one_kernel` without naming a concrete hash. `GENERIC` — a
    # device path whose marker this backend does not route (it inlines; right
    # bytes, no kernel), still traceable. `HOST` — a native library looped per
    # message; never traceable, and `digest`'s return type says so. Device rows
    # derive it per (hash, backend) at construction; a host row is the one
    # legitimate constant (`HOST` everywhere).
    fusion_path: FusionPath

    def digest(self, msg: ArrayLike) -> Array | np.ndarray:
        """Hash a batch of equal-length messages: uint8 `[B, L]` -> uint8
        `[B, digest_size]`, big-endian (the hash's standard output order). The
        result is a device `Array` (a marker hash) or a host `np.ndarray` (a
        host-library hash); consumers `np.asarray` it to bytes.

        One call is one function — the unit that lowers to one fused kernel on
        the `DEDICATED` path. `L` is static, so any padding is data-independent.
        Batch with the `B` axis: a dedicated-fusion hash lowers the whole batch
        through one shared decomposition (Merkle leaves, a PoW nonce window).

        **Whether it may be called inside a traced region is the return type.**
        An implementation returning a device `Array` accepts a traced `msg`, so a
        consumer can hash inside its own `@jit` or `vmap` — which is what lets a
        scheme reach the hash through this seam rather than naming a concrete one
        to get a traceable path. An implementation returning `np.ndarray` is a
        host call and never can: it has to read the bytes.

        `fusion_path` states the same split declaratively:
        `fusion_path.is_traceable` agrees with the return type by construction
        (a device row — `DEDICATED` or `GENERIC` — returns `Array`; a `HOST` row
        returns `np.ndarray`), and the return type remains the authority the
        attribute is held to. The lone bool this seam used to carry could not
        say it — a device hash whose marker the pinned plugin does not route
        (BLAKE3 on Metal) and a host hash both read `False` — which is what the
        three-state `FusionPath` exists to keep apart.

        Which hash sits in the `GENERIC` gap moves with the pin — Keccak was
        there until its emitter shipped — so the reason the return type stays
        the authority does not move with it.
        """
        ...


class Row:
    """What every `ByteHash` row repeats: the equality contract.

    A row's `__eq__`/`__hash__` is its **jit cache key** — two instances that
    compare equal share a trace, two that do not each get their own — so getting
    it wrong in the lenient direction serves one parameterization's compiled
    executable for another's, silently and with the right shape. The contract
    was written out thirty-two times in three different spellings (`return
    True`, a `digest_size` comparison, and `_parameters()`), which is three
    chances to get a jit cache key wrong.

    `_parameters` is the one thing a row overrides. It defaults to empty, which
    is right for a parameterless row and wrong the moment a row gains a
    parameter and forgets to name it here — so it is worth stating that the
    forgetting does not error: it compares two different keys equal and serves
    one key's trace for the other. `row_conformance_test` builds every
    parameterized row twice, with different parameters, and requires them to
    differ.
    """

    digest_size: int
    fusion_path: FusionPath

    def _parameters(self) -> tuple[object, ...]:
        """Everything two instances of this row compare on, beyond their type."""
        return ()

    def __eq__(self, other: object) -> bool:
        # `type(other) is not type(self)` rather than `isinstance`: isinstance is
        # asymmetric under subclassing, and returning False instead of
        # NotImplemented would block Python's reflected-`__eq__` fallback.
        if type(other) is not type(self):
            return NotImplemented
        return self._parameters() == other._parameters()

    def __hash__(self) -> int:
        return hash((type(self), self._parameters()))


class DeviceRow(Row):
    """A row whose `digest` returns a device `Array`.

    `fusion_path` is derived per instance from the family's routing gate, never
    pinned on the class: the emitter switch is a property of the pin and the
    backend, and a value read at import would fix the answer before anything
    could vary it. The gate arrives as a callable rather than a bool so it is
    read at construction — and so the module attribute stays the seam the
    family's own tests patch.
    """

    def __init__(self, routes: Callable[[], bool]) -> None:
        self.fusion_path = FusionPath.from_routing(routes())


class HostRow(Row):
    """A row backed by a host library: `digest` returns `np.ndarray` and can
    never take a tracer, because it reads the message bytes.

    `fusion_path` is the one legitimate class constant on this seam — a host
    path is `HOST` on every backend.
    """

    fusion_path = FusionPath.HOST


def device_message(msg: ArrayLike) -> Array:
    """A message as the uint8 `[B, L]` batch every device row hashes.

    The rank is checked at the seam so a caller holding the wrong one is told
    so eagerly, rather than from inside the trace of a marked region — where it
    surfaces as a reshape or concatenate error naming neither the seam nor the
    rank. A 1-D message is the common miss: a single message is `B = 1`, not a
    bare `[L]`.

    Checked *before* the conversion, so a wrong rank never reaches a device and
    the check itself needs no backend — which is what lets the seam's own test
    stay substrate-free, as a seam test must. `np.ndim` reads `.ndim` where
    there is one (an array, a tracer) and only falls back to converting for a
    plain sequence, so this holds under `jit` too.
    """
    if np.ndim(msg) != 2:
        raise ValueError(f"msg must be 2-D uint8 [B, L], got ndim={np.ndim(msg)}")
    return fnp.asarray(msg, dtype=fnp.uint8)


def host_digest(
    hash_one: Callable[[ReadableBuffer], bytes], digest_size: int, msg: ArrayLike
) -> np.ndarray:
    """The body every host implementation of this seam shares: `hash_one` per
    message. uint8 `[B, L]` -> uint8 `[B, digest_size]`.

    A host row is a loop over a one-message hash, and the only thing that differs
    between rows is which hash and how many bytes it reads out — so the loop lives
    here and a row is its `hash_one` plus its `digest_size`. `hash_one` closes over
    whatever the row's mode needs (a key, a context, an output length), which is
    what lets one body serve SHA-256, the Keccak family and BLAKE3's three modes.

    **The row reaches `hash_one` as the array, not as `bytes`.** The
    `ascontiguousarray` below already guarantees a contiguous buffer and every
    hash a row can be built on takes the buffer protocol, so a `tobytes()` per row
    would copy the message a second time to no end.

    What that is worth is small, and stated here rather than assumed, because
    routing through a shared body costs one Python call per row that an inlined
    loop did not — the copy saved and the call added are the same order. Measured
    on `HostSha256` at 256 rows against an inlined body that copies: a wash at
    short messages (within 1% at 64-300 B), 2-5% faster from 1 KiB up. This
    extraction is for the duplication; the speed is a rounding error either way.

    Handing over `row.data` — the row's `memoryview` — is the same idea and is
    *slower* below about 4 KiB, since building the view costs more than copying a
    short message. Passing the array is both the faster form and the shorter one.

    This is a host call by construction: it reads the message bytes, so `msg` can
    never be a tracer. That is the seam's return-type rule above, and returning
    `np.ndarray` is what states it.
    """
    rows = np.ascontiguousarray(np.asarray(msg, dtype=np.uint8))  # [B, L]
    out = np.empty((rows.shape[0], digest_size), dtype=np.uint8)
    for i, row in enumerate(rows):
        # `ndarray` implements the buffer protocol; numpy's stub does not
        # declare it, and this is the one place that has to say so.
        out[i] = np.frombuffer(hash_one(cast("ReadableBuffer", row)), dtype=np.uint8)
    return out

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
mean the same thing — how one call lowers on this backend.

Implementations define value-based `__eq__`/`__hash__` over their full parameter
surface — the same rule `Permutation` carries. A host byte transcript is a
`bytes` buffer rather than a jit-traced pytree, so it does not depend on this;
but the moment a `ByteHash` is carried as pytree aux (e.g. a byte Merkle threaded
through `@jit`), identity equality would silently re-trace the enclosing zone on
every freshly built instance. Defining it is cheap and keeps the seam
re-trace-safe by construction (a param-free hash compares by type).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.fusion import FusionPath


@runtime_checkable
class ByteHash(Protocol):
    digest_size: int
    # How `digest` lowers on the backend this instance was built for.
    # `DEDICATED` — a hash-dedicated marker the pinned plugin routes to one
    # kernel; consumers gate device-fusion wrapping on
    # `fusion_path.is_one_kernel` without naming a concrete hash. `GENERIC` — a
    # device path whose marker this backend does not route (it inlines; right
    # bytes, no kernel). Derived per (hash, backend) at construction, never a
    # class constant: the emitter switch is a property of the pin and the
    # backend.
    fusion_path: FusionPath

    # Supply it as a stored attribute — `DeviceRow.__init__`'s assignment —
    # and not as a read-only `@property`. Declared
    # here as a mutable attribute, which a property does not satisfy, so a row
    # that delegates through one stops being a `ByteHash` at all. The seam
    # conformance pin every implementation module carries
    # (docs/reference/conventions.md) is what catches it; `adapter/mgf1.py`
    # shipped a delegating property for as long as it was the one module
    # without one.

    def digest(self, msg: ArrayLike) -> Array:
        """Hash a batch of equal-length messages: uint8 `[B, L]` -> uint8
        `[B, digest_size]`, big-endian (the hash's standard output order). The
        result is a device `Array`; consumers `np.asarray` it to bytes.

        One call is one function — the unit that lowers to one fused kernel on
        the `DEDICATED` path. `L` is static, so any padding is data-independent.
        Batch with the `B` axis: a dedicated-fusion hash lowers the whole batch
        through one shared decomposition (Merkle leaves, a PoW nonce window).

        **Every implementation returns an `Array`**, so a consumer can hash
        inside its own `@jit` or `vmap` — which is what lets a scheme reach the
        hash through this seam rather than naming a concrete one to get a
        traceable path.

        `fusion_path` reports routing and nothing else: whether the marker
        lowers to one kernel on this backend. Which hash sits in the `GENERIC`
        gap moves with the pin — Keccak was there until its emitter shipped.
        """
        ...


class Row:
    """What every `ByteHash` row repeats: the equality contract.

    A row's `__eq__`/`__hash__` is its **jit cache key**. Two instances that
    compare equal share a trace; two that do not each get their own. Wrong in
    the lenient direction, one parameterization's compiled executable is served
    for another's — silently, with the right shape. It was written out
    thirty-three times in three spellings (`return True`, a `digest_size`
    comparison, and `_parameters()`), so it is written here instead.

    `_parameters` is the one thing a row overrides, and it defaults to the
    output length because that is what all but the keyed rows key on. A row
    that gains a further parameter and forgets to name it here does not error:
    it compares two different keys equal. `row_conformance_test` builds every
    parameterized row twice, with different parameters, and requires them to
    differ.
    """

    # `fusion_path` is deliberately NOT declared here: `DeviceRow`
    # supplies its own, and leaving it off is what lets an adapter that has
    # no fusion path at all — `Hmac` — share the equality contract rather than
    # keep a thirty-third copy of it.
    digest_size: int

    def _parameters(self) -> tuple[object, ...]:
        """Everything two instances of this row compare on, beyond their type.

        The output length is part of the hash rather than a formatting choice —
        BLAKE2 folds it into the initial state, the Keccak rows read it off a
        different rate — so two lengths are two hashes. On a fixed-output row it
        is a class constant and adds nothing the type did not already say.
        """
        return (self.digest_size,)

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
    """A row whose one hashing call returns a device `Array`.

    Usually that call is `digest` and the row is a `ByteHash`. It need not be:
    `keccak/tuple_hash.py` hashes a *sequence* of strings, so its call is
    `hash(strings)` and it deliberately implements no seam — but it lowers
    through the same sponge as every other Keccak row and so has a real
    `fusion_path` to report, which is what this base is for. A construction
    with no path at all stays a plain `Row` (`adapter/hmac.py`).

    Takes the resolved `FusionPath` rather than a routing gate, because a row
    has more than one honest way to reach one: most read their family's
    pin-and-backend gate (`FusionPath.from_routing(...)`), while the Keccak rows
    derive theirs from `KeccakF1600` so it cannot disagree with the routing
    `KeccakSponge.hash` actually takes. Both go through this door.

    It is passed in rather than read here, and never pinned on the class,
    because the emitter switch is a property of the pin and the backend: a value
    resolved at import would fix the answer before anything could vary it.
    """

    def __init__(self, fusion_path: FusionPath) -> None:
        self.fusion_path = fusion_path


def _require_batch_rank(msg: ArrayLike) -> None:
    """Reject anything that is not the seam's uint8 `[B, L]` batch.

    Both front doors call it — `device_message` before converting and
    `message_length` before reading the width — so the two cannot drift, and
    neither can the message: at least six tests match it by regex.

    A 1-D message is the common miss: a single message is `B = 1`, not a bare
    `[L]`. Checked before any conversion, so a wrong rank never reaches a device
    and the check itself needs no backend — which is what lets the seam's own
    test stay substrate-free, as a seam test must. `np.ndim` reads `.ndim` where
    there is one (an array, a tracer) and only falls back to converting for a
    plain sequence, so this holds under `jit` too.
    """
    if np.ndim(msg) != 2:
        raise ValueError(f"msg must be 2-D uint8 [B, L], got ndim={np.ndim(msg)}")


def device_message(msg: ArrayLike) -> Array:
    """A message as the uint8 `[B, L]` batch every device row hashes.

    The rank is checked before the conversion (`_require_batch_rank`, which
    carries the reasoning), so a wrong rank never reaches a device.
    """
    _require_batch_rank(msg)
    return fnp.asarray(msg, dtype=fnp.uint8)


def message_length(msg: ArrayLike) -> int:
    """The `L` of a uint8 `[B, L]` batch, read without converting the message.

    The rank check runs first (`_require_batch_rank`, which carries the
    reasoning), so a wrong rank surfaces here rather than as a confusing unpack.
    `np.shape` reads `.shape` where there is one — an array, a tracer — so this
    holds under `jit` too, and a caller that needs the length *before* deciding
    what to convert the message into does not have to reach for `.shape` itself.
    """
    _require_batch_rank(msg)
    return int(np.shape(msg)[-1])


def padded_batch(msg: Array, tail: Array) -> Array:
    """`msg` with its padding appended: uint8 `[B, L]` + `[T]` -> `[B, L + T]`.

    The tail is one row — it is a function of the message LENGTH, which every
    row of a batch shares — so it is broadcast rather than built per row. (The
    concatenate still materializes the `[B, L + T]` result; what the broadcast
    avoids is a second per-row copy of the tail.)

    Shared by both extensions rather than living with either. Ten files built
    this concatenate: the seven Merkle-Damgard families, both byte sponges, and
    BLAKE3's chunk padding.
    It is not an MD step and not a sponge step; it is the last thing that
    happens to a message before whichever schedule reads it, which is why it
    sits next to `device_message` on the seam.

    **There is no head-side twin, and that is a decision.** A
    `framed_batch(head, msg, tail)` looks earned — the prepend is written out in
    several families — but the sites do not share a shape: the SP 800-185 heads
    are host `bytes` concatenated OUTSIDE a marker, while Ascon-CXOF's are 1-D
    `Array` operands broadcast INSIDE a marked body, where op count is a
    correctness property and `ascon.py` declines the helper in its own comment
    for exactly that reason. KMAC's operand-key path puts a `[B, K]` key and a
    conditional zero-fill between head and message, and TupleHash's arity is a
    call parameter. One signature over that would encode the accidents of
    whichever site it was shaped against. What the SP 800-185 layer does share
    is the bytes-to-row step, which lives in `keccak/cshake.py::const_rows`.

    What each caller does NEXT is genuinely its own: SHA-2 and SM3 pack the
    result big-endian, RIPEMD-160 and BLAKE2 little-endian, and the sponges read
    lanes. Only the append is common, so only the append moved.
    """
    return fnp.concatenate(
        [msg, fnp.broadcast_to(tail, (msg.shape[0], tail.shape[0]))], axis=-1
    )


def capacity(msg: ArrayLike, block_size: int) -> int:
    """The buffer width a runtime-length marker hashes `msg` out of.

    A message still on the HOST is widened to the next power of two, floored at
    one block. The width is what compilation is keyed on, so the policy trades
    how many distinct buffers a caller compiles against how many padding bytes
    it ships per call: doubling bounds the second under 2x the message while
    keeping the first logarithmic — fifteen widths span a byte to a megabyte.
    Widening it costs a numpy copy, and collapsing a compile per length into one
    per width is the whole point of the form.

    A message ALREADY on the device keeps its own extent, so nothing is padded
    and only the marker changes. Widening it would be a dispatched device op
    rather than a host copy — measured at 4.2x the digest it precedes (B = 256,
    L = 65) — and it buys that caller nothing: a batch materialized at one length
    is not re-entering the trace cache with new ones, and under `jit` the
    enclosing trace compiles once whatever the width. So the capacity is
    whatever buffer the caller already has.

    Coarser widths are not ruled out by the kernel: measured on the pinned wheel,
    a 32-byte message costs the same (~6 us at B = 256) in a 32-byte buffer as in
    a 2048-byte one, because the emitter loops on the length operand — where
    hashing the whole buffer instead would have cost 20x. What bounds the width
    is the bytes crossing to the device, which a short message does not amortize.

    `block_size` is the family's, passed rather than read off a `PadRule`: the
    rule lives in `extension/pad.py`, which the extensions import from this seam
    and not the other way round. It is also the only thing about the family the
    policy needs — the floor is one block, and nothing else here is per-family.
    """
    length = message_length(msg)
    if isinstance(msg, Array):
        # `LMAX >= 1` is the recognizer's floor: the emitter's clamped
        # message-side index needs somewhere in bounds to land, even at `len` 0.
        return max(length, 1)
    return block_size if length <= block_size else 1 << (length - 1).bit_length()


def at_capacity(msg: ArrayLike, width: int) -> Array:
    """`msg` widened to the uint8 `[B, width]` buffer the marker hashes out of:
    `[B, L]` -> `[B, width]`, for `width >= L`.

    The width is a CAPACITY rather than a length: the marker takes the live byte
    count as an operand, so its emitter stops there and the bytes past `L` are
    never read. They are left zero on that ground — nothing derives from them.

    **Where the widening runs decides whether the compile saving is real.**
    Widening on device is itself an eager op keyed on `L`, so it would trade one
    compile per length for a cheaper compile per length rather than removing one:
    measured at ~22 ms against the ~90 ms whole-digest compile it exists to
    collapse. Host data is therefore widened with numpy, before it reaches a
    device at all — a copy and one transfer, no compile. An input already on
    device cannot take that path without a round trip, and a tracer cannot take
    it at all, so both are widened in-graph: right in either case, and free for
    the tracer, whose enclosing trace compiles once regardless.

    Sits next to `capacity` for the reason `padded_batch` sits next to
    `device_message`: the policy and the widening are only ever called together,
    and this is the last thing that happens to a message before a runtime-length
    marker reads it. `sha256` held both while it was the only family with such a
    marker; Grostl-256 is the second, and four more tail-operand rows follow.
    """
    length = message_length(msg)
    if width < length:
        raise ValueError(f"width ({width}) must be >= the message length ({length})")
    if isinstance(msg, Array):
        # An already-uint8 array is handed back untouched rather than run through
        # a converting call: those return the identical object and still pay a
        # full eager dispatch — `device_message` 5.1 us, `.astype` 2.6 us, against
        # the ~3 us digest they precede. This is the only branch a device-resident
        # caller takes, so that dispatch would be pure overhead on every call.
        u8 = msg if msg.dtype == fnp.uint8 else msg.astype(fnp.uint8)
        if width == length:
            return u8
        return padded_batch(u8, fnp.zeros(width - length, dtype=fnp.uint8))
    host = np.asarray(msg, dtype=np.uint8)
    if width == length:
        return fnp.asarray(host)
    buf = np.zeros((host.shape[0], width), dtype=np.uint8)
    buf[:, :length] = host
    return fnp.asarray(buf)


def require_capacity_buffer(buf: Array) -> None:
    """Reject a zero-width capacity buffer, before a marker is emitted from it.

    `LMAX >= 1` is a term of the runtime-length ABI rather than a detail of any
    one emitter: a recognizer declines a zero-width buffer, which leaves the
    decomposition to run — and that one gathers the message through a clamp with
    nowhere in bounds to land. The empty message is `length = 0` in a buffer of
    at least one byte, never a buffer of none.

    Here rather than in `padded_message_region`, which is where the clamp
    actually is: that function only runs when the marker is DECLINED, so a guard
    inside it would pass silently on exactly the backends that route. The check
    belongs where the buffer is handed to the marker, which is once per family
    and identical every time — it was already written out twice, verbatim down
    to the error string, before this existed.
    """
    if buf.shape[-1] < 1:
        raise ValueError(
            f"buf must be uint8 [B, LMAX >= 1], got width {buf.shape[-1]}: an "
            "empty message is length 0 in a non-empty buffer"
        )

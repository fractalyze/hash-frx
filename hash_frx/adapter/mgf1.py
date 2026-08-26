# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""MGF1 (RFC 8017 Appendix B.2.1) — a hash stretched into a mask.

The mask generation function RSA-OAEP and RSA-PSS are defined over, and an XOF
adapter in this package's sense: it turns a fixed-length `ByteHash` into an
arbitrary-length one by suffixing a counter.

    mask = T_0 ‖ T_1 ‖ …  truncated to `output_size`,  T_i = H(seed ‖ I2OSP(i, 4))

**It is a row, not a function of a length.** `Mgf1(Sha256(), 64)` is a
different hash from `Mgf1(Sha256(), 32)` rather than one hash asked for more
bytes, which is the family-wide rule `keccak/byte_hashes.py` states for
`Shake256` and every other variable-output row here. So the length rides in the
value surface `__eq__`/`__hash__` cover — a jit cache key, per `byte_hash.Row`
— and `Mgf1` is a `ByteHash` a consumer can hand anywhere one is taken.
`hmac.Hmac` is the precedent for an adapter that is a `Row` without being a
`DeviceRow`: what it is depends on the hash underneath.

`mgf1(hash, seed, length)` stays as a one-liner over it, because RFC 8017 spells
the call `MGF(mgfSeed, maskLen)` and a consumer transcribing the RFC should be
able to write it that way. It is **not** re-exported from `hash_frx`: `Mgf1` is
the construction's one public name, and a synonym in the package's export table
would be the second spelling this row exists to avoid. Import it from here when
the RFC's argument order is what you want.

**The blocks do not chain**, which is the whole difference from
[`hkdf.py`](hkdf.py)'s superficially identical shape: HKDF's `T(i)` feeds
`T(i+1)`, so its loop is a dependency chain, while every `T_i` here reads the
same seed and differs only in the counter. That independence is not a curiosity
— it is why this emits **one** digest over a `[B * blocks, S + 4]` message
rather than one per block. Measured on the GPU leg, where a recognized hash is
one kernel per digest: at eight blocks (an RSA-OAEP mask for a 2048-bit
modulus) the loop form emitted eight composites against one and ran 1.53×
slower at batch 1, 2.14× at batch 64; at thirty-two blocks, 4.2× and 6.4×.
Crossover is two blocks, so the loop wins only for a single-block mask.

Batching costs no extra copy, which is the thing that looks like it should and
does not: the loop already materialized the seed once per block, spread across
`blocks` separate concatenates. Tiling gathers those copies into one buffer
rather than adding any — the lowered byte counts are identical either way.

**The counter is a host constant, not a device value.** `output_size` is
static, so every index is known at trace time and `I2OSP(i, 4)` is four bytes
the program carries rather than one it computes. That keeps `seed` an operand —
it may be a tracer, so a consumer can call this inside its own `@jit` — and it
is why nothing here reads a seed byte.

**`hLen` is read off the underlying row.** RFC 8017 fixes the hash per scheme
rather than inside MGF1, so any `ByteHash` works and `Mgf1(Shake256(48), 96)`
means 48-byte `T_i` deliberately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.byte_hash import Row, device_message

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash
    from hash_frx.fusion import FusionPath

# RFC 8017 B.2.1: "mask length is at most 2^32 * hLen". The counter is four
# octets, so past that the sequence would repeat rather than extend.
_MAX_BLOCKS = 1 << 32


def _counter_block(index: int) -> np.ndarray:
    """`I2OSP(index, 4)` — the four-octet big-endian counter block `index` is
    suffixed with (RFC 8017 §4.1).

    A host value: `output_size` is static, so every index is known at trace
    time. Private, and unguarded: `Mgf1` bounds the block count before it loops,
    so an index past four octets is unreachable, and a check here would be a
    branch no caller can take. The width and byte order are pinned through the
    public surface instead — index 256 is reachable at 8224 bytes of mask, and
    a little-endian or two-octet counter changes the message there.
    """
    return np.frombuffer(index.to_bytes(4, "big"), dtype=np.uint8)


class Mgf1(Row):
    """`ByteHash` for MGF1 over `byte_hash`, read out to `output_size` bytes.

    byte_hash : the underlying hash `H`. Any `ByteHash` — a host row works
        eagerly, a device row also inside a consumer's `@jit`.
    output_size : bytes of mask, at most `2^32 * hLen` (RFC 8017 B.2.1).

    `digest(seed)` takes the seam's uint8 `[B, S]` batch and returns
    `[B, output_size]`. The seed is a message like any other — a single one is
    `B = 1`, not a bare `[S]` — which is the seam's own rule and the reason this
    is a row rather than a function that promotes.
    """

    def __init__(self, byte_hash: ByteHash, output_size: int) -> None:
        h_len = byte_hash.digest_size
        if not 1 <= output_size <= _MAX_BLOCKS * h_len:
            raise ValueError(
                f"output_size ({output_size}) must be in [1, 2^32 * hLen = "
                f"{_MAX_BLOCKS * h_len}] (RFC 8017 B.2.1): past that the "
                "four-octet counter would repeat rather than extend the mask"
            )
        self._byte_hash = byte_hash
        self.digest_size = output_size
        # The underlying hash's, because the mask IS that hash's digests.
        #
        # Read at CONSTRUCTION rather than pinned on the class, for the reason
        # `DeviceRow` gives for taking one rather than reading it: the emitter
        # switch is a property of the pin and the backend, and a value fixed at
        # import would answer before anything could vary it. Taking it here is
        # what `DeviceRow.__init__` does, and it costs nothing to be late — the
        # underlying row resolved its own at ITS construction, so there is no
        # later answer to wait for. `Hmac` carries none at all because its
        # region is two digests and a pair of XORs; a mask is one digest call,
        # so this one is honest.
        #
        # A stored attribute and not a `@property`, which the seam requires of
        # every implementation and states once (`byte_hash.ByteHash`).
        self.fusion_path: FusionPath = byte_hash.fusion_path

    def _parameters(self) -> tuple[object, ...]:
        # The underlying hash joins the length: `Mgf1(Sha256(), 32)` and
        # `Mgf1(Sha512(), 32)` are different hashes that agree on output width,
        # and a row that compared them equal would serve one's trace for the
        # other's.
        return (*super()._parameters(), self._byte_hash)

    def digest(self, seed: ArrayLike) -> Array | np.ndarray:
        """The mask: uint8 `[B, S]` -> `[B, output_size]`."""
        seed = device_message(seed)
        batch, seed_len = seed.shape
        h_len = self._byte_hash.digest_size
        blocks = -(-self.digest_size // h_len)

        # One digest over every (row, block) pair. The layout is row-major —
        # message `b`'s blocks are contiguous — so the trailing reshape splits
        # them back per row with no transpose.
        tiled = fnp.broadcast_to(seed[:, None, :], (batch, blocks, seed_len))
        counters = fnp.asarray(np.stack([_counter_block(i) for i in range(blocks)]))
        message = fnp.concatenate(
            [
                tiled.reshape(batch * blocks, seed_len),
                fnp.broadcast_to(counters, (batch, blocks, 4)).reshape(
                    batch * blocks, 4
                ),
            ],
            axis=1,
        )
        stream = fnp.asarray(self._byte_hash.digest(message), dtype=fnp.uint8)
        return stream.reshape(batch, blocks * h_len)[:, : self.digest_size]


def mgf1(byte_hash: ByteHash, seed: ArrayLike, length: int) -> Array | np.ndarray:
    """The MGF1 mask, in RFC 8017's own `MGF(mgfSeed, maskLen)` call shape.

    `Mgf1(byte_hash, length).digest(seed)` — the same construction, spelled the
    way a consumer transcribing the RFC would write it. Where the length varies
    per call this is the one to reach for; where it is fixed, build the row once
    so its jit cache key is stable.
    """
    return Mgf1(byte_hash, length).digest(seed)


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md). Load-bearing rather
    # than ceremonial: an adapter row has no in-tree consumer to fail instead,
    # so this is the only thing holding the class to the protocol.
    _bh_mgf1: type[ByteHash] = Mgf1

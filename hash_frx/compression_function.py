# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The second primitive seam: a compression function, state x block -> state.

`Permutation` covers the n->n primitives. This covers the n->m ones, which is
what every Merkle-Damgard hash in this package actually runs — SHA-256,
SHA-512, RIPEMD-160, SM3, BLAKE2b, BLAKE2s, the Grostl P/Q pair, and BLAKE3's
compression under a tree schedule rather than an MD one.

Until this existed, all eight lived inside their family's digest body with no
shared type, which is why the schedule around them was transcribed per family:
you cannot write the schedule once if the thing it schedules has no name.

**Not to be confused with `hash_frx.compression.Compression`**, which is a
different layer despite the name — the Merkle truncated-permutation adapter,
n chunks -> 1 over a `Permutation`. SHA-256 does not fit through it.

**HAIFA rides here rather than in a subtype.** BLAKE2 feeds its compression a
counter and a finalization flag alongside the block, and BLAKE3 feeds a counter
and mode flags; both are optional inputs to `compress` rather than a separate
seam, because the alternative — one protocol per input shape — would put the
extension in the business of knowing which primitive it has, and knowing that is
exactly what the extension exists not to do.

Implementations MUST define value-based `__eq__`/`__hash__` over their full
parameter surface, for the reason `byte_hash.Row` states: the pair is a jit
cache key. A stateless compression function that carries no parameters may
compare by type.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from frx import Array
from frx.typing import DTypeLike

from hash_frx.fusion import FusionPath


@runtime_checkable
class CompressionFunction(Protocol):
    # Bytes of message consumed per call. The extension's block boundary, and
    # what its padding rule pads to.
    block_size: int
    # Words of chaining state carried between calls. Words, not bytes, because
    # the state is threaded as `dtype` words and only serialized at the end —
    # `digest_size` is the extension's business, since SHA-384 and SHA-512/256
    # truncate the same state to different lengths.
    state_words: int
    # Dtype of a state word: uint32 for SHA-256/RIPEMD-160/SM3/BLAKE2s, and a
    # (lo, hi) uint32 pair for the 64-bit families, which frx has no native
    # 64-bit lane for.
    dtype: DTypeLike
    # How a marked region over this compression lowers on the backend it was
    # built for — the same three states, read the same way, as on the other two
    # seams. Derived per (primitive, backend) at construction.
    fusion_path: FusionPath
    # The composite name + version a marked region carries, for a consumer that
    # needs to RE-MARK a compression inside its own decomposition. One ABI
    # coordinate, so name and version travel together (`hash_frx.fusion`).
    fused_region_marker: tuple[str, int]

    def compress(self, state: Array, block: Array, **extras: Any) -> Array:
        """Absorb one block into the chaining state.

        `state` is `[..., state_words]` over `dtype`, `block` is the message
        block in whatever word shape this primitive reads, and the result is a
        new state of the same shape as `state`. Leading dimensions are batch and
        pass through untouched, which is what lets one marked region cover a
        whole batch of messages.

        `extras` carries the per-call inputs a HAIFA-style construction needs —
        BLAKE2's byte counter and final-block flag, BLAKE3's counter and mode
        flags. A construction that needs none passes none, and an implementation
        that reads none may reject unexpected ones.
        """
        ...

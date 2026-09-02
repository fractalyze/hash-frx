# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Permutation seam every fixed-width primitive builds on.

A fixed-width permutation over a single dtype. Consumers (duplex sponge,
Fiat-Shamir transcript, Merkle compression) read `width` to size state and
`dtype` to allocate it, then call `permute` — they never name a concrete hash.
Poseidon2 is one implementation; any other fixed-width permutation drops in
unchanged.

`dtype` is not required to be a field. `Sponge` and `Compression` only allocate,
index, and overwrite state lanes, so a machine-word permutation — Keccak-f[1600]
over uint32 lane halves, BLAKE3's compression over uint32 — satisfies this
Protocol as written. The one construction that does require an additive group is
`DuplexSponge`, and it states that requirement itself rather than narrowing this
seam for every other consumer's sake.

Implementations MUST define value-based `__eq__`/`__hash__` over their full
parameter surface: a permutation rides pytree aux, where identity equality
silently re-traces the enclosing jit zone on every freshly built instance — a
cost that does not error, it just makes every call slow. A Protocol cannot
enforce this; each implementation carries it. The `ByteHash` seam no longer
works this way — its rows inherit the pair from `byte_hash.Row` — so the five
permutations here are the remaining hand-written copies, and folding them onto a
shared base is open work rather than a settled difference.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from frx import Array
from frx.typing import DTypeLike

from hash_frx.fusion import FusionPath


@runtime_checkable
class Permutation(Protocol):
    width: int  # state length (rate + capacity)
    # Dtype of each state element: a field dtype for the algebraic hashes, a
    # machine word for the bit-oriented ones. `DTypeLike` rather than `Any`
    # because "whatever `fnp.zeros` accepts" is the actual contract, and it is
    # narrow enough for mypy to reject a non-dtype at the pin every
    # implementation carries.
    dtype: DTypeLike
    # Whether a whole-region composite over this permutation is expandable here.
    # `DEDICATED` — the pinned plugin can DRIVE this primitive on this backend,
    # so a vendor can expand an enclosing region (e.g. a Merkle commit, a sponge)
    # from the params the marker carries; consumers gate that wrapping on
    # `fusion_path.is_one_kernel` without naming a concrete hash. `GENERIC` —
    # same bytes, no expandable marker. Derived per (hash, backend) at
    # construction, because the emitter switch is a property of the pin and the
    # backend rather than of the hash.
    #
    # Usually this is the same question as which marker `permute` emits, and
    # most families read it off that (`FusionPath.from_marker`). The two come
    # apart on a backend that can drive the primitive while routing no
    # STANDALONE permute kernel — `poseidon2` on the GPU, where the permute's own
    # loops already lower to one kernel without a dedicated arm. Reading the
    # marker there would report `GENERIC` and cost every envelope its kernel, so
    # such a family answers this one directly.
    fusion_path: FusionPath
    # The composite name + version `permute`'s marker carries — what a consumer
    # needs to RE-MARK a permute inside its own composite decomposition (a duplex
    # absorb chain is one), so that if the enclosing composite inlines, its
    # fallback still runs the dedicated per-permute kernels instead of raw permute
    # bodies. The dedicated kernel is the byte authority, so a raw body standing
    # in for it would change what a fallback computes without failing anything.
    #
    # Name and version travel together because they are one ABI coordinate: a
    # contract change stages through `composite.version` rather than a rename (see
    # `hash_frx.fusion`), so a consumer holding the name alone can re-mark against
    # a stale contract. An undedicated permutation reports the generic marker at
    # version 0, which is what `fusion_path` is read off.
    fused_region_marker: tuple[str, int]

    def permute(self, state: Array) -> Array:
        """Apply the permutation: (width,) over `dtype` -> (width,).

        One call is one function — the unit that lowers to one fused kernel.
        Batch with `frx.vmap(permute)`: the dedicated-fusion marker lowers
        identically batched (one shared decomposition), so no batched twin is
        needed.
        """
        ...

    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """Pieces to wrap a computation over this permutation as ONE `fused_region`
        in its ABI, without the consumer knowing the operand layout. Returns
        ``(operands, permute_from_operands, attrs)``:

        - ``operands`` = ``(leading, *constants)``; the round constants ride as
          explicit operands because the emitter's ABI names them there.
        - ``permute_from_operands(state, *constants)`` = the const-free permute the
          `fused_region` runs.
        - ``attrs`` = identifying `composite.attributes` (a ``permutation``
          discriminator + shape).

        Meaningful only on the dedicated path; a non-fused permutation returns an
        inert spec.
        """
        ...

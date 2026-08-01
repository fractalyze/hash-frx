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
enforce this; each implementation carries it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from frx import Array


@runtime_checkable
class Permutation(Protocol):
    width: int  # state length (rate + capacity)
    # Dtype of each state element: any dtype a consumer can allocate with
    # `fnp.zeros` and index. A field dtype for the algebraic hashes, a machine
    # word for the bit-oriented ones.
    dtype: Any
    # Whether `permute` lowers to a hash-dedicated fusion marker (vs the generic
    # region marker). When true, a vendor can expand a whole-region composite —
    # e.g. a Merkle commit — by reading this hash's marker; consumers gate that
    # wrapping on it without naming a concrete hash.
    has_dedicated_fusion: bool

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
          explicit operands so a `lax.composite` can't lift them and break the ABI.
        - ``permute_from_operands(state, *constants)`` = the const-free permute the
          `fused_region` runs.
        - ``attrs`` = identifying `composite.attributes` (a ``permutation``
          discriminator + shape).

        Meaningful only on the dedicated path; a non-fused permutation returns an
        inert spec.
        """
        ...

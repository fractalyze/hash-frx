# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Rescue — the prime-field Marvellous permutation, at the RPO round structure.

The scheme is Rescue-Prime Optimized (Ashur-Kindi-Meier-Szepieniec-Threadbare,
https://eprint.iacr.org/2022/1577) — the production member of the Rescue
family (Miden VM's native hash), and the one the Task 0 anchoring gate on
fractalyze/hash-frx#190 could pin to published vectors. With 0-based round
index r in 0..N-1:

    round r:
        S <- mds @ S;  S <- S + C_{2r};    S[i] <- S[i] ** alpha
        S <- mds @ S;  S <- S + C_{2r+1};  S[i] <- S[i] ** inv_alpha

Note the half-round ORDER: linear layer, then constants, then S-box — the RPO
paper's Section 2.4 reordering. Vanilla Rescue-XLIX (the SoK, eprint
2020/1143, Algorithm 3) runs S-box, then linear, then constants; RPO is a
tweaked permutation, not a reparameterization, and this module implements
RPO's order. The parameter surface (`RescueParams`) carries both families
unchanged — only this body would move.

The permutation is one function (all rounds unrolled) wrapped in a
`frx.lax.composite`. No Fractalyze XLA plugin ships a Rescue emitter yet, so
`_DEDICATED_EMITTER_AVAILABLE` is False and `_EMITTER_BACKENDS` is empty:
every instance today marks the generic `zorch.fused_region` and reports
`fusion_path = GENERIC` — the same fallback Keccak takes on a backend without
its arm (`keccak.permutation` carries the family-wide rationale). The
dedicated marker `hash_frx.perm.rescue`, its operand ABI (`_abi_operands`),
and its attrs are already defined and registry-listed (`hash_frx.markers`), so
an emitter landing flips the two module flags and nothing else here moves.

The body is kept straight-line for the generic rewriter: rounds unroll (fixed,
small count), the MDS uses the normal-form `apply_matrix`, and both S-boxes
are `fnp.power` with static integer exponents — the Poseidon-family spelling:
a static exponent lowers to an explicit square-and-multiply chain of
element-wise multiplies (~2*log2(p) per layer, ~100 for the 64-bit RPO inverse
exponent), never a call, and the whole-body fusion-readiness and read-limit
cases in `testing/rescue_test.py` hold that lowering to the whitelist rather
than assuming it. Nothing here reduces, gathers, or calls.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
from frx import Array

from hash_frx.fusion import (
    FUSED_REGION_MARKER,
    FusionPath,
    fused_region,
    inert_region_spec,
)
from hash_frx.linear import apply_matrix
from hash_frx.rescue.params import RescueParams

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation

RESCUE_MARKER = "hash_frx.perm.rescue"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# below. It exists so a contract change can stage without a rename
# (`hash_frx.fusion`) once a recognizer ships.
RESCUE_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships a dedicated Rescue emitter.
# None exists yet — this is the pre-emitter half of the keccak arrangement
# (`keccak.permutation._DEDICATED_EMITTER_AVAILABLE` carries the family-wide
# rationale): flipped together with the `frx>=` floor in `pyproject.toml` when
# an emitter lands. While `_EMITTER_BACKENDS` is empty the pin is inert (the
# routing conjunction below is False either way) — the two flags are held to
# agree by `fusion_path_test`'s matrix law. Once a backend joins the tuple the
# pin cannot be left optimistic, because a DEDICATED fusion path also routes a
# `Sponge` over this permutation to `hash_frx.digest.field_sponge` carrying
# `permutation="rescue"`, which a plugin without the arm rejects outright — a
# failed compile, not a lost kernel.
_DEDICATED_EMITTER_AVAILABLE = False

# Which backends have that emitter — a different question from the pin, asked
# alongside it. Empty until one is written; a backend gaining an arm joins
# this tuple and nothing else here moves (the keccak/poseidon2 pattern).
_EMITTER_BACKENDS: tuple[str, ...] = ()


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry a Rescue emitter. Read per
    construction so importing does not initialize a backend; the lookup behind
    `frx.default_backend()` is memoized."""
    return _DEDICATED_EMITTER_AVAILABLE and frx.default_backend() in _EMITTER_BACKENDS


class Rescue:
    """A Rescue permutation built from a RescueParams; implements Permutation.

    permute = N rounds of (MDS -> +C -> x^alpha -> MDS -> +C -> x^inv_alpha),
    as ONE fused region.
    """

    def __init__(self, params: RescueParams) -> None:
        self._p = params
        self.width = params.width
        self.dtype = params.dtype
        name = RESCUE_MARKER if _routes_to_dedicated_emitter() else FUSED_REGION_MARKER
        # A generic region carries no version: the recognizer reads only the
        # name there, so a version would claim a contract the marker does not
        # have.
        self.fused_region_marker = (
            name,
            RESCUE_MARKER_VERSION if name != FUSED_REGION_MARKER else 0,
        )
        self.fusion_path = FusionPath.from_marker(name)

    def __eq__(self, other: object) -> bool:
        # Value identity IS the params surface plus the marker — the marker
        # tracks the emitter flags rather than the params, so without it a
        # dedicated and a generic perm on the same params collide in the
        # `_permute_body` static-arg cache (the arrangement `Poseidon2`
        # states).
        if not isinstance(other, Rescue):
            return NotImplemented
        return (
            self._p == other._p
            and self.fused_region_marker == other.fused_region_marker
        )

    def __hash__(self) -> int:
        return hash((self._p, self.fused_region_marker))

    def permute(self, state: Array) -> Array:
        """Apply the permutation: `(width,)` over `dtype` -> `(width,)`.

        Batch with `frx.vmap(permute)`; the body is element-wise over the
        state, so a batched call lowers to the same straight-line graph.
        """
        if state.ndim != 1 or state.shape[0] != self.width:
            raise ValueError(
                f"state must be a 1-D array of shape ({self.width},), got "
                f"{state.shape}"
            )
        if state.dtype != self.dtype:
            raise TypeError(
                f"state dtype {state.dtype} must match the permutation field "
                f"{self.dtype}"
            )
        return _permute_body(self, state)

    # Fused-region ABI (see `Permutation.fused_region_spec`).
    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """The Rescue ABI: operands `(leading, mds, round_constants)`, the
        operand-fed permute, and the identifying attrs. Dedicated path only;
        otherwise the shared inert stub."""
        if not self.fusion_path.is_one_kernel:
            return inert_region_spec(self, leading)
        return (
            _abi_operands(self, leading),
            partial(_permutation_body, self),
            _marker_attrs(self),
        )


def _permutation_body(
    perm: Rescue,
    s: Array,
    mds: Array,
    round_constants: Array,
) -> Array:
    """The straight-line permute on a single `(width,)` state, taking the ABI
    operands explicitly — the decomposition every marked Rescue region runs.
    Every caller invokes it positionally (marker attrs are absorbed by the
    `decomposition` wrapper in `_permute_body`). A batch is `vmap(permute)`,
    which lowers to the same body over a batched leading operand.
    """
    p = perm._p
    for r in range(p.rounds):
        # The RPO half-round order (its Section 2.4): linear layer, constants,
        # then the S-box — NOT the SoK's S-box-first order (module docstring).
        s = apply_matrix(mds, s)
        s = s + round_constants[2 * r]
        s = fnp.power(s, p.alpha)
        s = apply_matrix(mds, s)
        s = s + round_constants[2 * r + 1]
        s = fnp.power(s, p.inv_alpha)
    return s


def _abi_operands(perm: Rescue, state: Array) -> tuple[Array, ...]:
    """The marked region's operands, in the order a Rescue emitter's ABI would
    name them:

    ``[0] state``            `(..., width)` — single or batched
    ``[1] mds``              `(width, width)`
    ``[2] round_constants``  `(2*rounds, width)` — row 2r before round r's
                             alpha S-box, row 2r+1 before its inverse one

    Threading the tables as operands (rather than closing over them) is the
    operand-ABI rule in `docs/reference/conventions.md`: a host-materialised
    array would be lifted into an unnamed operand ahead of these, one per call
    site, leaving no layout to write down. The S-box exponents are NOT
    operands: they are static ints the body's power chains unroll over, so
    they identify the region (attrs) rather than parameterize a tensor.
    """
    p = perm._p
    return state, p.mds, p.round_constants


def _marker_attrs(perm: Rescue) -> dict[str, object]:
    """The dedicated marker's `composite.attributes` — identifying rather than
    parameterising (the tables ride as operands): the `permutation`
    discriminator a whole-region recognizer keys on, plus the shape ints and
    `alpha`, which fixes both power maps. `inv_alpha` deliberately does not
    ride: an emitter derives it as alpha^-1 mod (p-1) from its field (only the
    frx-side core treats `dtype` as opaque), and it cannot ride anyway — a
    composite int attribute lowers to a signed i64, and RPO-128's inv_alpha
    (10540996611094048183 > 2^63 - 1) does not fit one. Self-gating, as
    `Poseidon2` lays out: the generic marker stays attrs-free, and putting the
    gate here means no emission site can forget it."""
    if not perm.fusion_path.is_one_kernel:
        return {}
    return {
        "permutation": "rescue",
        "width": perm.width,
        "rounds": perm._p.rounds,
        "alpha": perm._p.alpha,
    }


# Module-level jit zone so the permutation body traces once per (params, state
# aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and a sponge absorb emits a permute per block. The permutation is
# the static key, compared by value; `inline=True` splices the cached jaxpr
# into the enclosing trace, so the emitted module is unchanged.
@partial(frx.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: Rescue, state: Array) -> Array:
    def decomposition(
        s: Array,
        mds: Array,
        round_constants: Array,
        **_attrs: object,
    ) -> Array:
        # `_attrs` is marker metadata passed through — the decomposition does
        # not read it. Inlined here so the region stays one straight-line body
        # (the generic marker's single-kernel requirement allows no call).
        return _permutation_body(perm, s, mds, round_constants)

    name, version = perm.fused_region_marker
    return fused_region(
        decomposition,
        *_abi_operands(perm, state),
        name=name,
        version=version,
        **_marker_attrs(perm),
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[Permutation] = Rescue

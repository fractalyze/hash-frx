# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Vision — the binary-field Marvellous permutation (Rescue's GF(2^n) sibling).

The scheme is the ToSC 2020(3) Marvellous paper's Vision
(https://eprint.iacr.org/2019/426, Section 5, Algorithm 1) in its unkeyed
sponge mode: the block cipher run under the all-zero master key, whose key
schedule degenerates to the fixed step constants `VisionParams.round_keys`.
One round is two steps; with 0-based step index r:

    S <- S + K_0                                  (before round 1)
    step r in 0..2N-1:
        S[i] <- S[i] ** -1        inverse S-box, 0 |-> 0 (the paper's
                                  x^(2^n - 2); the binary dtypes' native
                                  convention, so no masking)
        S    <- B_inv(S)  if r is even, else  B(S)
                                  the F2-linearized affine polynomial; the
                                  paper's pi_0 = B^{-1}(x^{-1}) opens the
                                  round, pi_1 = B(x^{-1}) closes it
                                  (its Figure 1)
        S    <- mds @ S + K_{r+1}

The permutation is one function (all steps unrolled) wrapped in a
`frx.lax.composite`. No Fractalyze XLA plugin ships a Vision emitter yet, so
`_DEDICATED_EMITTER_AVAILABLE` is False and `_EMITTER_BACKENDS` is empty:
every instance today marks the generic `zorch.fused_region` and reports
`fusion_path = GENERIC` — the same fallback Keccak takes on a backend without
its arm (`keccak.permutation` carries the family-wide rationale). The
dedicated marker `hash_frx.perm.vision`, its operand ABI (`_abi_operands`),
and its attrs are already defined and registry-listed (`hash_frx.markers`), so
an emitter landing flips the two module flags and nothing else here moves.

The body is kept straight-line for the generic rewriter: steps unroll (fixed,
small count), the linear layers use the normal-form helpers
(`apply_linearized_affine`, `apply_matrix`), and the inverse S-box lowers to
an element-wise `divide` — nothing reduces, gathers, or calls.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
from frx import Array

from hash_frx.fusion import (
    FUSED_REGION_MARKER,
    FusionPath,
    fused_region,
    inert_region_spec,
)
from hash_frx.vision.linear import apply_linearized_affine, apply_matrix
from hash_frx.vision.params import VisionParams

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation

VISION_MARKER = "hash_frx.perm.vision"
# Marker revision riding as `composite.version`; version 1 is the operand ABI
# below. It exists so a contract change can stage without a rename
# (`hash_frx.fusion`) once a recognizer ships.
VISION_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships a dedicated Vision emitter.
# None exists yet — this is the pre-emitter half of the keccak arrangement
# (`keccak.permutation._DEDICATED_EMITTER_AVAILABLE` carries the family-wide
# rationale): flipped together with the `frx>=` floor in `pyproject.toml` when
# an emitter lands. It cannot be left optimistic, because a DEDICATED fusion
# path also routes a `Sponge` over this permutation to
# `hash_frx.digest.field_sponge` carrying `permutation="vision"`, which a
# plugin without the arm rejects outright — a failed compile, not a lost
# kernel.
_DEDICATED_EMITTER_AVAILABLE = False

# Which backends have that emitter — a different question from the pin, asked
# alongside it. Empty until one is written; a backend gaining an arm joins
# this tuple and nothing else here moves (the keccak/poseidon2 pattern).
_EMITTER_BACKENDS: tuple[str, ...] = ()


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry a Vision emitter. Read per
    construction so importing does not initialize a backend; the lookup behind
    `frx.default_backend()` is memoized."""
    return _DEDICATED_EMITTER_AVAILABLE and frx.default_backend() in _EMITTER_BACKENDS


class Vision:
    """A Vision permutation built from a VisionParams; implements Permutation.

    permute = +K_0 -> 2N steps of (inverse S-box -> B_inv/B -> MDS -> +K),
    as ONE fused region.
    """

    def __init__(self, params: VisionParams) -> None:
        self._p = params
        self.width = params.width
        self.dtype = params.dtype
        name = VISION_MARKER if _routes_to_dedicated_emitter() else FUSED_REGION_MARKER
        # A generic region carries no version: the recognizer reads only the
        # name there, so a version would claim a contract the marker does not
        # have.
        self.fused_region_marker = (
            name,
            VISION_MARKER_VERSION if name != FUSED_REGION_MARKER else 0,
        )
        self.fusion_path = FusionPath.from_marker(name)

    def __eq__(self, other: object) -> bool:
        # Value identity IS the params surface plus the marker — the marker
        # tracks the emitter flags rather than the params, so without it a
        # dedicated and a generic perm on the same params collide in the
        # `_permute_body` static-arg cache (the arrangement `Poseidon2`
        # states).
        if not isinstance(other, Vision):
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
        """The Vision ABI: operands `(leading, b, b_inv, mds, round_keys)`,
        the operand-fed permute, and the identifying attrs. Dedicated path
        only; otherwise the shared inert stub."""
        if not self.fusion_path.is_one_kernel:
            return inert_region_spec(self, leading)
        return (
            _abi_operands(self, leading),
            partial(_permutation_body, self),
            _marker_attrs(self),
        )


def _permutation_body(
    perm: Vision,
    s: Array,
    b: Array,
    b_inv: Array,
    mds: Array,
    round_keys: Array,
    **_attrs: object,
) -> Array:
    """The straight-line permute on a single `(width,)` state, taking the ABI
    operands explicitly — the decomposition every marked Vision region runs.

    `_attrs` is marker metadata passed through, which the body does not read.
    A batch is `vmap(permute)`, which lowers to the same body over a batched
    leading operand.
    """
    s = s + round_keys[0]
    for r in range(2 * perm._p.rounds):
        # `** -1` IS the paper's S-box: the binary dtypes define `0 ** -1 == 0`
        # (x^(2^n - 2)), so no zero-mask branch reaches the body.
        s = s**-1
        s = apply_linearized_affine(b_inv if r % 2 == 0 else b, s)
        s = apply_matrix(mds, s)
        s = s + round_keys[r + 1]
    return s


def _abi_operands(perm: Vision, state: Array) -> tuple[Array, ...]:
    """The marked region's operands, in the order a Vision emitter's ABI would
    name them:

    ``[0] state``       `(..., width)` — single or batched
    ``[1] b``           `(k + 1,)`     — B, constant-first
    ``[2] b_inv``       `(n + 1,)`     — B^{-1}, constant-first
    ``[3] mds``         `(width, width)`
    ``[4] round_keys``  `(2*rounds + 1, width)` — step keys K_0..K_{2N}

    Threading the tables as operands (rather than closing over them) is the
    operand-ABI rule in `docs/reference/conventions.md`: a host-materialised
    array would be lifted into an unnamed operand ahead of these, one per call
    site, leaving no layout to write down.
    """
    p = perm._p
    return state, p.b, p.b_inv, p.mds, p.round_keys


def _marker_attrs(perm: Vision) -> dict[str, object]:
    """The dedicated marker's `composite.attributes` — identifying rather than
    parameterising (the tables ride as operands): the `permutation`
    discriminator a whole-region recognizer keys on, plus the shape ints.
    Caller gates on `fusion_path.is_one_kernel`; the generic marker stays
    attrs-free."""
    return {
        "permutation": "vision",
        "width": perm.width,
        "rounds": perm._p.rounds,
    }


# Module-level jit zone so the permutation body traces once per (params, state
# aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and a sponge absorb emits a permute per block. The permutation is
# the static key, compared by value; `inline=True` splices the cached jaxpr
# into the enclosing trace, so the emitted module is unchanged.
@partial(frx.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: Vision, state: Array) -> Array:
    def decomposition(
        s: Array,
        b: Array,
        b_inv: Array,
        mds: Array,
        round_keys: Array,
        **_attrs: object,
    ) -> Array:
        # Inlined here so the region stays one straight-line body (the generic
        # marker's single-kernel requirement allows no call).
        return _permutation_body(perm, s, b, b_inv, mds, round_keys)

    name, version = perm.fused_region_marker
    return fused_region(
        decomposition,
        *_abi_operands(perm, state),
        name=name,
        version=version,
        **(_marker_attrs(perm) if perm.fusion_path.is_one_kernel else {}),
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[Permutation] = Vision

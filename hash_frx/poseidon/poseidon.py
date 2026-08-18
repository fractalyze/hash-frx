"""Classic Poseidon permutation — scheme-agnostic, single-kernel by construction.

The permutation is one function (all rounds) wrapped in a `frx.lax.composite`
(`fused_region`): XLA's `ZorchFusedRegionRewriter` turns that marker into a
single custom-fusion kernel — one kernel by construction, not via a per-hash
compiler pattern match. The region is named `hash_frx.poseidon` (distinct from
`hash_frx.poseidon2`), the permutation shape riding as `composite.attributes`
(`width`/`full_rounds`/`partial_rounds`/`alpha`/`mds`), and routes to XLA's
dedicated, params-driven Poseidon emitter where the pin and the backend carry
it — elsewhere the generic `zorch.fused_region` marker stands in, correct but
un-routed. The body is kept straight-line:
rounds are unrolled (fixed, small counts) and the dense MDS uses the normal-form
helper (`apply_matrix`) so nothing lowers to a reduce/dot/gather that would
split the kernel.

Classic Poseidon (ark-sponge style): each round is `ARC -> S-box -> dense MDS`.
The rounds split full/partial/full — `full_rounds/2` full rounds (S-box `x^alpha`
on all lanes), then `partial_rounds` partial rounds (S-box on the last lane
only), then `full_rounds/2` full rounds — and the dense MDS runs every round.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array

from hash_frx.fusion import FUSED_REGION_MARKER, FusionPath, fused_region
from hash_frx.linear import apply_matrix
from hash_frx.poseidon.params import PoseidonParams

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation

POSEIDON_MARKER = "hash_frx.poseidon"
# Marker revision riding as `composite.version`. XLA recognizes the marker by
# name + attributes and deliberately does not gate on the version; it exists so
# a future contract change can be staged without renaming the marker.
POSEIDON_MARKER_VERSION = 1

# Whether the pinned Fractalyze XLA plugin ships the dedicated Poseidon emitter.
# Flipped together with the `frx>=` floor in `pyproject.toml`, like
# `keccak.permutation._DEDICATED_EMITTER_AVAILABLE` (its comment carries the
# family-wide rationale).
_DEDICATED_EMITTER_AVAILABLE = True

# Which backends have it — a different question from the pin, asked alongside
# it. The `ZorchFusedRegionRewriter` routes the classic-Poseidon arm in both
# the CPU and the GPU compiler, so both legs are here; a backend absent — Metal
# today — is on the generic path until an emitter is written for it. Contrast
# `poseidon.sparse._EMITTER_BACKENDS`, which is GPU-only.
_EMITTER_BACKENDS = ("cpu", "gpu")


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry the Poseidon emitter. Read
    per construction so importing does not initialize a backend; the lookup
    behind `frx.default_backend()` is memoized."""
    return _DEDICATED_EMITTER_AVAILABLE and frx.default_backend() in _EMITTER_BACKENDS


class Poseidon:
    """A classic Poseidon permutation built from a PoseidonParams; implements
    Permutation.

    permute = full_rounds/2 full rounds (initial RC) -> partial_rounds partial
              rounds -> full_rounds/2 full rounds (terminal RC), each round
              `ARC -> S-box -> dense MDS`, as ONE fused region.
    """

    def __init__(self, params: PoseidonParams) -> None:
        self._p = params
        self.width = params.width
        self.dtype = params.dtype
        # Extracted once here (eager): the MDS-to-canonical-int conversion would
        # stage into the jaxpr if done inside the traced `permute` body. The
        # marker carries these ints (flattened row-major) as the `mds` attribute.
        self._mds_rows = params.mds_rows
        # Classic Poseidon has no free-form fallback of its own — the only
        # routing question is whether the pin and the backend carry the
        # dedicated `hash_frx.poseidon` emitter.
        name = (
            POSEIDON_MARKER if _routes_to_dedicated_emitter() else FUSED_REGION_MARKER
        )
        # A generic region carries no version: the recognizer reads only the name
        # there, so a version would claim a contract the marker does not have.
        self.fused_region_marker = (
            name,
            POSEIDON_MARKER_VERSION if name != FUSED_REGION_MARKER else 0,
        )
        # Derived from the marker rather than asserted alongside it, so the
        # three cannot drift (mirrors Poseidon2 and SparsePoseidon); never HOST
        # — a permutation is a device function by construction.
        self.fusion_path = (
            FusionPath.DEDICATED if name != FUSED_REGION_MARKER else FusionPath.GENERIC
        )
        self.has_dedicated_fusion = self.fusion_path.is_one_kernel

    def __eq__(self, other: object) -> bool:
        # Value identity IS the params surface — required for the pytree-aux
        # seat in `DuplexTranscript` (docs/reference/conventions.md
        # "Pytree registration"). The marker joins the key because it is not a
        # function of the params alone (it tracks the emitter flags): without it
        # a dedicated and a generic perm on the same params collide in the
        # `_permute_body` static-arg cache (the arrangement `SparsePoseidon`
        # states).
        if not isinstance(other, Poseidon):
            return NotImplemented
        return (
            self._p == other._p
            and self.fused_region_marker == other.fused_region_marker
        )

    def __hash__(self) -> int:
        return hash((self._p, self.fused_region_marker))

    def permute(self, state: Array) -> Array:
        if state.ndim != 1 or state.shape[0] != self.width:
            raise ValueError(
                f"state must be a 1-D array of shape ({self.width},), got {state.shape}"
            )
        return _permute_body(self, state)

    # Fused-region ABI (see `Permutation.fused_region_spec`).
    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """The classic-Poseidon ABI: operands `(leading, round_constants)`, the
        full/partial/full dense-MDS permute, and attrs whose `mds` names the linear
        layer. Dedicated path only; otherwise an inert stub (non-dedicated, so
        consumers never route a whole-region composite through it)."""
        if not self.fusion_path.is_one_kernel:
            return (leading,), (lambda state, *_ops: self.permute(state)), {}
        return (
            (leading, self._p.round_constants.reshape(-1)),
            partial(_permute_from_rc, self),
            {"permutation": "poseidon", **_poseidon_marker_attrs(self)},
        )


# The classic Poseidon permute on `s` given round constants flattened row-major
# (the marked region's ABI operand). Shared by the permute marker and the
# sponge-hash marker so the two carry one round schedule. full/partial/full
# rounds, each ARC -> S-box -> dense MDS; the MDS is a closed-over field array
# (frx keeps composite consts inline, so it never surfaces as an operand).
def _permute_from_rc(perm: "Poseidon", s: Array, rc_flat: Array) -> Array:
    p = perm._p
    alpha = p.alpha
    w = perm.width
    half_full = p.full_rounds // 2
    partial = p.partial_rounds
    mds = p.mds

    def full_round(st: Array, rc: Array) -> Array:
        return apply_matrix(mds, fnp.power(st + rc, alpha))

    def partial_round(st: Array, rc: Array) -> Array:
        st = st + rc
        last = fnp.power(st[w - 1], alpha)
        # concatenate, not a static-index set: the latter lowers to scatter,
        # which would split the fused kernel.
        st = fnp.concatenate([st[: w - 1], last[None]])
        return apply_matrix(mds, st)

    rc = rc_flat.reshape(2 * half_full + partial, w)
    r = 0
    for _ in range(half_full):
        s = full_round(s, rc[r])
        r += 1
    for _ in range(partial):
        s = partial_round(s, rc[r])
        r += 1
    for _ in range(half_full):
        s = full_round(s, rc[r])
        r += 1
    return s


# Module-level jit zone so the permutation body traces once per (params, state
# aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits hundreds of identical-aval permutes (every
# Merkle level, leaf hash, and transcript observe/sample) — the uncached
# re-trace of this body dominates the first-trace-per-config floor.
# The permutation is the static key, compared by value; `inline=True`
# splices the cached jaxpr into the enclosing trace, so the emitted module
# (one composite marker per permute) is unchanged.
@partial(frx.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: Poseidon, state: Array) -> Array:
    p = perm._p
    w = perm.width

    def permutation(s: Array, rc_flat: Array, **_attrs: object) -> Array:
        # `_attrs` is marker metadata (name + attrs); the body ignores it.
        return _permute_from_rc(perm, s, rc_flat)

    # ABI operands [state, round_constants flattened row-major].
    operands = (state, p.round_constants.reshape(-1))
    # `mds` is a numpy int64 value (not a Python list) so it lowers to a
    # `dense<[..]>:tensor<N*Nxi64>` the recognizer parses with
    # GetCompositeAttrIntArray; a plain list lowers to an unparsed ArrayAttr.
    # The generic marker stays attrs-free — the recognizer reads only its name.
    marker_attrs: dict[str, object] = (
        _poseidon_marker_attrs(perm) if perm.fusion_path.is_one_kernel else {}
    )
    name, version = perm.fused_region_marker
    return fused_region(
        permutation,
        *operands,
        name=name,
        version=version,
        **marker_attrs,
    )


# The permute shape as `composite.attributes` (the recognizer's contract: shape
# ints — alpha is its s-box degree — plus `mds`, the width*width MDS flattened
# row-major, applied as the dense linear layer). Shared with the sponge marker.
def _poseidon_marker_attrs(perm: "Poseidon") -> dict[str, object]:
    p = perm._p
    w = perm.width
    return {
        "width": w,
        "full_rounds": p.full_rounds,
        "partial_rounds": p.partial_rounds,
        "alpha": p.alpha,
        "mds": np.array(perm._mds_rows, dtype=np.int64).flatten(),
    }


if TYPE_CHECKING:
    _: type[Permutation] = Poseidon

"""Classic Poseidon permutation — scheme-agnostic, single-kernel by construction.

The permutation is one function (all rounds) wrapped in a `frx.lax.composite`
(`fused_region`): XLA's `ZorchFusedRegionRewriter` turns that marker into a
single custom-fusion kernel — one kernel by construction, not via a per-hash
compiler pattern match. The region is named `hash_frx.perm.poseidon` (distinct from
`hash_frx.perm.poseidon2`), the permutation shape riding as `composite.attributes`
(`width`/`full_rounds`/`partial_rounds`/`alpha`/`mds`), and routes to XLA's
dedicated, params-driven Poseidon emitter where the pin and the backend carry
it — elsewhere the generic `zorch.fused_region` marker stands in, correct but
un-routed. The body is kept straight-line:
rounds are unrolled (fixed, small counts) and the dense MDS uses the normal-form
helper (`apply_matrix`) so nothing lowers to a reduce/dot/gather that would
split the kernel.

The dedicated emitter does not serve every parameterization. It applies the MDS
as a small-integer add-chain, so a matrix outside that range — which is every
matrix over a real field — takes the generic marker instead, one kernel either
way. `_select_fused_region_name` is where that is decided.

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

from hash_frx.fusion import (
    FUSED_REGION_MARKER,
    FusionPath,
    fused_region,
    inert_region_spec,
    permute_marker,
    routing,
)
from hash_frx.linear import apply_matrix
from hash_frx.poseidon.params import PoseidonParams

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation

POSEIDON_MARKER = "hash_frx.perm.poseidon"
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


# The dedicated emitter applies the MDS with a small-integer **add-chain** —
# `c * x` is `x` added `c` times — so it takes entries in `[0, 64)` and no
# all-zero row, and its recognizer rejects the config outright otherwise
# (`ParsePoseidonConfig`, Fractalyze XLA `xla/codegen/emitters/poseidon.cc`,
# whose own comment calls the add-chain a stopgap for the toy config). Flipped
# together with the `frx>=` floor in `pyproject.toml`, like
# `_DEDICATED_EMITTER_AVAILABLE` above: it is that emitter's envelope, so it
# moves when the pin does — a copy that drifts either brings the failed compile
# back or loses fusion silently.
#
# It also stands in for the int64 representability the marker attribute needs
# (#117) — a bound this far below `2**63 - 1` rejects a too-wide entry before the
# cast can see it, which `_goldilocks_params` in the test pins. That falls out of
# the bound rather than being a second gate, so it stops holding if
# fractalyze/xla#604 lifts the add-chain restriction, and #117 then needs the
# explicit `_fits_i64`-shaped check `sparse.py` already carries.
_DEDICATED_EMITTER_MDS_BOUND = 64


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend both carry this emitter
    (`fusion.routing`, which carries the rationale)."""
    return routing(_DEDICATED_EMITTER_AVAILABLE, _EMITTER_BACKENDS)


def _fits_add_chain(mds_rows: tuple[tuple[int, ...], ...]) -> bool:
    """Whether the dedicated emitter's add-chain can apply this MDS."""
    in_range = all(
        0 <= entry < _DEDICATED_EMITTER_MDS_BOUND for row in mds_rows for entry in row
    )
    return in_range and all(any(row) for row in mds_rows)


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
        name = self._select_fused_region_name(self._mds_rows)
        self.fused_region_marker = permute_marker(name, POSEIDON_MARKER_VERSION)
        self.fusion_path = FusionPath.from_marker(name)

    def _select_fused_region_name(self, mds_rows: tuple[tuple[int, ...], ...]) -> str:
        """Route to the dedicated `PoseidonFusion` when the pin *and* the
        backend ship it AND this MDS is one its add-chain can apply, else the
        generic marker — which the `ZorchFusedRegionRewriter` still fuses to one
        kernel, so an unroutable set gets the right bytes off the dedicated path
        rather than a compile that fails on the recognizer's
        "unparsable composite.attributes".

        The MDS is what decides it, and a real one rarely fits: every entry has
        to sit in `[0, 64)` (see `_DEDICATED_EMITTER_MDS_BOUND`), which no matrix
        over a 31-bit field does.
        """
        if not _routes_to_dedicated_emitter():
            return FUSED_REGION_MARKER
        return POSEIDON_MARKER if _fits_add_chain(mds_rows) else FUSED_REGION_MARKER

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
        layer. Dedicated path only; otherwise the shared inert stub."""
        if not self.fusion_path.is_one_kernel:
            return inert_region_spec(self, leading)
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

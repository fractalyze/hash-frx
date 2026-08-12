"""Optimized-sparse Poseidon permutation — a naive Hades Poseidon, faster.

`SparsePoseidon` re-factors the partial rounds of a naive Hades Poseidon: a
transition matrix `P` after the last pre-partial round, then a rank-structured
sparse update each partial round, instead of the dense MDS. It produces the same
permutation that *naive* schedule would with the equivalent (unfolded) constants
— dense MDS every round, full-width constants — but with fewer multiplications,
so it consumes a reference's already-optimized constants directly rather than
un-folding them.

The naive schedule it matches is *not* this package's `Poseidon` class, which
follows two different conventions: `Poseidon` S-boxes the last lane in a partial
round and orders each round `ARC -> S-box -> MDS`, whereas `SparsePoseidon`
S-boxes lane 0 and orders `S-box -> ARC(next) -> matrix` (the S-box precedes the
constant; the initial ARC seeds the first). The equivalence is to a naive Hades
with *those* conventions; `testing/sparse_test.py` pins it by deriving the sparse
factorization from a random naive instance and byte-matching.

Like `Poseidon`, the whole permute is one straight-line function (rounds
unrolled, linear layers via the normal-form helpers, no reduce/dot/gather) so it
lowers to a single kernel under the generic `zorch.fused_region` marker.

There is also a dedicated `hash_frx.sparse_poseidon` name-routed marker — mirroring
`hash_frx.poseidon2` — whose emitter (`SparsePoseidonFusion`, in the Fractalyze XLA
plugin) exploits the sparse structure: the schedule shape and the four matrices
(`mds`, `transition_matrix`, and the per-round `partial_dot` / `partial_col`
pairs) ride as int64 marker attributes, the additive round constants as operands.
That path is gated behind `_DEDICATED_EMITTER_AVAILABLE`: when the pinned plugin
ships the emitter the permutation emits the dedicated marker, otherwise it falls
back to the generic marker (which fuses this body to one kernel just the same),
because emitting a marker the plugin cannot recognize fails every compile with
`custom op 'stablehlo.composite' is unknown`. Canonical values in
`[2**63, 2**64)` — Goldilocks, `p = 2^64 - 2^32 + 1` — ride the attributes as a
u64 bit-cast (negative i64 bit patterns the emitter reinterprets), gated on
`_WIDE_ATTR_EMITTER_AVAILABLE`; a field wider than a u64 has nothing the
attributes can carry and always takes the generic marker. Either way the
dedicated path's attributes and reference body are exercised directly by
`testing/sparse_test.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array

from hash_frx.fusion import FUSED_REGION_MARKER, fused_region
from hash_frx.linear import apply_matrix
from hash_frx.poseidon.linear import apply_sparse_partial
from hash_frx.poseidon.params import SparsePoseidonParams

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation

POSEIDON_SPARSE_MARKER = "hash_frx.sparse_poseidon"
# Marker revision riding as `composite.version`. XLA recognizes the marker by
# name + attributes and deliberately does not gate on the version; it exists so
# a future contract change can be staged without renaming the marker. Version 2
# is the revision at which the wide (u64 bit-cast) matrix attributes entered
# the encoding contract; it rides on every dedicated instance, narrow fields
# included.
POSEIDON_SPARSE_MARKER_VERSION = 2

_I64_MAX = 2**63 - 1
_U64_MAX = 2**64 - 1

# Whether the pinned Fractalyze XLA plugin ships the dedicated
# `SparsePoseidonFusion` emitter. When True, `permute` emits the dedicated
# `hash_frx.sparse_poseidon` marker; when False it falls back to the generic
# `zorch.fused_region` marker (which fuses the same body to one kernel), because
# emitting a marker the plugin cannot recognize fails every compile with
# `custom op 'stablehlo.composite' is unknown`.
_DEDICATED_EMITTER_AVAILABLE = True

# Whether the pinned plugin accepts the wide (bit-cast) attribute range. A
# plugin without the recognizer (xla#440) hard-fails compiling the negative
# bit-cast i64s rather than degrading, hence a pin-tracking flag and not a data
# property; flipped together with the pin, like `_DEDICATED_EMITTER_AVAILABLE`.
_WIDE_ATTR_EMITTER_AVAILABLE = True


class SparsePoseidon:
    """An optimized-sparse Poseidon permutation built from a SparsePoseidonParams;
    implements Permutation.

    permute = initial ARC -> (half-1) full rounds -> transition round (P)
              -> n_partial sparse rounds -> (half-1) full rounds
              -> final full round (no trailing constant), as one fused region.
    """

    def __init__(self, params: SparsePoseidonParams) -> None:
        self._p = params
        self.width = params.width
        self.dtype = params.dtype
        # Extracted eagerly, BEFORE marker selection: attribute
        # representability is part of the routing decision.
        rows = (
            params.mds_rows,
            params.transition_matrix_rows,
            params.partial_dot_rows,
            params.partial_col_rows,
        )
        name = self._select_fused_region_name(rows)
        # A generic region carries no version: the recognizer reads only the name
        # there, so a version would claim a contract the marker does not have.
        self.fused_region_marker = (
            name,
            POSEIDON_SPARSE_MARKER_VERSION if name != FUSED_REGION_MARKER else 0,
        )
        # Derived from the marker choice itself so the two can't drift.
        self.has_dedicated_fusion = name != FUSED_REGION_MARKER
        if self.has_dedicated_fusion:
            (
                self._mds_rows,
                self._transition_rows,
                self._partial_dot_rows,
                self._partial_col_rows,
            ) = rows

    def _select_fused_region_name(
        self, rows: tuple[tuple[tuple[int, ...], ...], ...]
    ) -> str:
        """Route to the dedicated `SparsePoseidonFusion` when the pinned plugin
        ships it AND the four matrices are representable in its int64 attributes,
        else the generic marker so compiles don't fail on an unknown composite.

        Every linear layer here is a matrix of field elements (unlike Poseidon2,
        whose one matrix attribute is the small structural M4). Values below 2^63
        ride as literal ints; values in `[2^63, 2^64)` — Goldilocks,
        `p = 2^64 - 2^32 + 1` — ride as a u64 bit-cast, gated on
        `_WIDE_ATTR_EMITTER_AVAILABLE` (see its rationale). A wider field has
        nothing the attributes can carry and takes the generic marker."""
        if not _DEDICATED_EMITTER_AVAILABLE:
            return FUSED_REGION_MARKER
        if all(_fits_i64(m) for m in rows):
            return POSEIDON_SPARSE_MARKER
        if _WIDE_ATTR_EMITTER_AVAILABLE and all(_fits_u64(m) for m in rows):
            return POSEIDON_SPARSE_MARKER
        return FUSED_REGION_MARKER

    def __eq__(self, other: object) -> bool:
        # Value identity IS the params surface — required for the pytree-aux seat
        # in `DuplexTranscript` (docs/reference/conventions.md "Pytree
        # registration"). The marker joins the key because it is not a function
        # of the params alone (it tracks the emitter flags): without it a
        # dedicated and a generic perm on the same params collide in the
        # `_permute_body` static-arg cache.
        if not isinstance(other, SparsePoseidon):
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
        if state.dtype != self.dtype:
            raise TypeError(
                f"state dtype {state.dtype} must match the permutation field "
                f"{self.dtype}"
            )
        return _permute_body(self, state)

    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """The SparsePoseidonFusion ABI: operands `(leading, *additive round
        constants)`, the operand-fed permute, and attrs whose four matrices
        (`mds` / `transition_matrix` / `partial_dot` / `partial_col`) name the
        linear layers. Dedicated path only; otherwise an inert stub (non-dedicated,
        so consumers never route a whole-region composite through it)."""
        if not self.has_dedicated_fusion:
            return (leading,), (lambda state, *ops: self.permute(state)), {}
        attrs = _marker_attrs(self)
        return (
            _abi_operands(self, leading),
            partial(_permute_from_operands, self),
            {"permutation": "sparse_poseidon", **attrs},
        )


def _permute_from_params(perm: "SparsePoseidon", s: Array) -> Array:
    """The optimized-sparse permute on a single `(width,)` state. Round constants
    and matrices ride as closed-over field arrays, staying inline in the marked
    body; the matrices apply via the field-array normal-form helpers."""
    p = perm._p
    alpha = p.alpha
    half = p.half_full_rounds
    mds = p.mds

    # S-box THEN add the round constant (the constant seeds the NEXT round's
    # S-box), then the linear layer — the constant follows the power map.
    def full_round(state: Array, rc: Array, matrix: Array) -> Array:
        return apply_matrix(matrix, fnp.power(state, alpha) + rc)

    # Initial ARC seeds the first S-box (added before any power map).
    s = s + p.initial_arc
    # Pre-partial full rounds with the dense MDS.
    for r in range(half - 1):
        s = full_round(s, p.full_rc_pre[r], mds)
    # Transition full round: the linear layer is P, not M.
    s = full_round(s, p.transition_rc, p.transition_matrix)
    # Partial rounds: the S-box hits lane 0 only (that is what makes them cheap).
    for r in range(p.n_partial_rounds):
        a = fnp.power(s[0], alpha) + p.partial_rc[r]
        s = apply_sparse_partial(p.partial_dot[r], p.partial_col[r], a, s[1:])
    # Post-partial full rounds with the dense MDS.
    for r in range(half - 1):
        s = full_round(s, p.full_rc_post[r], mds)
    # Final full round: S-box + MDS, no trailing constant.
    s = apply_matrix(mds, fnp.power(s, alpha))
    return s


def _permute_from_operands(
    perm: "SparsePoseidon",
    s: Array,
    initial_arc: Array,
    full_rc_pre: Array,
    transition_rc: Array,
    partial_rc: Array,
    full_rc_post: Array,
) -> Array:
    """The SparsePoseidonFusion ABI reference body: the same schedule as
    `_permute_from_params`, but the additive round constants arrive as explicit
    operands. The four matrices are closed-over field arrays: frx keeps
    composite consts inline in the decomposition, so they never surface as
    operands, and the sparse structure the emitter reads rides as int64 marker
    attributes. `full_rc_pre` / `full_rc_post` arrive flattened row-major."""
    p = perm._p
    alpha = p.alpha
    w = perm.width
    half = p.half_full_rounds
    pre = full_rc_pre.reshape(half - 1, w)
    post = full_rc_post.reshape(half - 1, w)

    def full_round(state: Array, rc: Array, matrix: Array) -> Array:
        return apply_matrix(matrix, fnp.power(state, alpha) + rc)

    s = s + initial_arc
    for r in range(half - 1):
        s = full_round(s, pre[r], p.mds)
    # Transition full round: the linear layer is P, not M.
    s = full_round(s, transition_rc, p.transition_matrix)
    for r in range(p.n_partial_rounds):
        a = fnp.power(s[0], alpha) + partial_rc[r]
        s = apply_sparse_partial(p.partial_dot[r], p.partial_col[r], a, s[1:])
    for r in range(half - 1):
        s = full_round(s, post[r], p.mds)
    s = apply_matrix(p.mds, fnp.power(s, alpha))
    return s


def _abi_operands(perm: "SparsePoseidon", state: Array) -> tuple[Array, ...]:
    """The SparsePoseidonFusion ABI operands: `state` then the additive round
    constants (the transition round's `P` and the partial rounds' sparse pairs
    ride as attributes, not operands). `state` is `(width,)` single or
    `(n, width)` batched; the constants are identical either way. The full-round
    constants flatten row-major."""
    p = perm._p
    return (
        state,
        p.initial_arc,
        p.full_rc_pre.reshape(-1),
        p.transition_rc,
        p.partial_rc,
        p.full_rc_post.reshape(-1),
    )


def _fits_i64(rows: tuple[tuple[int, ...], ...]) -> bool:
    """Whether every canonical entry rides the int64 marker attributes as a
    literal (no bit-cast). Canonical values are non-negative."""
    return all(0 <= v <= _I64_MAX for row in rows for v in row)


def _fits_u64(rows: tuple[tuple[int, ...], ...]) -> bool:
    """Whether every canonical entry is representable at all — as a literal or
    as the u64 bit-cast `_rows_to_i64` encodes. Checked before it builds an
    attribute, which would otherwise raise OverflowError at construction."""
    return all(0 <= v <= _U64_MAX for row in rows for v in row)


def _rows_to_i64(rows: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Canonical-int rows flattened row-major as a numpy int64 array — a numpy
    value (not a Python list) so the marker attribute lowers to a
    `dense<[..]> : tensor<Nxi64>` the recognizer parses, not an unparsed ArrayAttr.
    Mirrors `Poseidon._poseidon_marker_attrs`' `mds`. Encoded through uint64 and
    bit-cast to int64: identity for values below 2^63, and a value in
    `[2^63, 2^64)` becomes the negative i64 with the same bits, which the
    emitter's uint64 reinterpretation restores. `_select_fused_region_name` has
    already established representability for a dedicated perm."""
    return np.array(rows, dtype=np.uint64).flatten().view(np.int64)


def _marker_attrs(perm: "SparsePoseidon") -> dict[str, object]:
    """The dedicated marker's `composite.attributes`: the schedule shape
    ints (the recognizer maps `alpha` to its s-box degree) plus the four matrices
    flattened row-major as int64 dense attrs. The marker NAME disambiguates the
    permutation, so no `permutation` discriminator here (that rides only in
    `fused_region_spec`, where a region could wrap any permutation). The body
    ignores these (metadata only); the generic marker stays attrs-free."""
    if not perm.has_dedicated_fusion:
        return {}
    p = perm._p
    return {
        "width": perm.width,
        "half_full_rounds": p.half_full_rounds,
        "n_partial_rounds": p.n_partial_rounds,
        "alpha": p.alpha,
        "mds": _rows_to_i64(perm._mds_rows),
        "transition_matrix": _rows_to_i64(perm._transition_rows),
        "partial_dot": _rows_to_i64(perm._partial_dot_rows),
        "partial_col": _rows_to_i64(perm._partial_col_rows),
    }


# Module-level jit zone so the permutation body traces once per (params, state
# aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits hundreds of identical-aval permutes. The
# permutation is the static key, compared by value; `inline=True` splices the
# cached jaxpr into the enclosing trace, so the emitted module is unchanged.
@partial(frx.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: SparsePoseidon, state: Array) -> Array:
    # `perm` is static, so this branch resolves at trace time (no HLO select).
    if not perm.has_dedicated_fusion:

        def decomposition(s: Array) -> Array:
            return _permute_from_params(perm, s)

        return fused_region(decomposition, state, name=FUSED_REGION_MARKER)

    def dedicated(
        s: Array,
        initial_arc: Array,
        full_rc_pre: Array,
        transition_rc: Array,
        partial_rc: Array,
        full_rc_post: Array,
        **_attrs: object,
    ) -> Array:
        return _permute_from_operands(
            perm, s, initial_arc, full_rc_pre, transition_rc, partial_rc, full_rc_post
        )

    name, version = perm.fused_region_marker
    return fused_region(
        dedicated,
        *_abi_operands(perm, state),
        name=name,
        version=version,
        **_marker_attrs(perm),
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[Permutation] = SparsePoseidon

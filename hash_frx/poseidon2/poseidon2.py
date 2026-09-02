"""Poseidon2 permutation — scheme-agnostic, single-kernel by construction.

The permutation is one function (all rounds) wrapped in a `frx.lax.composite`
(`fused_region`): XLA's `ZorchFusedRegionRewriter` turns that marker into a
single custom-fusion kernel — one kernel by construction, not via a per-hash
compiler pattern match. With the standard external matrix the region is named
`hash_frx.perm.poseidon2`, the permutation shape riding as `composite.attributes`
(`width`/`external_rounds`/`internal_rounds`/`alpha`), and routes to XLA's
dedicated, params-driven Poseidon2Fusion emitter; a non-standard external
matrix falls back to the generic
`zorch.fused_region` marker (whose generic LoopFusion would compile a full
permute into one register-spilling kernel). The body is kept straight-line:
rounds are unrolled (fixed, small counts) and the linear layers use the
normal-form helpers (`apply_matrix`, `apply_internal`) so nothing lowers to a
reduce/dot/gather that would split the kernel.
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
from hash_frx.poseidon2.linear import (
    apply_external_m4,
    apply_internal,
    apply_matrix,
)
from hash_frx.poseidon2.params import Poseidon2Params

if TYPE_CHECKING:
    from hash_frx.permutation import Permutation

POSEIDON2_MARKER = "hash_frx.perm.poseidon2"
# Marker revision riding as `composite.version`. XLA recognizes the marker by
# name + attributes and deliberately does not gate on the version; it exists so
# a future contract change can be staged without renaming the marker. v2: the J
# scale is the `internal_j_scale` attribute, not an operand.
POSEIDON2_MARKER_VERSION = 2

# Whether the pinned Fractalyze XLA plugin ships the dedicated Poseidon2Fusion
# emitters. Flipped together with the `frx>=` floor in `pyproject.toml`, like
# `keccak.permutation._DEDICATED_EMITTER_AVAILABLE` (its comment carries the
# family-wide rationale: below the floor the marker is emitted, ignored, and
# computes the right bytes with none of the kernels).
_DEDICATED_EMITTER_AVAILABLE = True

# Which backends can DRIVE this primitive — run its round schedule from the
# params a marker carries. Both legs do, and a whole-hash envelope over the
# permutation (a sponge, a Merkle compress) needs only this: those emitters
# build the permutation from the plugin's primitive registry, naming no
# permutation of their own. A backend absent here — Metal today — is on the
# generic path until an emitter is written for it.
_EMITTER_BACKENDS = ("cpu", "gpu")

# Which backends route the STANDALONE permute marker to a kernel of its own — a
# narrower question than driving the primitive, and a separate tuple for the
# same reason `sha256`'s raw-byte form carries one. The GPU is absent by
# measurement rather than omission: there the marker's own decomposition becomes
# a single kFor fusion, byte-identical to the dedicated kernel and within 1.02x
# of it from 2^14 rows up, so a dedicated arm would buy nothing. The CPU has no
# such conversion — the same body lowers to 64 kernels and 13-32x the runtime —
# and keeps its emitter.
_PERMUTE_EMITTER_BACKENDS = ("cpu",)


def _routes_to_dedicated_emitter() -> bool:
    """Whether the pin *and* the backend route the standalone permute marker
    (`fusion.routing`, which carries the rationale)."""
    return routing(_DEDICATED_EMITTER_AVAILABLE, _PERMUTE_EMITTER_BACKENDS)


def _drives_the_primitive() -> bool:
    """Whether the pin *and* the backend can run this permutation from its
    params — what an envelope over it needs, and what `fusion_path` reports."""
    return routing(_DEDICATED_EMITTER_AVAILABLE, _EMITTER_BACKENDS)


class Poseidon2:
    """A Poseidon2 permutation built from a Poseidon2Params; implements Permutation.

    permute = pre-MDS -> external_rounds (initial RC) -> internal_rounds
              -> external_rounds (terminal RC), as ONE fused region.
    """

    def __init__(self, params: Poseidon2Params) -> None:
        self._p = params
        self.width = params.width
        self.dtype = params.dtype
        # Decided once here (eager): the structure check + M4 extraction would
        # stage into the jaxpr if done inside the traced `permute` body. Gates both
        # the marker name and the const-free (literal-M4) external layer.
        self._is_m4_structured = params.is_m4_block_structured
        self._external_m4 = params.external_m4 if self._is_m4_structured else None
        name = self._select_fused_region_name()
        self.fused_region_marker = permute_marker(name, POSEIDON2_MARKER_VERSION)
        # Read off the primitive rather than off the marker (`from_marker`, which
        # the families whose two answers coincide still use): `fusion_path` gates
        # whether a whole-region composite over this permutation is expandable,
        # and that holds on a backend which drives the primitive but routes no
        # standalone permute marker. Deriving it from the marker would report
        # GENERIC there and cost the enclosing sponge its kernel.
        self.fusion_path = FusionPath.from_routing(
            self._is_m4_structured and _drives_the_primitive()
        )

    def __eq__(self, other: object) -> bool:
        # Value identity IS the params surface — required for the pytree-aux
        # seat in `DuplexTranscript` (docs/reference/conventions.md
        # "Pytree registration"). The marker joins the key because it is not a
        # function of the params alone (it tracks the emitter flags): without it
        # a dedicated and a generic perm on the same params collide in the
        # `_permute_body` static-arg cache (the arrangement `SparsePoseidon`
        # states).
        if not isinstance(other, Poseidon2):
            return NotImplemented
        return (
            self._p == other._p
            and self.fused_region_marker == other.fused_region_marker
        )

    def __hash__(self) -> int:
        return hash((self._p, self.fused_region_marker))

    def _select_fused_region_name(self) -> str:
        """Route to the dedicated Poseidon2Fusion for any `(I + J_blocks) ⊗ M4`
        external matrix — the emitter applies the M4 the marker carries as an
        attribute, so Plonky3's `circ(2,3,1,1)` and the HorizenLabs matrix both
        take the dedicated path — where the pin *and* the backend carry the
        emitter. A free-form matrix is not M4-block-structured, and a backend
        without the arm cannot route the name, so both keep the generic marker
        (LoopFusion lowers the real body, staying correct, just slow to
        compile).
        """
        if self._is_m4_structured and _routes_to_dedicated_emitter():
            return POSEIDON2_MARKER
        return FUSED_REGION_MARKER

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

    # Fused-region ABI (see `Permutation.fused_region_spec`).
    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, Any]]:
        """The Poseidon2Fusion ABI: operands `(leading, *round_constants)`, the
        M4-external + internal-diffusion permute, and attrs whose `external_m4`
        names the M4. Dedicated path only (which implies M4-structured);
        otherwise the shared inert stub."""
        if not self.fusion_path.is_one_kernel:
            return inert_region_spec(self, leading)
        p = self._p
        attrs: dict[str, Any] = {
            "permutation": "poseidon2",
            "width": self.width,
            "external_rounds": p.external_rounds,
            "internal_rounds": p.internal_rounds,
            "alpha": p.alpha,
            "external_m4": _external_m4_attr(self),
            "internal_j_scale": _internal_j_scale_attr(self),
        }
        return _abi_operands(self, leading), partial(_permutation_body, self), attrs


def _permutation_body(
    perm: Poseidon2,
    s: Array,
    ext_init_rc: Array,
    int_rc: Array,
    ext_term_rc: Array,
    diag: Array,
) -> Array:
    """The round-loop permute on a single `(width,)` state, taking the
    Poseidon2Fusion ABI operands explicitly: round constants flattened row-major,
    int_rc one constant per partial round. The internal J scale rides as the
    `internal_j_scale` marker attribute (a scalar constant survives `lax.scan`,
    where an operand is hoisted).

    The decomposition every `hash_frx.perm.poseidon2` region runs, spliced inline (the
    generic marker's single-kernel requirement allows no call). A batch is
    `vmap(permute)`, which lowers to the same marker over a batched operand."""
    p = perm._p
    alpha = p.alpha
    w, e_rounds, i_rounds = perm.width, p.external_rounds, p.internal_rounds

    # The Poseidon2Fusion emitter understands only an M4-block-structured
    # external layer; a free-form matrix takes the generic fallback.
    if perm._is_m4_structured:
        m4 = perm._external_m4
        assert m4 is not None  # _is_m4_structured ⇒ M4 was extracted

        def apply_external(state: Array) -> Array:
            return apply_external_m4(state, m4)

    else:
        mds = p.external_matrix

        def apply_external(state: Array) -> Array:
            return apply_matrix(mds, state)

    # +rc -> sbox(all lanes) -> MDS. Shaped as a `lax.scan` step — carry in,
    # carry and an empty per-step output out — so the round sequence below is
    # one loop rather than a copy of the body per round.
    def external_round(state: Array, rc: Array) -> tuple[Array, None]:
        return apply_external(fnp.power(state + rc, alpha)), None

    # The J scale rebuilds from its canonical (attribute) value so the identity
    # default skips the multiply at trace time.
    j_scale_canonical = _internal_j_scale_attr(perm)
    off_diag = (
        None
        if j_scale_canonical == 1
        else fnp.asarray(j_scale_canonical, dtype=p.dtype)
    )

    def internal_round(state: Array, rc0: Array) -> tuple[Array, None]:
        s0 = fnp.power(state[0] + rc0, alpha)
        # concatenate, not state.at[0].set: a static-index set lowers to scatter,
        # which would split the fused kernel.
        state = fnp.concatenate([s0[None], state[1:]])
        return apply_internal(diag, state, off_diag), None

    # Each round sequence is one `lax.scan` over its round constants, not a
    # Python loop that copies the body per round. The constants are the scanned
    # operand, so a round reads its own row at the loop's index rather than a
    # literal spliced into a unique copy of the body.
    #
    # What that buys is downstream: the marker body holds one round, so it
    # inlines to a `while` XLA's generic static_while path lowers in a single
    # kernel — the same one kernel a per-backend emitter is written to produce.
    # Unrolled, the body is `2*external_rounds + internal_rounds` copies, which
    # is what made the emitter worth writing in the first place.
    ext_init = ext_init_rc.reshape(e_rounds, w)
    ext_term = ext_term_rc.reshape(e_rounds, w)
    s = apply_external(s)  # initial pre-MDS
    s, _ = frx.lax.scan(external_round, s, ext_init)
    s, _ = frx.lax.scan(internal_round, s, int_rc)
    s, _ = frx.lax.scan(external_round, s, ext_term)
    return s


def _abi_operands(perm: Poseidon2, state: Array) -> tuple[Array, ...]:
    """The Poseidon2Fusion ABI operands [state, ext_init_rc, int_rc, ext_term_rc,
    diag]; `state` is `(width,)` single or `(n, width)` batched, the constants
    identical either way. The internal matrix is internal_j_scale*J +
    Diag(internal_diag); the J scale rides as the `internal_j_scale` attribute — a
    scalar constant survives `lax.scan` where an operand would be hoisted."""
    p = perm._p
    return (
        state,
        p.external_constants_initial.reshape(-1),
        p.internal_constants,
        p.external_constants_terminal.reshape(-1),
        p.internal_diag,
    )


def _external_m4_attr(perm: Poseidon2) -> np.ndarray:
    """The base M4 flattened row-major as a numpy int64 `(16,)`. A numpy value
    (not a Python list) so it lowers to a `dense<[..]> : tensor<16xi64>` the XLA
    recognizer can parse (a plain list lowers to an unparsed ArrayAttr). Caller
    guards on `fusion_path.is_one_kernel`."""
    assert perm._external_m4 is not None  # dedicated ⇒ M4-structured
    return np.array(perm._external_m4, dtype=np.int64).flatten()


def _internal_j_scale_attr(perm: Poseidon2) -> int:
    """The J scale's canonical value, for the `internal_j_scale` marker attribute
    and the reference body's in-trace constant. A numpy object cast Montgomery-
    decodes without frx x64 (as `Poseidon2Params.external_m4` does); the recognizer
    value-encodes it back per field, so the wire value is field-representation-
    independent (e.g. plain `1` for identity)."""
    return int(np.asarray(perm._p.internal_j_scale).astype(object))


def _marker_attrs(perm: Poseidon2) -> dict[str, object]:
    """The dedicated marker's `composite.attributes`. The permutation
    shape rides as attributes — the XLA recognizer's contract: the four
    shape ints (it maps `alpha` to its s-box degree) plus `external_m4`, the 4×4
    base M4. The
    recognizer rewrites only the M4 its kernel implements (the canonical
    `circ(2,3,1,1)`) and leaves any other matrix to inline its real body, so the
    attr is what identifies the matrix. The body ignores these attrs (metadata
    only); the generic marker stays attrs-free."""
    if not perm.fusion_path.is_one_kernel:
        return {}
    return {
        "width": perm.width,
        "external_rounds": perm._p.external_rounds,
        "internal_rounds": perm._p.internal_rounds,
        "alpha": perm._p.alpha,
        "external_m4": _external_m4_attr(perm),
        "internal_j_scale": _internal_j_scale_attr(perm),
    }


# Module-level jit zone so the permutation body traces once per (params, state
# aval) process-wide: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits hundreds of identical-aval permutes (every
# Merkle level, leaf hash, and transcript observe/sample) — the uncached
# re-trace of this body dominates the first-trace-per-config floor.
# The permutation is the static key, compared by value; `inline=True`
# splices the cached jaxpr into the enclosing trace, so the emitted module
# (one composite marker per permute) is unchanged.
@partial(frx.jit, static_argnames=("perm",), inline=True)
def _permute_body(perm: Poseidon2, state: Array) -> Array:
    def decomposition(
        s: Array,
        ext_init_rc: Array,
        int_rc: Array,
        ext_term_rc: Array,
        diag: Array,
        **_attrs: object,
    ) -> Array:
        # `_attrs` is marker metadata passed through (incl. `internal_j_scale`) —
        # the decomposition itself does not read it. Inlined here so the
        # single-state region stays one straight-line body (the generic marker's
        # single-kernel requirement allows no call).
        return _permutation_body(perm, s, ext_init_rc, int_rc, ext_term_rc, diag)

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
    _: type[Permutation] = Poseidon2

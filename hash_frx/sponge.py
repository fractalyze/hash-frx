"""Sponge hash — scheme-agnostic over a Permutation, one entry per chaining rule.

`Sponge.hash(input, chaining=...)` absorbs in `rate`-sized blocks and squeezes
the first `out` lanes. `chaining` (`SpongeChaining`) picks how the chain crosses
capacity via a `_MODES` row, so a new one is a member + a row, never a method.

That extension point covers sponges shaped like this one: 1-D field elements,
an overwrite absorb, and a squeeze that truncates the final state. A sponge that
differs in its *input domain* is not a row — a byte-oriented one takes a `[B, L]`
batch and needs a packing layer no other row would use, which is why Keccak has
its own ([`keccak/sponge.py`](keccak/sponge.py)) and why `duplex_sponge.py`
refuses an absorb-mode flag for the same reason.

`rate`/`out` are the free params on `SpongeParams` (capacity = width - rate). A
`DEDICATED`-path permutation lowers the whole absorb to one
`hash_frx.digest.field_sponge` region (assembled here over the permutation's
fused-region ABI); a non-fused one runs the `while_loop` absorb over `permute`.
Both read the absorb length at runtime, so concrete and symbolic `n` lower the
same way.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import frx
import frx.numpy as fnp
from frx import Array

from hash_frx.extension.sponge import field_absorb
from hash_frx.fusion import FusionPath, fused_region_over
from hash_frx.permutation import Permutation


class SpongeChaining(enum.Enum):
    """How the chain crosses capacity between blocks (`Sponge.hash(..., chaining=...)`).

    Both members share the sponge skeleton — absorb into rate, permute, squeeze —
    and differ on one axis only: whether capacity carries the chain on its own or
    the prior block's digest is written into it. Behaviour is a `_MODES` row, so a
    new one is a member + a row, not a method.

    Naming this axis after Merkle–Damgård, as an earlier spelling did, described a
    whole other construction: an axis-B domain extension rather than a discipline
    inside this one. `sha256_merkle_damgard` names Merkle–Damgård because it *is*
    Merkle–Damgård; nothing here is (`docs/blocks/hash.md`, "Names say the
    construction, not the shape").

    **The values are a frozen wire spelling, not the concept names.** Each rides
    the `hash_frx.digest.field_sponge` composite as `construction=` and the
    emitter switches on it, so changing one is a wire-ABI change needing
    dual-spelling recognition in Fractalyze XLA first (`fusion.py`; #165 is the
    worked example). They stay as they are until that is staged.
    """

    # Plonky3 PaddingFreeSponge (default): masked lanes keep the prior state, so
    # untouched capacity carries the chain implicitly.
    IMPLICIT_CAPACITY = "padding_free"
    # The prior block's digest is written into capacity lanes [rate:rate+out].
    DIGEST_IN_CAPACITY = "merkle_damgard"


# The whole absorb+squeeze as one region the vendor expands into the fused kernel.
# Attrs carry a `permutation` discriminator + shape; the kernel reads the absorb
# length at runtime, so one cubin serves every width and symbolic `n`.
SPONGE_HASH_MARKER = "hash_frx.digest.field_sponge"
SPONGE_HASH_MARKER_VERSION = 1


@dataclass(frozen=True)
class _Mode:
    """One chaining discipline's behaviour — the whole per-member surface. A new
    one is a single `_MODES` row (a `SpongeChaining` member + its two hooks), no
    new module-level functions.

    `tail_fill(state, rate)` fills a partial block's masked rate lanes;
    `set_capacity(state, cap, rate, out)` places the prior digest `cap`
    (`state[:out]`) into capacity."""

    tail_fill: Callable[[Array, int], Array]
    set_capacity: Callable[[Array, Array, int, int], Array]
    # The digest fills the whole capacity, so the caller must have
    # rate + out == width (the out-lane digest is carried as capacity).
    fills_capacity: bool


_MODES: dict[SpongeChaining, _Mode] = {
    # Masked lanes keep the prior state; capacity carries the chain on its own,
    # so set_capacity is a no-op.
    SpongeChaining.IMPLICIT_CAPACITY: _Mode(
        tail_fill=lambda s, rate: s[:rate],
        set_capacity=lambda s, cap, rate, out: s,
        fills_capacity=False,
    ),
    # Masked lanes are zero-padded; capacity lanes [rate:rate+out] take the
    # prior block's digest.
    SpongeChaining.DIGEST_IN_CAPACITY: _Mode(
        tail_fill=lambda s, rate: s[:rate] - s[:rate],
        set_capacity=lambda s, cap, rate, out: s.at[rate : rate + out].set(cap),
        fills_capacity=True,
    ),
}


def _absorb(
    input: Array,
    state: Array,
    rate: int,
    out: int,
    permute: Callable[[Array], Array],
    chaining: SpongeChaining,
) -> Array:
    """Absorb over ``ceil(n/rate)`` blocks and squeeze the first ``out`` lanes —
    shared by every `SpongeChaining`, whose `tail_fill`/`set_capacity` are the only
    per-member pieces.

    The schedule is `extension.sponge.field_absorb`: a ``while_loop``, so the
    bound is read at runtime and concrete and symbolic ``n`` lower the same way.
    That loop form is what keeps this construction out of the byte sponges'
    `absorb_squeeze`, whose counts are static — the two schedules are siblings
    in `extension/sponge.py` rather than one body with the loop as a parameter.

    The squeeze is a truncation of the final state rather than an iterated read,
    which is why nothing here needs the byte schedule's permute-between-reads
    rule."""
    mode = _MODES[chaining]
    n = input.shape[0]
    nb = (n + rate - 1) // rate
    lanes = fnp.arange(rate)

    def merge(s: Array, i: Array) -> Array:
        start = i * rate
        w = fnp.minimum(rate, n - start)
        # Last block reads past n; clamp OOB indices (masked out below).
        block = input[fnp.clip(start + lanes, 0, n - 1)]
        cap = s[:out]  # prior digest (zeros on block 0); snapshot before overwrite
        s = s.at[:rate].set(fnp.where(lanes < w, block, mode.tail_fill(s, rate)))
        return mode.set_capacity(s, cap, rate, out)

    state = field_absorb(state, blocks=nb, absorb=merge, permute=permute)
    return state[:out]


def _fused_hash(
    perm: Permutation,
    input: Array,
    rate: int,
    out: int,
    chaining: SpongeChaining,
) -> Array:
    """Absorb+squeeze as ONE `hash_frx.digest.field_sponge` region over a
    dedicated-fusion permutation. `fused_region_over` rebuilds `permute` from
    the ABI operands — the emitter's operand contract names them there — so the
    fallback HLO matches the generic path. Caller gates on
    `fusion_path.is_one_kernel`."""

    def sponge(inp: Array, permute: Callable[[Array], Array]) -> Array:
        state = fnp.zeros(perm.width, dtype=inp.dtype)
        return _absorb(inp, state, rate, out, permute, chaining)

    # This sponge's shape (`rate`/`digest_elems`) and `construction` — the string
    # the kernel switches on (extensible to any `SpongeChaining`) — alongside the
    # permutation's own attrs. The marker name/version are the sponge's.
    return fused_region_over(
        perm,
        input,
        sponge,
        name=SPONGE_HASH_MARKER,
        version=SPONGE_HASH_MARKER_VERSION,
        rate=rate,
        digest_elems=out,
        construction=chaining.value,
    )


@dataclass(frozen=True)
class SpongeParams:
    """Free parameters of a sponge hash, whichever `SpongeChaining` it runs.

    rate : field elements absorbed per permutation (capacity = width - rate).
    out  : field elements squeezed (the digest size).

    Contract: rate >= 1 and out >= 1 (validated here); rate < permutation.width
    and out <= permutation.width (validated by ``Sponge``, which knows the width).
    """

    rate: int
    out: int

    def __post_init__(self) -> None:
        if self.rate < 1:
            raise ValueError(f"rate ({self.rate}) must be >= 1")
        if self.out < 1:
            raise ValueError(f"out ({self.out}) must be >= 1")


class Sponge:
    """Sponge hash over a fixed-width Permutation.

    `hash(input, chaining=...)` is the one entry point (one call = one function
    = one fused `hash_frx.digest.field_sponge` kernel). The permutation supplies
    only its arithmetic — `permute` and its fused-region ABI; the construction
    lives here.
    """

    def __init__(self, permutation: Permutation, params: SpongeParams) -> None:
        if params.rate >= permutation.width:
            raise ValueError(
                f"rate ({params.rate}) must be < permutation width "
                f"({permutation.width})"
            )
        if params.out > permutation.width:
            raise ValueError(
                f"out ({params.out}) must be <= permutation width ({permutation.width})"
            )
        self._permutation = permutation
        self.rate = params.rate
        self.out = params.out

    # Value equality/hash for static jit-zone keys, like the Permutation
    # contract it builds on — identity equality re-traces per instance.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sponge):
            return NotImplemented
        return (self._permutation, self.rate, self.out) == (
            other._permutation,
            other.rate,
            other.out,
        )

    def __hash__(self) -> int:
        return hash((self._permutation, self.rate, self.out))

    @property
    def fusion_path(self) -> FusionPath:
        """How the underlying permute lowers on this backend — `DEDICATED` is
        what lets a consumer wrap a whole region using this hash in an
        expandable composite. Delegates to the permutation."""
        return self._permutation.fusion_path

    @property
    def dtype(self) -> Any:
        """The base field the sponge absorbs and squeezes (the permutation's)."""
        return self._permutation.dtype

    def hash(
        self, input: Array, chaining: SpongeChaining = SpongeChaining.IMPLICIT_CAPACITY
    ) -> Array:
        """Absorb `input` (1-D) and squeeze the first `out` lanes: (n,) -> (out,).

        `chaining` picks how the chain crosses capacity (see `SpongeChaining`). A
        `DEDICATED`-path permutation lowers to one `hash_frx.digest.field_sponge`
        region; a non-fused one runs the `while_loop` absorb. Both read the absorb
        length at runtime, so symbolic `n` lowers like a concrete one.
        """
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        # Else a mismatch surfaces deep in the absorb as an opaque promotion error.
        if input.dtype != self.dtype:
            raise TypeError(
                f"input dtype {input.dtype} must match the sponge field {self.dtype}"
            )
        # A capacity-filling rule (DIGEST_IN_CAPACITY) needs rate + out == width.
        if _MODES[chaining].fills_capacity and (
            self.rate + self.out != self._permutation.width
        ):
            raise ValueError(
                f"{chaining.value} hash needs rate + out == width (the digest "
                f"fills the capacity), got {self.rate} + {self.out} != "
                f"{self._permutation.width}"
            )
        # Zero absorbed blocks leave the state at its zero initialization, so
        # the digest is the zero prefix (Plonky3's PaddingFreeSponge on an
        # empty iterator; the digest-feedback mode chains nothing either).
        # Static, so no marker or permute is emitted for a constant result —
        # without this the absorb gather slices a length-0 input and fails.
        if input.shape[0] == 0:
            return fnp.zeros(self.out, dtype=input.dtype)
        return _hash_body(self, input, chaining)


# Module-level jit zone so the hash decomposition traces once per (sponge,
# chaining, input aval) process-wide: `lax.composite` re-traces its
# decomposition on every emission and again under every vmap batching (see
# `hash_frx._composite`), and one Merkle commit emits hundreds of identical-aval
# leaf hashes. The sponge (permutation + rate + out, compared by value) and the
# chaining rule are the static key — together they fully determine the
# decomposition; `inline=True` splices the cached jaxpr into the enclosing trace,
# so the emitted module (one `hash_frx.digest.field_sponge` marker per hash on the
# dedicated path) is unchanged and a recognizing emitter still sees the composite
# form. Same hoist as `hash_frx.poseidon2.poseidon2._permute_body`.
#
# Being output-invariant is exactly why it needs its own test: byte-identity and
# lowered-module comparison both hold with or without it, so only a trace count
# can see it.
@partial(frx.jit, static_argnames=("sponge", "chaining"), inline=True)
def _hash_body(sponge: Sponge, input: Array, chaining: SpongeChaining) -> Array:
    # Dedicated-fusion → one `hash_frx.digest.field_sponge` region (built here);
    # else the generic `while_loop` absorb over `permute`.
    perm = sponge._permutation
    if perm.fusion_path.is_one_kernel:
        return _fused_hash(perm, input, sponge.rate, sponge.out, chaining)
    state = fnp.zeros(perm.width, dtype=input.dtype)
    return _absorb(input, state, sponge.rate, sponge.out, perm.permute, chaining)

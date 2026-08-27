"""Add-absorb duplex sponge — scheme-agnostic over a Permutation.

A stateful sponge supporting interleaved absorb/squeeze (duplex): absorb ADDS
input into the rate lanes (`state[:rate] += block`, not overwrite — contrast the
one-shot overwrite `Sponge`), permuting when a rate block fills or the duplex
direction switches; squeeze reads the rate lanes, permuting when they drain.
This is the agnostic primitive a classic Fiat-Shamir prover (e.g. an
ark-crypto-primitives-0.5-faithful accumulation prover) drives; the
scheme-specific challenge packing, domain separation, and field conversions live
in the consumer.

A consumer that must byte-match a specific release names its convention:
[`adapter/duplex.py`](adapter/duplex.py) ships `ARK_0_3` and `ARK_0_5`, and
`DuplexSponge(perm, rate, convention=ARK_0_3)` reproduces that release exactly.
The default is neither of them (`DEFAULT_CONVENTION` below). Three axes separate
the two — where the rate sits, whether a spilling squeeze always permutes, and
whether a squeeze-to-absorb switch does — and each is a `RateLanes` /
`SpillPermute` / `SwitchPermute` member here, so a convention says which
mechanism it changes rather than that something is odd about it.

Unlike its siblings, this one constrains the permutation's dtype, and enforces it
in `__init__` rather than only stating it: the absorb merge is `+`, so `+` must be
the intended group operation. That holds for every field dtype — prime,
extension, and binary, where `+` is XOR — and fails for machine words, where `+`
carries between bits. A Keccak-style sponge merges by XOR, so running one here
would wrap rather than raise and compute wrong bytes silently; a rejected dtype
is the loud failure that replaces it. A bit-oriented sponge therefore belongs in
its own construction, not behind an absorb-mode flag here, for the same reason
`DuplexTranscript` is separate (below).

Kept separate from `DuplexTranscript` (the overwrite-mode Fiat-Shamir sponge),
not unified under an absorb-mode flag: the two implement different sponge
conventions and diverge on three independent axes — the absorb merge (add here
vs overwrite there), the squeeze read direction (this reads the rate low→high;
`DuplexTranscript` pops it high→low, so a partial squeeze returns different lanes,
not merely reversed ones), and the permute timing (this defers the permute on an
exactly-filled rate block). A shared core would have to parameterize all three — two
conventions in one config object, not real reuse — so the genuinely shared part
(a buffer, a position, a permute call) does not justify merging them.

Width comes from `permutation.width`; `rate` is the free parameter
(capacity = width - rate). The absorb/squeeze schedule is static (known element
counts), so the mode machine and permute triggers resolve at trace time —
mode/position are Python-level, only the field-element state is traced.

Unlike the one-shot `Sponge`/`Compression` (static configs used as jit-zone
keys), this carries traced per-step state and is threaded by return value, so it
deliberately omits the value-equality/hash those siblings define; pytree
registration and a static-key surface are left to the consumer that threads it
through `jit`, where the threading pattern can be validated rather than guessed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import DTypeLike

from hash_frx.fusion import FusionPath
from hash_frx.permutation import Permutation

_ABSORBING = "absorbing"
_SQUEEZING = "squeezing"


class RateLanes(enum.Enum):
    """Which end of the state the rate occupies — the axis two implementations
    of this construction disagree on before they disagree on anything else.

    Both halves are contiguous and the permutation runs over the whole state, so
    this changes nothing about security and everything about which lanes a given
    absorb writes and a given squeeze reads. A sponge on the wrong end of it
    produces well-formed field elements that match no other implementation.
    """

    LEADING = "leading"
    """Rate is `state[:rate]`, capacity the tail. ark-sponge 0.3, Plonky3."""

    TRAILING = "trailing"
    """Rate is `state[capacity:]`, capacity the head. ark-crypto-primitives 0.5."""


class SpillPermute(enum.Enum):
    """When a squeeze that outruns the current rate block permutes.

    `ALWAYS` is the correct rule and the only one a new consumer should want;
    the other exists because a shipped implementation got it wrong and other
    code now depends on the wrong bytes.
    """

    ALWAYS = "always"
    """Permute whenever the request outruns the block — ark-crypto-primitives 0.5."""

    SKIP_WHEN_REQUEST_EQUALS_RATE = "skip_when_request_equals_rate"
    """Skip it when the still-outstanding request happens to equal the rate.

    ark-sponge 0.3's bug, reproduced deliberately. Its `squeeze_internal` tests
    `output_remaining.len() != self.rate` **before** advancing the output slice,
    so at exactly that length the permute is skipped and the next read comes off
    an unpermuted state — re-emitting lanes an earlier squeeze already returned.
    Only reachable with a non-zero rate offset, which the loop condition forces.
    """


class SwitchPermute(enum.Enum):
    """Which duplex direction switches permute before the next operation.

    Absorb-to-squeeze permutes in every implementation of this construction —
    reading the rate without first mixing what was just absorbed would return
    the absorbed data itself. The squeeze-to-absorb direction is the one they
    disagree on, and the disagreement is invisible to any script that never
    turns back around, which is why the reference set carries one that does.
    """

    BOTH_DIRECTIONS = "both_directions"
    """Permute on either switch — ark-sponge 0.3."""

    ABSORB_TO_SQUEEZE_ONLY = "absorb_to_squeeze_only"
    """Permute only when absorbing gives way to squeezing — ark-crypto-primitives
    0.5, whose `absorb` dropped the `self.permute()` its `Squeezing` arm carried
    in 0.3. New material is then added into rate lanes a squeeze has already
    published, with no permutation in between."""


@dataclass(frozen=True)
class DuplexConvention:
    """Which implementation's duplex a sponge reproduces, as its axes.

    Named by mechanism rather than by quirk: `SKIP_WHEN_REQUEST_EQUALS_RATE`
    says what the sponge does, where a flag called `legacy_spill` would say only
    that something is odd. The named members that pair these axes with the
    releases they reproduce are in
    [`adapter/duplex.py`](adapter/duplex.py) — a convention is a statement about
    an external codebase, which is an adapter's business rather than this
    construction's.

    Frozen, so it is value-compared and hashable: a consumer can hand it to
    `jit` as a static argument, and two sponges built with the same convention
    share a trace.
    """

    rate_lanes: RateLanes
    spill_permute: SpillPermute
    switch_permute: SwitchPermute


DEFAULT_CONVENTION = DuplexConvention(
    RateLanes.LEADING, SpillPermute.ALWAYS, SwitchPermute.BOTH_DIRECTIONS
)
"""This construction's own convention, and **not** either ark release.

Leading rate is ark-sponge 0.3's layout; the always-permute squeeze is the
behaviour 0.5 corrected to. A consumer that needs to byte-match either release
names it (`adapter/duplex.py`); this default exists so one that needs neither
does not have to choose.
"""


def _merges_by_addition(dtype: DTypeLike) -> bool:
    """Whether `+` on `dtype` is the group operation this sponge absorbs with.

    A machine word is the one case where it is not: `+` carries between bits, so
    a sponge that should merge by XOR would wrap instead. Every field dtype
    qualifies — including a binary field, whose `+` *is* XOR.
    """
    return not np.issubdtype(np.dtype(dtype), np.integer)


class DuplexSponge:
    """Add-absorb duplex sponge over a fixed-width Permutation."""

    def __init__(
        self,
        permutation: Permutation,
        rate: int,
        convention: DuplexConvention = DEFAULT_CONVENTION,
    ) -> None:
        if rate < 1:
            raise ValueError(f"rate ({rate}) must be >= 1")
        if rate >= permutation.width:
            raise ValueError(
                f"rate ({rate}) must be < permutation width ({permutation.width})"
            )
        # Caught here rather than left to produce wrong bytes in `absorb`.
        if not _merges_by_addition(permutation.dtype):
            raise TypeError(
                f"dtype {permutation.dtype} merges by carrying addition, but this "
                "sponge absorbs with +; a bit-oriented permutation needs an "
                "XOR-absorb sponge, not this one"
            )
        self._permutation = permutation
        self.rate = rate
        self.convention = convention
        # Where lane 0 of the rate sits in the state. Resolved once here rather
        # than branched at every read: the axis is fixed for the sponge's life,
        # and a branch per slice would put a Python conditional inside the
        # traced schedule for no gain.
        self._rate_base = (
            0
            if convention.rate_lanes is RateLanes.LEADING
            else permutation.width - rate
        )
        self._state = fnp.zeros(permutation.width, dtype=permutation.dtype)
        self._mode = _ABSORBING
        self._pos = 0

    @property
    def fusion_path(self) -> FusionPath:
        """How the underlying permute lowers on this backend — `DEDICATED` is
        what lets a consumer wrap a whole region using this hash in an
        expandable composite. Delegates to the permutation; names no hash."""
        return self._permutation.fusion_path

    def _with(self, *, state: Array, mode: str, pos: int) -> "DuplexSponge":
        new = object.__new__(DuplexSponge)
        new._permutation = self._permutation
        new.rate = self.rate
        new.convention = self.convention
        new._rate_base = self._rate_base
        new._state = state
        new._mode = mode
        new._pos = pos
        return new

    def absorb(self, elems: Array) -> "DuplexSponge":
        if elems.ndim != 1:
            raise ValueError(f"elems must be 1-D, got ndim={elems.ndim}")
        if elems.shape[0] == 0:
            return self  # empty input never touches the state (no direction switch)
        state, pos = self._state, self._pos
        if self._mode == _SQUEEZING:
            # Direction switch (squeeze -> absorb) resets to rate 0, and permutes
            # first under every convention but one: ark-crypto-primitives 0.5
            # dropped the permute its 0.3 `absorb` did here, so new material lands
            # on rate lanes a squeeze has already published.
            if self.convention.switch_permute is SwitchPermute.BOTH_DIRECTIONS:
                state = self._permutation.permute(state)
            pos = 0
        state, pos = self._absorb_into_rate(state, pos, elems)
        return self._with(state=state, mode=_ABSORBING, pos=pos)

    def _absorb_into_rate(
        self, state: Array, start: int, elems: Array
    ) -> tuple[Array, int]:
        # Add elements into the rate lanes from `start`; when they spill past
        # the rate block, add what fits, permute, and continue on the fresh
        # block. A loop rather than recursion: the block count is input-length /
        # rate, and recursing per block hit Python's recursion limit near a
        # thousand blocks.
        base = self._rate_base
        while True:
            n = elems.shape[0]
            if start + n <= self.rate:
                return state.at[base + start : base + start + n].add(elems), start + n
            take = self.rate - start
            state = state.at[base + start : base + self.rate].add(elems[:take])
            state = self._permutation.permute(state)
            elems = elems[take:]
            start = 0

    def squeeze(self, n: int) -> tuple["DuplexSponge", Array]:
        if n < 0:
            raise ValueError(f"n ({n}) must be >= 0")
        state, pos = self._state, self._pos
        if self._mode == _ABSORBING:
            # Direction switch (absorb -> squeeze) permutes and resets to rate 0.
            state = self._permutation.permute(state)
            pos = 0
        elif pos == self.rate:
            # Rate fully drained: permute before reading the next block.
            state = self._permutation.permute(state)
            pos = 0
        state, pos, out = self._squeeze_from_rate(state, pos, n)
        return self._with(state=state, mode=_SQUEEZING, pos=pos), out

    def _squeeze_from_rate(
        self, state: Array, start: int, n: int
    ) -> tuple[Array, int, Array]:
        # Read n elements from the rate lanes starting at `start`; when the
        # request drains past the rate block, read what is there, permute, and
        # continue on the fresh block.
        #
        # Whether that permute is unconditional is the convention's second axis.
        # `ALWAYS` is ark-crypto-primitives 0.5's `if !output_remaining.is_empty()`
        # and the rule anything new should want.
        # `SKIP_WHEN_REQUEST_EQUALS_RATE` is ark-sponge 0.3's
        # `if output_remaining.len() != self.rate`, tested BEFORE the output
        # slice advances — so the still-outstanding `n` is the one compared, and
        # skipping leaves the next read on an unpermuted state, re-emitting
        # lanes an earlier squeeze already returned.
        #
        # Block reads are collected and concatenated once, so the traced copy is
        # linear in the squeeze length rather than quadratic in the block count.
        base = self._rate_base
        skip_at_rate = (
            self.convention.spill_permute is SpillPermute.SKIP_WHEN_REQUEST_EQUALS_RATE
        )
        chunks = []
        while start + n > self.rate:
            chunks.append(state[base + start : base + self.rate])
            take = self.rate - start
            if not (skip_at_rate and n == self.rate):
                state = self._permutation.permute(state)
            n -= take
            start = 0
        chunks.append(state[base + start : base + start + n])
        out = chunks[0] if len(chunks) == 1 else fnp.concatenate(chunks)
        return state, start + n, out

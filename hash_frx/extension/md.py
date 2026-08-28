# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Merkle-Damgard: the schedule that feeds a compression function a message.

Two shapes, and both are here because both are device code. `chain` is the
one-shot marked region — broadcast the midstate, walk the blocks, serialize —
and `MdStream` is the incremental one, whose absorb and finalize were
transcribed once per SHA-2 family before this.

The padding rule they schedule around lives in `pad.py`, which is pure host
arithmetic and deliberately frx-free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import frx
import frx.numpy as fnp
from frx import Array

from hash_frx.extension.pad import PadRule, Trailer
from hash_frx.fusion import fused_region
from hash_frx.markers import (
    STREAM_ABSORB_MARKER,
    STREAM_ABSORB_MARKER_VERSION,
    STREAM_FINALIZE_MARKER,
    STREAM_FINALIZE_MARKER_VERSION,
)
from hash_frx.word import unpack_be, unpack_le


def masked_chain(
    state: Array,
    *,
    count: int,
    compress_block: Callable[[Array, int], Array],
    live: Array | None = None,
) -> Array:
    """Fold `count` blocks into `state` one at a time, keeping only the first
    `live` of them. Returns the final midstate; serializing it is the caller's,
    since Grøstl finishes with Ω rather than a serialize.

    `live` is what separates the two walks a family can need, and it is the whole
    reason this is one function:

    - **`None`** — every block is a message block. The static walk `chain` runs,
      where the block count came from the message's shape. No predicate is
      emitted at all, so this branch is `tree.chain`'s body line for line. They
      stay apart for the reason `tree.chain` already gives — a chunk's blocks
      are independent of each other and a message's are not, which is what makes
      the tree parallel — and because `tree.py` carries no deps at all. The
      shape being shared is what the two schedules have in common; the
      construction is not.
    - **a traced block count** — the runtime-length forms, where the count is
      data. Every block the buffer could need is compressed and the ones past
      the message are selected away, which is unconditional work rather than a
      branch: the count is not known at trace time, and a `while` would be a
      control-flow boundary the marked region cannot contain
      (`padded_message_region` says why the speculation is affordable).

    In both cases the loop is a Python `for` over a static, small count, per the
    unrolled-round-loop rule in docs/reference/conventions.md.

    **`compress_block(state, i)` takes the block's INDEX, not the block.** Where
    block `i` comes from, how it is shaped, and which constants it folds are all
    the caller's — this schedule's only emitted op is the select. That is the
    shape both sibling schedules already take (`tree.chain`,
    `sponge.absorb_squeeze`) and it is taken for the reason `sponge.py` writes
    down: a helper that receives a block has to emit that block first, which
    fixes one emission order for every caller and silently excludes the ones
    that build theirs in the other order. A schedule that emits nothing imposes
    no order, so a flat byte region (Grøstl slices) and a packed word region
    (everything else indexes) both ride it unchanged, and a family whose
    compression needs its position — BLAKE2's HAIFA counter is `(i, nblocks)` —
    reads it off `i` instead of being locked out by the arity.
    """
    for i in range(count):  # static, small
        folded = compress_block(state, i)
        state = folded if live is None else fnp.where(i < live, folded, state)
    return state


def chain(
    h0: Array,
    blocks: Array,
    *,
    constants: Array,
    compress_block: Callable[[Array, Array, Array], Array],
    serialize: Callable[[Array], Array],
    marker: tuple[str, int],
    primitive: str,
) -> Array:
    """The compression chain from midstate `h0` over `blocks`, as one marked
    region: `[B, nblocks, ...]` -> the serialized final state.

    **The loop is the schedule, so it is not the primitive's.** `compress_block`
    absorbs one block; walking the blocks is Merkle-Damgard's job, and every
    family that had its own copy wrote the same `for i in range(nblocks)` over a
    static, small count. The walk itself is `masked_chain`, which this calls
    unmasked.

    `h0` is the shared midstate — an initial state for a whole-message digest,
    or a resumed one, which is what lets a streaming transcript and a batch
    digest share a single marker. It broadcasts across the batch rather than
    being materialized per row.

    **The operand order is the wire ABI.** `[h0, constants, blocks]` is what the
    recognizing emitters read positionally, so all three are passed explicitly
    rather than captured: a captured constant is lifted into an unnamed operand
    ahead of the declared ones and lands at position 0, silently handing the
    emitter a different message (`hash_frx.fusion` states the rule). That is why
    `constants` is threaded through untouched — the chain never reads the table,
    it only has to put it in the right place.

    `primitive` is the compression's key in the plugin's primitive registry —
    the same string the entry is registered under, which is what an
    operation-named marker resolves through instead of a name suffix. It rides
    as the `primitive` composite attribute and is emitted under BOTH spellings:
    inert while the family still carries its own name, already in place when
    `markers.words_in_digest_marker` flips to the operation name, so the flip is
    a rename and nothing else.

    `marker` is the (name, version) pair the region carries. A whole-hash MD
    marker is exempt from the generic single-kernel rule the way a permutation's
    is: a compression is a round loop rather than straight-line, so it takes a
    name-routed marker an emitter expands. With no emitter wired the marker
    inlines its decomposition and the bytes are unchanged.
    """
    name, version = marker

    def decomposition(
        h0: Array, constants: Array, blocks: Array, **_attrs: object
    ) -> Array:
        # Width comes off the operand: `h0` is the unbatched midstate, so any
        # other value would be a broadcast error rather than a different
        # outcome — a parameter that cannot change a result is one nobody needs.
        state = fnp.broadcast_to(h0, (blocks.shape[0], h0.shape[-1]))
        return serialize(
            masked_chain(
                state,
                count=blocks.shape[1],
                compress_block=lambda s, i: compress_block(s, blocks[:, i], constants),
            )
        )

    return fused_region(
        decomposition,
        h0,
        constants,
        blocks,
        name=name,
        version=version,
        primitive=primitive,
    )


class _Midstate(Protocol):
    """What every family's streaming state carries, whatever it calls itself.

    The state stays a per-family pytree — `Sha256State` and `Sha512State` are
    registered types a consumer threads through `scan`, and collapsing them into
    one would change every consumer's type. What is shared is the SHAPE, so
    `MdStream` codes against this and each family keeps its own class.
    """

    # Read-only, because the concrete states are frozen dataclasses and a
    # Protocol declaring a mutable attribute does not match one.
    @property
    def h(self) -> Array:
        """The midstate over every complete block so far."""

    @property
    def pending(self) -> Array:
        """The trailing partial block, valid prefix `[:pending_len]`."""

    @property
    def counts(self) -> Array:
        """`int32[2] = [pending_len, total_len]`, the packed stream position.

        Required as the LEAF rather than as its two properties because it is a
        positional operand of the stream-finalize ABI: restacking the two
        halves would emit the slice/reshape/concatenate glue outside the marked
        region, which is the glue the marker exists to remove. `absorb` already
        builds it in this layout, so this states a requirement that was already
        load-bearing.
        """

    @property
    def pending_len(self) -> Array:
        """Bytes currently buffered in `pending`."""

    @property
    def total_len(self) -> Array:
        """Total bytes absorbed so far."""


def trailer_field(pad: PadRule, msg_bytes: Array) -> Array:
    """`PadRule.tail`'s strengthening bytes, at a RUNTIME length:
    uint8 [pad.reserve].

    The traced half of `PadRule.tail`, and written to be diffed against it line
    for line: the same switch on `pad.trailer` for the VALUE, the same
    `pad.big_endian` for its byte order, and the same `pad.reserve` width with
    the eight-byte field at the END of it. `tail` builds those bytes on the host
    from a static length; a runtime-length path cannot, so it selects them
    in-graph instead — and every axis it has to branch on is already a named
    field on the rule rather than a fact about which file the caller is in.

    Before this existed the branch was hand-written once per family, and the
    copies had begun to disagree about what they even took: SHA-256's took a
    byte count because a BIT length is not otherwise in hand, and Grøstl's took
    a block count because its caller had already derived one. Three spellings
    of `unpack_be(stack([hi, lo]))` for one encoding, in a package whose
    conventions state that "a tenth copy is a regression rather than a new
    family".

    **The value.** `BIT_LENGTH` is the byte count times eight, carried as a
    uint32 half pair because the product outgrows one word — a count near 2^31
    has a bit length near 2^34, so a single word wraps at 512 MiB and encodes a
    shorter message, which is the same digest for two different lengths.
    `BLOCK_COUNT` is `nblocks`, whose high word is zero outright: the count
    rides as an int32, so the block count cannot reach 2^32. It is derived in
    int32 and cast after, so the expression is the one a caller sizing its
    region has already written and XLA folds the two together.

    `Trailer.NONE` has no field to build — HAIFA feeds the length to the
    compression instead — so it is rejected here rather than answered with
    zeros, the way `PadRule.__post_init__` rejects a trailer with too few
    reserved bytes.
    """
    if pad.trailer is Trailer.NONE:
        raise ValueError(
            "Trailer.NONE has no length field: HAIFA carries the length as a "
            "counter into the compression (`pad.haifa_counter`), not in the "
            "padding"
        )
    if pad.trailer is Trailer.BIT_LENGTH:
        count = msg_bytes.astype(fnp.uint32)
        high, low = count >> fnp.uint32(29), count << fnp.uint32(3)
    else:
        high = fnp.uint32(0)
        low = fnp.asarray(pad.nblocks(msg_bytes), dtype=fnp.uint32)
    # The word order flips with the byte order: big-endian is high word first,
    # little-endian is the low word's low byte first. Splitting this the wrong
    # way is a byte-reversed field, which `pad_traced_test` catches by holding
    # every family's rule against the host `tail` it mirrors.
    field = (
        unpack_be(fnp.stack([high, low]))
        if pad.big_endian
        else unpack_le(fnp.stack([low, high]))
    )
    if pad.reserve == 8:
        # Not a readability choice, and not skippable: a zero-width concatenate
        # does NOT fold away here. Dropping this arm and always concatenating
        # measured +1 `concatenate`, +1 `broadcast_in_dim` and +1 `constant` in
        # every lowered digest of every reserve-8 family — SHA-256's lowered
        # module stops being byte-identical to the one it had before the field
        # was shared, which is the property this extraction is held to.
        return field
    # A wider reservation is zero-filled AHEAD of the field, because `tail`
    # writes the eight bytes at `[-8:]` whatever `reserve` is — SHA-512 claims
    # sixteen bytes for a 128-bit length it never fills past sixty-four bits.
    return fnp.concatenate([fnp.zeros(pad.reserve - 8, dtype=fnp.uint8), field])


def padded_region(
    pad: PadRule,
    content: Array,
    content_len: Array,
    active_bytes: Array,
    len_bytes: Array,
) -> Array:
    """A padded block region built at a RUNTIME length: uint8 [B, N] -> [B, N].

    `content` holds the message bytes in the first `content_len` slots of an
    N-slot region (N a whole number of blocks); what lies past them is unread,
    so a caller may clamp or gap-shift its gather however suits it. The result
    is `content[:content_len] ‖ 0x80 ‖ 0x00* ‖ len_bytes`, with the length field
    ending at `active_bytes` and everything past it zero.

    This is `PadRule.tail` for the paths that cannot use it: the tail is a host
    constant built from a STATIC length, and here the length is traced. So the
    same three regions are selected in-graph off an index vector instead.

    Shared rather than spelled per caller because that spelling is the subtlest
    traced-index machinery in the package and getting it wrong is a wrong digest
    rather than a slow one — the hazard `MdStream` below was extracted to end.

    Two callers build a region, and only one of them still passes its own
    gather. `MdStream.finalize`'s is gap-shifted around the pending block, so it
    stays there. The whole-message one is `padded_message_region` below: it was
    a caller's too while there was a single family writing it, and the moment
    there were two the five lines were character-identical, comment included. A
    new runtime-length family calls that rather than spelling this.

    The padding is built as ONE row and broadcast against `content`, the way
    `byte_hash.padded_batch` broadcasts the static tail: it is a function of the
    length, which every row of a batch shares.
    """
    reserve = pad.reserve
    pos = fnp.arange(content.shape[-1], dtype=fnp.int32)
    len_start = active_bytes - fnp.int32(reserve)
    padding = fnp.where(
        pos == content_len,
        fnp.uint8(0x80),
        fnp.where(
            (pos >= len_start) & (pos < active_bytes),
            len_bytes[fnp.clip(pos - len_start, 0, reserve - 1)],
            fnp.uint8(0),
        ),
    )
    return fnp.where((pos < content_len)[None, :], content, padding[None, :])


def padded_message_region(pad: PadRule, buf: Array, length: Array) -> Array:
    """`buf[:, :length]` padded to whole blocks at a RUNTIME length:
    uint8 [B, LMAX] plus an int32 scalar -> uint8 [B, NB * pad.block_size].

    The whole-message half of `padded_region`, which that function's docstring
    deferred while there was one caller to defer it for: *"a whole-message
    runtime-length digest builds one as wide as its buffer. Only the gather and
    the width differ, which is why those stay with the callers."* Two callers
    later the gathers were character-identical, comment included, so the premise
    had stopped holding and the five lines moved here.

    Everything the caller still owns is what it does with the result: SHA-256
    packs the region into big-endian words, Grøstl-256 compresses the bytes as
    they are. Only the region is common, so only the region moved — the rule
    `byte_hash.padded_batch` states for the static tail.

    **Merkle-Damgard rules with a trailer only.** A `Trailer.NONE` rule is
    rejected here rather than two frames down: HAIFA pads with a bare zero fill
    and feeds its length to the compression as a counter, so it wants neither
    the `0x80` byte `padded_region` writes nor a length field at all. Its traced
    counterpart is `pad.haifa_counter`'s to grow when a HAIFA family adopts this
    ABI — not built ahead of a consumer, the way the byte-sponge seam was paid
    for by waiting until a second byte sponge existed.

    **The width is the buffer's, not the message's.** `NB` covers every block
    `LMAX` could need, and the blocks past the live message are selected away by
    the caller's own loop rather than skipped: the block count is runtime data,
    which is exactly the cost a recognizing emitter exists to avoid paying. So
    this is written for correctness on the path where the marker is declined,
    and the family's `frx>=` floor is what keeps that path off a release.
    """
    if pad.trailer is Trailer.NONE:
        raise ValueError(
            "padded_message_region needs a rule with a trailer: HAIFA pads with "
            "a bare zero fill and carries its length as a counter into the "
            "compression (`pad.haifa_counter`), so neither the 0x80 byte nor "
            "the length field this region writes belongs to it"
        )
    lmax = buf.shape[-1]
    pos = fnp.arange(pad.nblocks(lmax) * pad.block_size, dtype=fnp.int32)
    # Bytes at or past `length` are padding, so the message read is clamped into
    # range rather than guarded: every lane it could spoil is selected away.
    content = buf[:, fnp.clip(pos, 0, lmax - 1)]
    active = pad.nblocks(length) * fnp.int32(pad.block_size)
    return padded_region(pad, content, length, active, trailer_field(pad, length))


@dataclass(frozen=True)
class MdStream:
    """The incremental half of Merkle-Damgard, shared by the SHA-2 families.

    #192 measured these two absorbs at zero differing code lines after rename
    normalization, and re-measuring on the current tree agrees: the eight lines
    that differ are comment reflow and one block size inside a comment. The
    gap-skip gather, the two-candidate compress-and-select and its "the
    discarded candidate is the only one that ever sees the junk tail" rationale
    were transcribed in full, twice.

    Which is the kind of duplication that is a correctness hazard rather than an
    annoyance: this is the subtlest traced-index machinery in the package, a
    consumer runs it (`Sha256FieldTranscript`), and a drift between the copies
    would be a wrong digest rather than a slow one.

    The pieces below are the family's, and the schedule is this class's.
    """

    # The family's padding rule, which already carries the block size and the
    # bytes the length field claims. Restating them here let the batch digest
    # and the streaming finalize disagree about where the length field goes.
    pad: PadRule
    block_to_words: Callable[[Array], Array]
    deserialize: Callable[[Array], Array]
    # `(h0, blocks, constants) -> serialized final state`, the family's own
    # marked chain, so the streaming path and the batch digest go through ONE
    # marker.
    #
    # The table is an ARGUMENT rather than the closure's capture so a caller
    # inside a marked region can thread that region's own operand — the rule
    # `sha256.compress` states. That is what makes the stream-finalize region's
    # declared `constants` operand and the table it actually folds the same
    # value by construction, rather than two independently-supplied ones with
    # nothing holding them equal.
    chain: Callable[[Array, Array, Array], Array]
    make_state: Callable[[Array, Array, Array], Any]
    # The family's constant table and its key in the plugin's primitive
    # registry. `chain` already closes over both; the STREAM FINALIZE marker
    # needs them again because its region WRAPS `chain` rather than being it,
    # and the table is a positional operand of that wider ABI
    # (`markers.STREAM_FINALIZE_MARKER`). Threaded rather than captured, for
    # the reason `chain` above states.
    constants: Array
    primitive: str
    # No `length_field`: `finalize` derives it from `pad` (`trailer_field`).

    def absorb(self, state: _Midstate, data: Array) -> Any:
        """Fold every newly-complete block of `data` (uint8 [L], L static) into
        the midstate and keep the remainder pending.

        The block loop is Python-unrolled and active-count-masked over STATIC
        slices — never a traced-index gather or a scan-carry scatter — which is
        the CPU-safe pattern `transcript.DuplexTranscript` uses.
        """
        block = self.pad.block_size
        length = data.shape[0]

        def decomposition(
            h: Array,
            constants: Array,
            pending: Array,
            counts: Array,
            data: Array,
            **_attrs: object,
        ) -> Any:
            pl = counts[0]
            combined_src = fnp.concatenate([pending, data.astype(fnp.uint8)])
            new_len = pl + fnp.int32(length)
            active_blocks = new_len // block
            max_blocks = (block - 1 + length) // block  # static upper bound

            # Drop the pending buffer's invalid gap [pending_len:block] from the
            # stream: for stream position j the source index is j while
            # j < pending_len, and shifted past the gap after that.
            total_slots = (max_blocks + 1) * block
            pos = fnp.arange(total_slots, dtype=fnp.int32)
            src_idx = pos + fnp.where(pos < pl, fnp.int32(0), block - pl)
            src_idx = fnp.clip(src_idx, 0, combined_src.shape[0] - 1)
            combined = combined_src[src_idx]  # valid prefix [0:new_len]

            # The live block count depends on pending_len by AT MOST one — (pl + L)
            # // block spans {L // block, (block - 1 + L) // block} — so both static
            # candidates are computed and selected between. The discarded one is the
            # only thing that ever sees the gap-shifted junk tail block.
            #
            # This select is what the marker exists to remove: which candidate is
            # live depends on `counts[0]`, which is runtime data, so the count of
            # emitted kernels tracks `L % block` rather than the hashing. An
            # emitter handed the position as an operand runs the one count the
            # position implies.
            #
            # The two candidates share a PREFIX, which is what keeps this from
            # costing two full passes: the low one is the chain over the first
            # `min_blocks`, and the high one continues from it over the single block
            # that separates them. That composition is just Merkle-Damgard's resume
            # property, the same one that lets a stream pick up from a midstate.
            # Running both from `h` instead costs min + max compressions — 2N - 1,
            # measured at 31 for a 1000-byte absorb where 16 suffice.
            #
            # `keccak.streaming.ShakeAbsorb.absorb` has done this since it was
            # written, snapshotting the carry mid-fold rather than folding twice.
            # The MD copies did not, and this one was re-derived rather than read
            # off it — the two absorbs are close enough that neither should be
            # changed without looking at the other.
            min_blocks = length // block
            h_new = h
            if max_blocks:
                words = self.block_to_words(
                    combined[: max_blocks * block].reshape(1, max_blocks * block)
                )
                if min_blocks:
                    h_new = self.deserialize(
                        self.chain(h_new, words[:, :min_blocks], constants)
                    )[0]
                if max_blocks > min_blocks:
                    h_hi = self.deserialize(
                        self.chain(h_new, words[:, min_blocks:], constants)
                    )[0]
                    h_new = fnp.where(active_blocks == max_blocks, h_hi, h_new)

            tail_len = new_len - active_blocks * block
            tail = frx.lax.dynamic_slice(combined, (active_blocks * block,), (block,))
            slot = fnp.arange(block, dtype=fnp.int32)
            new_pending = fnp.where(slot < tail_len, tail, fnp.uint8(0))
            # One fused counter update: [pending_len', total'] = [tail_len, total+L].
            new_counts = fnp.stack([tail_len, counts[1] + fnp.int32(length)])
            return self.make_state(h_new, new_pending, new_counts)

        # The whole absorb as ONE region, and the first marked region here whose
        # output is the STATE rather than a digest — three leaves, which is what
        # `markers.MarkerKind.STREAM` names.
        #
        # `constants` is threaded down rather than captured, for the reason
        # `md_chain` and the finalize region both state: a captured table is
        # lifted into an unnamed operand AHEAD of the declared ones and lands at
        # position 0, so the region would advertise one table and fold another
        # with nothing holding them equal.
        #
        # Emitted whether or not the pinned plugin knows the name — an
        # unrecognized marker only inlines, so being early costs the fusion and
        # nothing else. Nothing gates it, for the reason the finalize region
        # gives: this marker decides nothing about how anything ELSE lowers.
        return fused_region(
            decomposition,
            state.h,
            self.constants,
            state.pending,
            state.counts,
            data,
            name=STREAM_ABSORB_MARKER,
            version=STREAM_ABSORB_MARKER_VERSION,
            primitive=self.primitive,
        )

    def finalize(self, state: _Midstate, extras: Array) -> Array:
        """`H(absorbed ‖ extras[b])` for each row of `extras` (uint8 [B, E], E
        static), without mutating the state — the hash finished at the current
        length, once per row, from one shared midstate.

        The trailing content is `pending[:pending_len] ‖ extras[b]`, so with the
        0x80 byte and the length field it spans at most two blocks. Both shapes
        are emitted and the 1-vs-2-block choice selected, because it depends on
        `pending_len`, which is runtime data.

        That is also what bounds `E`: only a tail that fits the two-block layout
        at EVERY reachable `pending_len` is representable, so a wider one is
        rejected rather than silently overlapping the padding. Absorb the prefix
        and finalize with the remainder.
        """
        block, lb = self.pad.block_size, self.pad.reserve
        batch, e = extras.shape
        if e > block - lb:
            raise ValueError(
                f"extras width ({e}) must be <= {block - lb}: pending_len is "
                "runtime data, so a wider tail cannot be guaranteed to fit the "
                "two-block finalize layout at every offset — absorb the prefix "
                "instead"
            )

        def decomposition(
            h: Array,
            constants: Array,
            pending: Array,
            counts: Array,
            extras: Array,
            **_attrs: object,
        ) -> Array:
            pl = counts[0]
            content_len = pl + fnp.int32(e)
            len_bytes = trailer_field(self.pad, counts[1] + fnp.int32(e))

            # One block only if the content, the 0x80 and the length field all fit.
            two_blocks = content_len > fnp.int32(block - lb - 1)
            active_bytes = fnp.where(two_blocks, fnp.int32(2 * block), fnp.int32(block))

            pos = fnp.arange(2 * block, dtype=fnp.int32)
            # content = pending[:pl] ‖ extras[b], skipping the pending gap [pl:block].
            combined_src = fnp.concatenate(
                [fnp.broadcast_to(pending, (batch, block)), extras.astype(fnp.uint8)],
                axis=1,
            )
            src_idx = fnp.clip(
                pos + fnp.where(pos < pl, fnp.int32(0), block - pl), 0, block + e - 1
            )
            content = combined_src[:, src_idx]

            region = padded_region(
                self.pad, content, content_len, active_bytes, len_bytes
            )

            words = self.block_to_words(region)
            return fnp.where(
                two_blocks,
                self.chain(h, words, constants),
                self.chain(h, words[:, :1], constants),
            )

        # The whole hop as ONE region. Its ABI is the words-in digest's, one
        # schema over: the stream POSITION rides as the `counts` operand, so a
        # recognizing emitter runs the single block count the position implies
        # where this decomposition emits both candidates and selects.
        #
        # Emitted whether or not the pinned plugin knows the name -- an
        # unrecognized marker only inlines, so being early costs the fusion and
        # nothing else, which is how every `hash_frx.digest.*` here already
        # ships. Nothing gates it: unlike a permutation's marker, this one
        # decides nothing about how anything ELSE lowers.
        return fused_region(
            decomposition,
            state.h,
            self.constants,
            state.pending,
            state.counts,
            extras,
            name=STREAM_FINALIZE_MARKER,
            version=STREAM_FINALIZE_MARKER_VERSION,
            primitive=self.primitive,
        )

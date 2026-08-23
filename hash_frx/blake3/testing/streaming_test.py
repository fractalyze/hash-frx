# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Incremental BLAKE3 — byte-exact against the reference binding, and threadable.

`HostBlake3` is the oracle, as it is for the `ByteHash` differential sweep: it
wraps the BLAKE3 team's own Rust implementation, so agreement is agreement with
something outside this tree rather than two readings of one spec
(`docs/reference/conventions.md`).

Two properties matter and neither is an ordinary value check. The first is that
*where the message is split cannot matter* — the state carries a partial block
and a partial chunk, and a split that lands anywhere but a block boundary is
what exercises them. The second is that the state's shape does not depend on how
much has been absorbed, which is the whole reason it is a fixed-shape pytree
rather than a buffer; a transcript's round loop carries one, so that is tested
as a `fori_loop` carry rather than by proxy.

The split patterns matter more than the sizes. BLAKE3 is a tree hash over
1024-byte chunks with a 64-byte block inside each, so the interesting cuts land
exactly on a block edge, exactly on a chunk edge, and one byte either side of
both.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hash_frx.blake3.modes import BLOCK_LEN, CHUNK_LEN, DIGEST_LEN
from hash_frx.blake3.rows import BLAKE3_MARKER, HostBlake3
from hash_frx.blake3.streaming import (
    BLAKE3_COMPRESS_MARKER,
    Blake3Stream,
    blake3_stream_init,
)


def _u8(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8)


def _host(msg: bytes, out_len: int = DIGEST_LEN) -> bytes:
    return bytes(np.asarray(HostBlake3(out_len).digest(_u8(msg)[None, :]))[0])


def _device(pieces: list[bytes], out_len: int = DIGEST_LEN) -> bytes:
    """Absorb `pieces` in order into a fresh state, then read `out_len` bytes.

    Jitted as one program: absorbing eagerly compiles every `lax` primitive
    inside `absorb` separately, which for a long split is minutes of tracing for
    microseconds of hashing.
    """

    @frx.jit
    def run(*parts: frx.Array) -> frx.Array:
        state = blake3_stream_init()
        for part in parts:
            state = state.absorb(part)
        return state.finalize(out_len)

    arrays = [frx.device_put(_u8(p)) for p in pieces]
    return bytes(np.asarray(run(*arrays)))


def _splits(msg: bytes) -> list[list[bytes]]:
    """Piece sequences worth trying for a message of this length."""
    n = len(msg)
    out = [[msg]]  # one shot
    cuts = {1, BLOCK_LEN - 1, BLOCK_LEN, BLOCK_LEN + 1}
    cuts |= {CHUNK_LEN - 1, CHUNK_LEN, CHUNK_LEN + 1, n // 2}
    for cut in cuts:
        if 0 < cut < n:
            out.append([msg[:cut], msg[cut:]])
    # Many small pieces exercises the partial-block path repeatedly, but each
    # distinct piece length is its own trace, so this is the expensive arm —
    # keep it to messages short enough that it covers the partial-block cases
    # without paying for them at every size.
    if 3 < n <= 128:
        out.append([msg[i : i + 7] for i in range(0, n, 7)])
    return out


class Blake3StreamTest(absltest.TestCase):
    # One length per class the tree hash distinguishes: empty, sub-block, the
    # 64-byte block edge either side, the 1024-byte chunk edge either side, and
    # a two-dozen-chunk message — the only one that carries several subtree
    # chaining values at once, so the only one that exercises the stack merge.
    # Sizes between those classes cost a trace each and test nothing new.
    LENGTHS = (0, 1, 63, 64, 65, 1023, 1024, 1025, 24163)

    def test_every_split_matches_the_one_shot_hash(self) -> None:
        rng = np.random.default_rng(0)
        for n in self.LENGTHS:
            msg = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
            want = _host(msg)
            for pieces in _splits(msg):
                with self.subTest(n=n, pieces=[len(p) for p in pieces]):
                    self.assertEqual(_device(pieces).hex(), want.hex())

    def test_the_extendable_read_matches_at_every_length(self) -> None:
        # A consumer reads other than 32 bytes — a 16-byte field draw is the
        # common one — so the extendable output is part of the contract, not
        # `digest` with a slice on it.
        rng = np.random.default_rng(1)
        msg = rng.integers(0, 256, size=200, dtype=np.uint8).tobytes()
        for out_len in (1, 16, 31, 32, 33, 64, 96):
            with self.subTest(out_len=out_len):
                self.assertEqual(
                    _device([msg], out_len).hex(), _host(msg, out_len).hex()
                )

    def test_finalize_does_not_consume_the_state(self) -> None:
        # The root is a compression of the tree rather than a state the read
        # drains, so reading twice — and absorbing after a read — is defined.
        # A `ShakeSqueeze` cannot answer this and does not claim to.
        head, tail = b"the first half..", b"..and the second"
        state = blake3_stream_init().absorb(frx.device_put(_u8(head)))
        first = bytes(np.asarray(state.finalize(DIGEST_LEN)))
        self.assertEqual(first.hex(), _host(head).hex())
        self.assertEqual(
            bytes(np.asarray(state.finalize(DIGEST_LEN))).hex(), first.hex()
        )
        resumed = state.absorb(frx.device_put(_u8(tail)))
        self.assertEqual(
            bytes(np.asarray(resumed.finalize(DIGEST_LEN))).hex(),
            _host(head + tail).hex(),
        )


class PytreeThreadingTest(absltest.TestCase):
    """The reason this is a registered dataclass rather than a Python object."""

    def test_state_shape_is_absorb_invariant(self) -> None:
        # The point of the whole exercise: the state's pytree structure must not
        # depend on how much has been absorbed, or it cannot be a loop carry.
        fresh = blake3_stream_init()
        short = fresh.absorb(frx.device_put(np.zeros(3, np.uint8)))
        long = short.absorb(frx.device_put(np.zeros(5000, np.uint8)))
        base = frx.tree_util.tree_structure(fresh)
        self.assertEqual(frx.tree_util.tree_structure(short), base)
        self.assertEqual(frx.tree_util.tree_structure(long), base)
        for before, after in zip(
            frx.tree_util.tree_leaves(fresh), frx.tree_util.tree_leaves(long)
        ):
            self.assertEqual(before.shape, after.shape)
            self.assertEqual(before.dtype, after.dtype)

    def test_absorb_threads_a_fori_loop_carry(self) -> None:
        # What a transcript actually needs: survive `lax.fori_loop` as the carry.
        # If this compiles and matches, a round loop can stay in the program.
        piece = np.arange(100, dtype=np.uint8)

        @frx.jit
        def run(part: frx.Array) -> frx.Array:
            state = blake3_stream_init()
            state = frx.lax.fori_loop(0, 10, lambda _, s: s.absorb(part), state)
            return state.finalize(DIGEST_LEN)

        got = bytes(np.asarray(run(frx.device_put(piece))))
        self.assertEqual(got.hex(), _host(piece.tobytes() * 10).hex())

    def test_the_state_is_a_jit_boundary_argument(self) -> None:
        # A state built outside a zone and absorbed into inside it, which is the
        # hand-off a caller holding one across rounds makes.
        @frx.jit
        def step(state: Blake3Stream, part: frx.Array) -> Blake3Stream:
            return state.absorb(part)

        msg = b"handed across the boundary"
        state = step(blake3_stream_init(), frx.device_put(_u8(msg)))
        self.assertEqual(
            bytes(np.asarray(state.finalize(DIGEST_LEN))).hex(), _host(msg).hex()
        )

    def test_the_node_finishing_hops_are_marked(self) -> None:
        # Counted, not found: a resumable state cannot reach `hash_frx.digest.blake3`,
        # so every node it finishes has to carry its own region, and an
        # `assertIn` passes with one of them marked and the rest inline — which
        # is exactly the state this replaced. Three hops finish a node (the
        # absorb path's block, the subtree merge, finalize's stack fold); the
        # root read is a batch of output blocks and stays with `modes.py`.
        block = frx.device_put(_u8(b"x" * BLOCK_LEN))

        def absorb_then_finalize(part: frx.Array) -> frx.Array:
            return blake3_stream_init().absorb(part).finalize(DIGEST_LEN)

        text = frx.jit(absorb_then_finalize).lower(block).as_text()
        self.assertEqual(text.count(f'"{BLAKE3_COMPRESS_MARKER}"'), 3)
        # A whole-hash marker here would mean the resumable path fell back to
        # the one region that cannot express it.
        self.assertNotIn(f'"{BLAKE3_MARKER}"', text)


class AbsorbValidationTest(absltest.TestCase):
    """`absorb` rejects what it cannot hash rather than coercing it (#215).

    Coercing returned a well-formed digest of a *different* message: an int32
    payload narrows mod 256, a `[B, L]` one flattens into a single stream.
    """

    def test_rejects_a_payload_that_is_not_bytes(self) -> None:
        with self.assertRaisesRegex(TypeError, "uint8"):
            blake3_stream_init().absorb(fnp.asarray([1, 2], dtype=fnp.int32))

    def test_rejects_a_batched_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "1-D"):
            blake3_stream_init().absorb(fnp.zeros((2, 4), dtype=fnp.uint8))


if __name__ == "__main__":
    absltest.main()

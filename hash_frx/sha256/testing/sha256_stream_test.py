# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The fixed-shape streaming SHA-256 midstate (`Sha256State`) — byte-exact vs the
universal reference `hashlib.sha256`, named by no consumer.

`digest` pads a whole message once on host; this incremental core keeps the
Merkle–Damgård chaining value as a fixed-shape pytree so a byte Fiat-Shamir
transcript threads `@jit` / a `lax.scan` carry (`Sha256FieldTranscript`).
"""
from __future__ import annotations

import hashlib

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from hash_frx.markers import STREAM_FINALIZE_MARKER
from hash_frx.sha256 import sha256
from hash_frx.sha256.sha256 import (
    Sha256State,
    sha256_stream_absorb,
    sha256_stream_finalize,
    sha256_stream_init,
)
from hash_frx.testing.marker_recognized import emitted_composites


def _u8(data: bytes) -> fnp.ndarray:
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _stream_absorb_all(chunks: list[bytes]) -> Sha256State:
    state = sha256_stream_init()
    for c in chunks:
        state = sha256_stream_absorb(state, _u8(c))
    return state


class Sha256StreamTest(absltest.TestCase):
    def test_stream_matches_hashlib_across_lengths(self) -> None:
        # Lengths straddling the 55/56/64 finalization + block boundaries, each
        # finalized with no extra (SHA256 of the buffer) and with an 8-byte extra
        # (the transcript's counter-mode append).
        for n in (0, 1, 31, 32, 55, 56, 63, 64, 65, 100, 127, 128, 191, 200):
            msg = bytes((i * 7 + 3) & 0xFF for i in range(n))
            for extra in (b"", b"\x00\x11\x22\x33\x44\x55\x66\x77"):
                state = _stream_absorb_all([msg])
                got = bytes(
                    np.asarray(
                        sha256_stream_finalize(state, _u8(extra).reshape(1, -1))[0]
                    )
                )
                self.assertEqual(
                    got, hashlib.sha256(msg + extra).digest(), f"n={n} extra={extra!r}"
                )

    def test_stream_matches_hashlib_across_splits(self) -> None:
        # The same message absorbed in different chunk splits must hash identically
        # — pins the pending-block carry across absorb calls.
        msg = bytes((i * 13 + 1) & 0xFF for i in range(150))
        ref = hashlib.sha256(msg).digest()
        for split in ([150], [64, 86], [1, 63, 86], [50, 50, 50], [63, 1, 63, 23]):
            chunks, off = [], 0
            for s in split:
                chunks.append(msg[off : off + s])
                off += s
            state = _stream_absorb_all(chunks)
            got = bytes(
                np.asarray(sha256_stream_finalize(state, _u8(b"").reshape(1, 0))[0])
            )
            self.assertEqual(got, ref, f"split={split}")

    def test_stream_counter_mode_batch(self) -> None:
        # One finalize over a batch of 8-byte counters == per-counter hashlib. This
        # is exactly the transcript's `SHA256(buffer ‖ ctr_le8)` squeeze.
        msg = b"transcript-buffer-bytes"
        state = _stream_absorb_all([msg])
        counters = np.stack(
            [
                np.frombuffer(int(c).to_bytes(8, "little"), dtype=np.uint8)
                for c in range(5)
            ]
        )
        digs = np.asarray(sha256_stream_finalize(state, fnp.asarray(counters)))
        for c in range(5):
            ref = hashlib.sha256(msg + int(c).to_bytes(8, "little")).digest()
            self.assertEqual(bytes(digs[c]), ref, f"ctr={c}")

    def test_stream_threads_under_jit(self) -> None:
        # The whole point: absorb + finalize are pure FRX on a fixed-shape pytree,
        # so they run under @jit unchanged (a `lax.scan` carry is the same contract).
        msg = bytes(range(70))
        extra = b"\x01\x02\x03\x04\x05\x06\x07\x08"

        @frx.jit
        def run(data: fnp.ndarray, ex: fnp.ndarray) -> fnp.ndarray:
            state = sha256_stream_absorb(sha256_stream_init(), data)
            return sha256_stream_finalize(state, ex.reshape(1, -1))

        got = bytes(np.asarray(run(_u8(msg), _u8(extra)))[0])
        self.assertEqual(got, hashlib.sha256(msg + extra).digest())

    def test_stream_threads_through_scan(self) -> None:
        # The design claim `test_stream_threads_under_jit` alludes to: `Sha256State`'s
        # fixed shapes make it a valid `lax.scan` carry. Fold equal-size chunks
        # through a scan and check the finalized digest still matches hashlib.
        msg = bytes(range(96))  # 6 chunks of 16 -> one full block + a 32 B remainder
        chunks = fnp.asarray(np.frombuffer(msg, np.uint8)).reshape(6, 16)

        @frx.jit
        def run(xs: fnp.ndarray) -> fnp.ndarray:
            def step(
                state: Sha256State, chunk: fnp.ndarray
            ) -> tuple[Sha256State, None]:
                return sha256_stream_absorb(state, chunk), None

            state, _ = frx.lax.scan(step, sha256_stream_init(), xs)
            return sha256_stream_finalize(state, fnp.zeros((1, 0), dtype=fnp.uint8))

        got = bytes(np.asarray(run(chunks))[0])
        self.assertEqual(got, hashlib.sha256(msg).digest())


class FinalizeBoundsTest(parameterized.TestCase):
    """The finalize's two boundaries, both silent before (#212).

    `extras` wider than the block minus the length field cannot fit the
    two-block layout at every runtime `pending_len`, and used to overlap the
    padding and return a wrong digest with no error. The length field is a
    64-bit count in bits, and a byte count near 2^31 has a bit length near
    2^34 — held to `int.to_bytes` across the wrap the one-word form had.
    """

    @parameterized.parameters((63, 56), (0, 56), (55, 1), (63, 1), (32, 24))
    def test_widest_extras_still_match_hashlib(self, pending: int, width: int) -> None:
        state = sha256.sha256_stream_absorb(
            sha256.sha256_stream_init(), fnp.zeros(pending, dtype=fnp.uint8)
        )
        got = np.asarray(
            sha256.sha256_stream_finalize(state, fnp.zeros((1, width), dtype=fnp.uint8))
        )[0]
        want = hashlib.sha256(bytes(pending + width)).digest()
        self.assertEqual(got.tobytes(), want)

    def test_extras_past_the_layout_are_rejected(self) -> None:
        state = sha256.sha256_stream_init()
        with self.assertRaisesRegex(ValueError, "extras width"):
            sha256.sha256_stream_finalize(state, fnp.zeros((1, 57), dtype=fnp.uint8))


class FinalizeMarkerTest(parameterized.TestCase):
    """The whole hop is ONE marked region.

    `sha256_stream_finalize` emits both padding-block candidates and selects
    between them, because the live block count depends on `pending_len`, which
    is traced. The marker exists so a recognizing emitter runs the one block
    count the runtime position implies instead. Until the pinned plugin carries
    the recognizer the marker inlines, so these assert what this repo controls:
    that the region is on the wire, and that wrapping it changed no byte.
    """

    @parameterized.parameters(0, 3, 40, 63, 64, 200)
    def test_one_region_whatever_the_stream_position(self, prefix_len: int) -> None:
        state = sha256.sha256_stream_init()
        if prefix_len:
            state = frx.jit(sha256.sha256_stream_absorb)(
                state, fnp.zeros(prefix_len, dtype=fnp.uint8)
            )
        extras = fnp.zeros((4, 8), dtype=fnp.uint8)
        names = emitted_composites(
            lambda s, e: sha256.sha256_stream_finalize(s, e), state, extras
        )
        # Exactly one, at every position -- the count must not track the
        # candidate the schedule happens to take.
        self.assertEqual(names.count(STREAM_FINALIZE_MARKER), 1)

    @parameterized.parameters(1, 8, 40, 56)
    def test_byte_neutral_against_hashlib(self, width: int) -> None:
        """The marker is byte-neutral: an unrecognized name only inlines.

        Compared against a ONE-SHOT `hashlib` digest rather than a streaming
        one, because the property is that a resumed hash equals a one-shot hash
        of the same message -- an oracle that also resumed could agree for the
        same wrong reason.
        """
        prefix = bytes((i * 11 + 3) % 256 for i in range(100))
        state = frx.jit(sha256.sha256_stream_absorb)(
            sha256.sha256_stream_init(),
            fnp.asarray(np.frombuffer(prefix, dtype=np.uint8)),
        )
        rows = np.array(
            [[(r * 31 + i * 7 + 1) % 256 for i in range(width)] for r in range(4)],
            dtype=np.uint8,
        )
        got = np.asarray(
            frx.jit(sha256.sha256_stream_finalize)(state, fnp.asarray(rows))
        )
        for r in range(rows.shape[0]):
            self.assertEqual(
                bytes(got[r]), hashlib.sha256(prefix + bytes(rows[r])).digest()
            )


if __name__ == "__main__":
    absltest.main()

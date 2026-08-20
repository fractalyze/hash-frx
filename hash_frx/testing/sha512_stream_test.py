# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The fixed-shape streaming SHA-512 midstate (`Sha512State`) — byte-exact vs the
universal reference `hashlib.sha512`, named by no consumer.

`digest` pads a whole message once on host; this incremental core keeps the
Merkle–Damgård chaining value as a fixed-shape pytree so a byte Fiat-Shamir
transcript threads `@jit` / a `lax.scan` carry — `sha256_stream_test`'s sweep at
the 64-bit parameters (128-byte blocks, the 111/112 finalization cutoff, a
16-byte length field).
"""
from __future__ import annotations

import hashlib

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hash_frx.sha512 import (
    Sha512State,
    sha512_stream_absorb,
    sha512_stream_finalize,
    sha512_stream_init,
)


def _u8(data: bytes) -> fnp.ndarray:
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _stream_absorb_all(chunks: list[bytes]) -> Sha512State:
    state = sha512_stream_init()
    for c in chunks:
        state = sha512_stream_absorb(state, _u8(c))
    return state


class Sha512StreamTest(absltest.TestCase):
    def test_stream_matches_hashlib_across_lengths(self) -> None:
        # Lengths straddling the 111/112/128 finalization + block boundaries,
        # each finalized with no extra (SHA512 of the buffer) and with an
        # 8-byte extra (the transcript's counter-mode append).
        for n in (0, 1, 63, 64, 111, 112, 119, 120, 127, 128, 129, 200, 255, 256, 383):
            msg = bytes((i * 7 + 3) & 0xFF for i in range(n))
            # The absorb depends only on `n`, and finalize is a non-mutating
            # copy — one absorbed state serves both extras.
            state = _stream_absorb_all([msg])
            for extra in (b"", b"\x00\x11\x22\x33\x44\x55\x66\x77"):
                got = bytes(
                    np.asarray(
                        sha512_stream_finalize(state, _u8(extra).reshape(1, -1))[0]
                    )
                )
                self.assertEqual(
                    got, hashlib.sha512(msg + extra).digest(), f"n={n} extra={extra!r}"
                )

    def test_stream_matches_hashlib_across_splits(self) -> None:
        # The same message absorbed in different chunk splits must hash
        # identically — pins the pending-block carry across absorb calls.
        msg = bytes((i * 13 + 1) & 0xFF for i in range(300))
        ref = hashlib.sha512(msg).digest()
        for split in (
            [300],
            [128, 172],
            [1, 127, 172],
            [100, 100, 100],
            [127, 1, 127, 45],
        ):
            chunks, off = [], 0
            for s in split:
                chunks.append(msg[off : off + s])
                off += s
            state = _stream_absorb_all(chunks)
            got = bytes(
                np.asarray(sha512_stream_finalize(state, _u8(b"").reshape(1, 0))[0])
            )
            self.assertEqual(got, ref, f"split={split}")

    def test_stream_counter_mode_batch(self) -> None:
        # One finalize over a batch of 8-byte counters == per-counter hashlib.
        # This is exactly the transcript's `SHA512(buffer ‖ ctr_le8)` squeeze.
        msg = b"transcript-buffer-bytes"
        state = _stream_absorb_all([msg])
        counters = np.stack(
            [
                np.frombuffer(int(c).to_bytes(8, "little"), dtype=np.uint8)
                for c in range(5)
            ]
        )
        digs = np.asarray(sha512_stream_finalize(state, fnp.asarray(counters)))
        for c in range(5):
            ref = hashlib.sha512(msg + int(c).to_bytes(8, "little")).digest()
            self.assertEqual(bytes(digs[c]), ref, f"ctr={c}")

    def test_stream_threads_under_jit(self) -> None:
        # The whole point: absorb + finalize are pure FRX on a fixed-shape
        # pytree, so they run under @jit unchanged (a `lax.scan` carry is the
        # same contract).
        msg = bytes(range(140))
        extra = b"\x01\x02\x03\x04\x05\x06\x07\x08"

        @frx.jit
        def run(data: fnp.ndarray, ex: fnp.ndarray) -> fnp.ndarray:
            state = sha512_stream_absorb(sha512_stream_init(), data)
            return sha512_stream_finalize(state, ex.reshape(1, -1))

        got = bytes(np.asarray(run(_u8(msg), _u8(extra)))[0])
        self.assertEqual(got, hashlib.sha512(msg + extra).digest())

    def test_stream_threads_through_scan(self) -> None:
        # The design claim `test_stream_threads_under_jit` alludes to:
        # `Sha512State`'s fixed shapes make it a valid `lax.scan` carry. Fold
        # equal-size chunks through a scan and check the finalized digest still
        # matches hashlib.
        msg = bytes(range(192))  # 6 chunks of 32 -> one full block + 64 B left
        chunks = fnp.asarray(np.frombuffer(msg, np.uint8)).reshape(6, 32)

        @frx.jit
        def run(xs: fnp.ndarray) -> fnp.ndarray:
            def step(
                state: Sha512State, chunk: fnp.ndarray
            ) -> tuple[Sha512State, None]:
                return sha512_stream_absorb(state, chunk), None

            state, _ = frx.lax.scan(step, sha512_stream_init(), xs)
            return sha512_stream_finalize(state, fnp.zeros((1, 0), dtype=fnp.uint8))

        got = bytes(np.asarray(run(chunks))[0])
        self.assertEqual(got, hashlib.sha512(msg).digest())


if __name__ == "__main__":
    absltest.main()

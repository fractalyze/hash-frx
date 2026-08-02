# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Keccak-f[1600]'s fixed tables — FIPS 202 sections 3.2 and 3.4.

Keccak-f[1600] has no free parameters: the width, the round count, the rotation
offsets, and the round constants are all fixed by the standard, so these are
module constants rather than a params dataclass like `Poseidon2Params`.

Lanes are indexed flat as `x + 5*y` for the `(x, y)` of FIPS 202, matching the
byte order the sponge reads: lane `i` occupies bytes `8i .. 8i+7` of the state,
little-endian. Held as `uint32` halves (see `lane.py`), lane `i` is state
elements `2i` (low) and `2i + 1` (high), which keeps a rate of `r` lanes the
contiguous prefix `state[: 2r]` — a halves-first layout would scatter it.
"""

from __future__ import annotations

from hash_frx.keccak.lane import split

LANES = 25
ROUNDS = 24
# uint32 halves, so the state is twice the lane count.
WIDTH = 2 * LANES

# iota's round constants (FIPS 202 section 3.2.5), split on the host.
# The grids below are transcribed from FIPS 202 and their shape carries
# meaning — rho's offsets are the standard's 5x5 [y][x] table — so they are
# fenced from the formatter, which would flatten them one value per line.
# fmt: off
_RC64 = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
# fmt: on
ROUND_CONSTANTS: tuple[tuple[int, int], ...] = tuple(split(rc) for rc in _RC64)

# rho's rotation offsets (FIPS 202 section 3.2.2), flat by x + 5*y.
# fmt: off
ROTATION_OFFSETS: tuple[int, ...] = (
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
)
# fmt: on

assert len(ROUND_CONSTANTS) == ROUNDS
assert len(ROTATION_OFFSETS) == LANES

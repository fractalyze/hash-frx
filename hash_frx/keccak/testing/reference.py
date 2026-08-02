# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A plain-Python Keccak-f[1600], as the oracle the frx implementation is held to.

Deliberately transcribed straight from FIPS 202 over 64-bit Python integers, with
none of the lane-half representation or single-kernel authoring rules that shape
`permutation.py`. An oracle that shared those would fail the same way the thing
it checks does.

Python integers are arbitrary precision, so the 64-bit constants here are exact —
this is the same reason `lane.py` splits constants on the host rather than
materialising them as device values.

The oracle is itself anchored: `sponge` builds SHA3-256 and SHAKE128 on top of
this permutation and `reference_test` checks them against `hashlib`, which is a
separate implementation of the same standard. Without that, a shared
misreading of the spec would look like agreement.
"""

from __future__ import annotations

_MASK = (1 << 64) - 1

# The grids below are transcribed from FIPS 202 and their shape carries
# meaning — rho's offsets are the standard's 5x5 [y][x] table — so they are
# fenced from the formatter, which would flatten them one value per line.
# fmt: off
ROUND_CONSTANTS = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
# fmt: on

# fmt: off
ROTATION_OFFSETS = (
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
)
# fmt: on


def rotl64(value: int, n: int) -> int:
    """Rotate a 64-bit integer left by `n`."""
    n %= 64
    if n == 0:
        return value & _MASK
    return ((value << n) | (value >> (64 - n))) & _MASK


def keccak_f1600(state: list[int]) -> list[int]:
    """The permutation over 25 lanes, flat by `x + 5*y`."""
    if len(state) != 25:
        raise ValueError(f"need 25 lanes, got {len(state)}")
    a = list(state)
    for rnd in range(24):
        # theta
        c = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
        d = [c[(x - 1) % 5] ^ rotl64(c[(x + 1) % 5], 1) for x in range(5)]
        a = [a[x + 5 * y] ^ d[x] for y in range(5) for x in range(5)]
        # rho + pi
        b = [0] * 25
        for y in range(5):
            for x in range(5):
                b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64(
                    a[x + 5 * y], ROTATION_OFFSETS[x + 5 * y]
                )
        # chi
        a = [
            b[x + 5 * y] ^ ((~b[(x + 1) % 5 + 5 * y] & _MASK) & b[(x + 2) % 5 + 5 * y])
            for y in range(5)
            for x in range(5)
        ]
        # iota
        a[0] ^= ROUND_CONSTANTS[rnd]
    return a


def sponge(message: bytes, rate_bytes: int, suffix: int, out_bytes: int) -> bytes:
    """FIPS 202 sponge over `keccak_f1600` — only so the oracle can be anchored
    against `hashlib`. The library's own sponge is a separate construction; this
    is the shortest path to a digest a third party also computes."""
    a = [0] * 25
    padded = bytearray(message) + bytearray([suffix])
    while len(padded) % rate_bytes:
        padded.append(0)
    padded[-1] |= 0x80

    for off in range(0, len(padded), rate_bytes):
        block = padded[off : off + rate_bytes]
        for i in range(rate_bytes // 8):
            a[i] ^= int.from_bytes(block[8 * i : 8 * i + 8], "little")
        a = keccak_f1600(a)

    out = bytearray()
    while True:
        for i in range(rate_bytes // 8):
            out += a[i].to_bytes(8, "little")
            if len(out) >= out_bytes:
                return bytes(out[:out_bytes])
        a = keccak_f1600(a)


def to_state(lanes: list[int]) -> list[int]:
    """25 lanes -> the 50 `uint32` halves the permutation takes, interleaved."""
    out: list[int] = []
    for v in lanes:
        out.append(v & 0xFFFFFFFF)
        out.append((v >> 32) & 0xFFFFFFFF)
    return out


def from_state(halves: list[int]) -> list[int]:
    """The 50 `uint32` halves back to 25 lanes."""
    return [halves[2 * i] | (halves[2 * i + 1] << 32) for i in range(25)]

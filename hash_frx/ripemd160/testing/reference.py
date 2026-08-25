# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A plain-Python RIPEMD-160, as the oracle the frx one is held to.

Transcribed from the designers' specification — H. Dobbertin, A. Bosselaers,
B. Preneel, "RIPEMD-160: A Strengthened Version of RIPEMD" (FSE 1996), whose
definitions the designers publish as pseudocode at
https://homes.esat.kuleuven.be/~bosselae/ripemd/rmd160.txt (every constant and
table below is that document's); the hash is also registered in ISO/IEC
10118-3. Everything runs over plain Python ints and bytes, sharing none of the
array authoring rules that shape `ripemd160.py` — an oracle that shared those
would fail the same way the thing it checks does.

The oracle is itself anchored: `ripemd160_reference_test` holds `ripemd160` to
the designers' published vectors transcribed below (`VECTORS`, including the
million-"a" record), so agreement with the oracle means agreement with
RIPEMD-160 rather than with a second copy of the same misreading. The tables
were additionally cross-checked, entry for entry, against Bitcoin Core's
independent pure-Python implementation
(test/functional/test_framework/crypto/ripemd160.py), which carries the same
nine vectors.
"""

from __future__ import annotations

# Message-word selection r(j) / r'(j) and left-rotation amounts s(j) / s'(j),
# indexed by the flat 0..79 counter (five rounds of sixteen per line) — the
# four tables of the designers' pseudocode, verbatim.
R_LEFT: tuple[int, ...] = (
    # fmt: off
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13,
    # fmt: on
)
R_RIGHT: tuple[int, ...] = (
    # fmt: off
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11,
    # fmt: on
)
S_LEFT: tuple[int, ...] = (
    # fmt: off
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6,
    # fmt: on
)
S_RIGHT: tuple[int, ...] = (
    # fmt: off
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11,
    # fmt: on
)

# Added constants K(j) / K'(j), one per round: 0, then the integer parts of
# 2^30·sqrt(2, 3, 5, 7) on the left line; 2^30·cbrt(2, 3, 5, 7), then 0 on the
# right (the designers' pseudocode, "added constants").
K_LEFT: tuple[int, ...] = (0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E)
K_RIGHT: tuple[int, ...] = (0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000)

# Initial chaining value h0..h4 (the designers' pseudocode, "initial value" —
# MD4's four words extended by a fifth).
INITIAL_STATE: tuple[int, int, int, int, int] = (
    0x67452301,
    0xEFCDAB89,
    0x98BADCFE,
    0x10325476,
    0xC3D2E1F0,
)

_MASK = 0xFFFFFFFF


def _f(round_index: int, x: int, y: int, z: int) -> int:
    """The five nonlinear functions f1..f5 (the designers' pseudocode,
    "nonlinear functions at bit level: exor, mux, -, mux, -"), selected by the
    16-step round. The left line applies them in this order; the right line
    reversed (f(79-j) in the pseudocode), which callers spell `4 - round`."""
    if round_index == 0:
        return x ^ y ^ z
    if round_index == 1:
        return (x & y) | (~x & z)
    if round_index == 2:
        return (x | ~y) ^ z
    if round_index == 3:
        return (x & z) | (y & ~z)
    return x ^ (y | ~z)


def _rol(x: int, n: int) -> int:
    """Rotate the low 32 bits of `x` left by `n` — the pseudocode's rol_s."""
    x &= _MASK
    return ((x << n) | (x >> (32 - n))) & _MASK


def compress(
    state: tuple[int, int, int, int, int], block: bytes
) -> tuple[int, int, int, int, int]:
    """One 64-byte block into the 5-word chain — the pseudocode's round
    function: 80 iterations per line, two lines in parallel from the same
    chaining value, then the cross-line combination T := h1 + C + D'; ... that
    rotates the register roles by one."""
    x = [int.from_bytes(block[4 * i : 4 * (i + 1)], "little") for i in range(16)]
    al, bl, cl, dl, el = state
    ar, br, cr, dr, er = state
    for j in range(80):
        rnd = j // 16
        # T := rol_s(j)(A + f(j, B, C, D) + X[r(j)] + K(j)) + E;
        # A := E; E := D; D := rol_10(C); C := B; B := T.
        t = _rol(al + _f(rnd, bl, cl, dl) + x[R_LEFT[j]] + K_LEFT[rnd], S_LEFT[j]) + el
        al, bl, cl, dl, el = el, t & _MASK, bl, _rol(cl, 10), dl
        t = (
            _rol(
                ar + _f(4 - rnd, br, cr, dr) + x[R_RIGHT[j]] + K_RIGHT[rnd], S_RIGHT[j]
            )
            + er
        )
        ar, br, cr, dr, er = er, t & _MASK, br, _rol(cr, 10), dr
    h0, h1, h2, h3, h4 = state
    return (
        (h1 + cl + dr) & _MASK,
        (h2 + dl + er) & _MASK,
        (h3 + el + ar) & _MASK,
        (h4 + al + br) & _MASK,
        (h0 + bl + cr) & _MASK,
    )


def pad(msg: bytes) -> bytes:
    """MD-strengthening padding, "identical to that of MD4" (the designers'
    pseudocode): the 0x80 byte, zeros to 56 mod 64, then the 64-bit bit length
    **little-endian** — the opposite byte order of SHA-2's length field, and
    the documented trap the frx module restates."""
    nblocks = (len(msg) + 8) // 64 + 1  # room for the 0x80 byte + the length
    zeros = nblocks * 64 - len(msg) - 9
    return msg + b"\x80" + b"\x00" * zeros + (8 * len(msg)).to_bytes(8, "little")


def ripemd160(msg: bytes) -> bytes:
    """RIPEMD-160 of one message: the compression chain from the initial value
    over the padded blocks, the final chain serialized little-endian — 20
    bytes."""
    state = INITIAL_STATE
    padded = pad(msg)
    for i in range(0, len(padded), 64):
        state = compress(state, padded[i : i + 64])
    return b"".join(h.to_bytes(4, "little") for h in state)


# ---------------------------------------------------------------------------
# The designers' published test vectors, (message, digest-hex) — the table on
# https://homes.esat.kuleuven.be/~bosselae/ripemd160.html, complete. The same
# nine rows appear in Bitcoin Core's independent pure-Python implementation
# (test/functional/test_framework/crypto/ripemd160.py), which is the required
# second source. The million-"a" row is stated as a digest constant rather
# than a `VECTORS` entry so no fixture holds a million-byte literal;
# `ripemd160_reference_test` builds the message and checks it on the oracle
# (cheap there — the device digest would unroll ~15k compression blocks into
# one graph, so it never runs this row).
# ---------------------------------------------------------------------------
VECTORS: tuple[tuple[bytes, str], ...] = (
    (b"", "9c1185a5c5e9fc54612808977ee8f548b2258d31"),
    (b"a", "0bdc9d2d256b3ee9daae347be6f4dc835a467ffe"),
    (b"abc", "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"),
    (b"message digest", "5d0689ef49d2fae572b881b123a85ffa21595f36"),
    (
        b"abcdefghijklmnopqrstuvwxyz",
        "f71c27109c692c1b56bbdceb5b9d2865b3708dbc",
    ),
    (
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "12a053384a9c0c88e405a06c27dcf49ada62eb2b",
    ),
    (
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "b0e20b6e3116640286ed3a87a5713079b21f5189",
    ),
    (b"1234567890" * 8, "9b752e45573d4b39f4dbd3323cab82bf63326bfb"),
)

MILLION_A_DIGEST = "52783243c1697bdbe16d37f97f68f08325dc1528"

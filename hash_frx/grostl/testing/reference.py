# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A plain-Python Grøstl-256, as the oracle the frx one is held to.

Transcribed from the final-round Grøstl specification — "Grøstl — a SHA-3
candidate", document version 2.0.1, March 2, 2011 (http://www.groestl.info/
Groestl.pdf; the *tweaked* algorithm, which the pre-final "Grøstl-0" round
constants and shifts do not match. Section references below are to that
document.) Everything runs over plain Python ints, bytes and lists, sharing
neither the byte-plane arithmetic nor the no-gather authoring rules that shape
`grostl.py` — an oracle that shared those would fail the same way the thing it
checks does. In particular the S-box here is the 256-entry table the standard
defines it as, where the frx spelling is a bitsliced circuit; the two meeting
on all 256 entries is one of `grostl_test`'s cases.

The oracle is itself anchored: `reference_test` holds `grostl256` to the
final-round KAT vectors transcribed below (`KAT_VECTORS`), so agreement with
the oracle means agreement with Grøstl-256 rather than with a second copy of
the same misreading.
"""

from __future__ import annotations

from collections.abc import Iterable


def gf_mul(a: int, b: int) -> int:
    """Multiply in F_256 — the Rijndael field, reduction polynomial
    x^8 + x^4 + x^3 + x + 1 (0x11b), least significant bit = the x^0
    coefficient (spec section 3.4.5)."""
    acc = 0
    for _ in range(8):
        if b & 1:
            acc ^= a
        high = a & 0x80
        a = (a << 1) & 0xFF
        if high:
            a ^= 0x1B
        b >>= 1
    return acc


def _sbox_entry(x: int) -> int:
    """One AES S-box entry from its definition (FIPS 197 section 5.1.1): the
    multiplicative inverse in F_256 with 0 mapped to 0, then the affine
    transform b'_i = b_i + b_{i+4} + b_{i+5} + b_{i+6} + b_{i+7} + c_i over
    F_2 (indices mod 8) with c = 0x63. Grøstl uses this S-box unchanged (spec
    section 3.4.3 and Appendix B). Derived rather than transcribed, so no
    256-entry copy exists to mistype; `reference_test` anchors spot values to
    the published FIPS 197 table."""
    inverse = x
    if x != 0:
        # x^254 = x^-1 in F_256 (Fermat).
        inverse = 1
        for _ in range(254):
            inverse = gf_mul(inverse, x)
    out = 0
    for i in range(8):
        bit = (
            (inverse >> i)
            ^ (inverse >> ((i + 4) % 8))
            ^ (inverse >> ((i + 5) % 8))
            ^ (inverse >> ((i + 6) % 8))
            ^ (inverse >> ((i + 7) % 8))
            ^ (0x63 >> i)
        ) & 1
        out |= bit << i
    return out


AES_SBOX: tuple[int, ...] = tuple(_sbox_entry(x) for x in range(256))

ROUNDS = 10  # spec section 3.4.6: r = 10 for P_512 and Q_512
SHIFT_P: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)  # spec section 3.4.4
SHIFT_Q: tuple[int, ...] = (1, 3, 5, 7, 0, 2, 4, 6)  # spec section 3.4.4
# The first row of B = circ(02, 02, 03, 04, 05, 03, 05, 07); B[i][j] is
# MIX_ROW[(j - i) mod 8] (spec section 3.4.5: each row is the one above
# rotated right by one).
MIX_ROW: tuple[int, ...] = (0x02, 0x02, 0x03, 0x04, 0x05, 0x03, 0x05, 0x07)


def _to_matrix(block: bytes) -> list[list[int]]:
    """64 bytes -> 8x8 matrix A[row][col], column by column: byte k lands at
    row k mod 8, column k div 8 (spec section 3.4.1)."""
    return [[block[col * 8 + row] for col in range(8)] for row in range(8)]


def _from_matrix(a: list[list[int]]) -> bytes:
    return bytes(a[row][col] for col in range(8) for row in range(8))


def _permutation(a: list[list[int]], is_q: bool) -> list[list[int]]:
    """P_512 (is_q=False) or Q_512 (is_q=True): ROUNDS rounds of
    AddRoundConstant -> SubBytes -> ShiftBytes -> MixBytes (spec section 3.4).
    """
    for r in range(ROUNDS):
        # AddRoundConstant (spec section 3.4.2): P's C[i] is (10*j) XOR i
        # across row 0 and zero elsewhere; Q's is ff everywhere except row 7,
        # which is (ff - 10*j) XOR i.
        if is_q:
            a = [[byte ^ 0xFF for byte in row] for row in a[:7]] + [
                [a[7][j] ^ 0xFF ^ (j << 4) ^ r for j in range(8)]
            ]
        else:
            a = [[a[0][j] ^ (j << 4) ^ r for j in range(8)]] + a[1:]
        # SubBytes (spec section 3.4.3).
        a = [[AES_SBOX[byte] for byte in row] for row in a]
        # ShiftBytes (spec section 3.4.4): row i rotates left by sigma_i.
        shifts = SHIFT_Q if is_q else SHIFT_P
        a = [row[s:] + row[:s] for row, s in zip(a, shifts)]
        # MixBytes (spec section 3.4.5): A <- B x A, per column.
        a = [
            [
                _xor_fold(gf_mul(MIX_ROW[(j - i) % 8], a[j][col]) for j in range(8))
                for col in range(8)
            ]
            for i in range(8)
        ]
    return a


def _xor_fold(values: Iterable[int]) -> int:
    acc = 0
    for v in values:
        acc ^= v
    return acc


def _xor64(x: bytes, y: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(x, y))


def compress(h: bytes, m: bytes) -> bytes:
    """f(h, m) = P(h XOR m) XOR Q(m) XOR h (spec section 3.2, eq. 1)."""
    p = _from_matrix(_permutation(_to_matrix(_xor64(h, m)), is_q=False))
    q = _from_matrix(_permutation(_to_matrix(m), is_q=True))
    return _xor64(_xor64(p, q), h)


def pad(msg: bytes) -> bytes:
    """The spec section 3.6 padding at byte granularity: the '1' bit plus
    seven '0's is the byte 80, then '0' bytes, then the 64-bit big-endian
    count of 512-bit BLOCKS in the padded message — a block count, not the
    bit length ((N + w + 65) / l in the spec's terms)."""
    nblocks = (len(msg) + 8) // 64 + 1  # room for the 80 byte + the count
    zeros = nblocks * 64 - len(msg) - 9
    return msg + b"\x80" + b"\x00" * zeros + nblocks.to_bytes(8, "big")


def grostl256(msg: bytes) -> bytes:
    """Grøstl-256 of one message, per the final-round spec: iv_256 (spec
    section 3.5: the 512-bit representation of 256, i.e. 00..00 01 00), the
    compression chain over the padded blocks, then the output transformation
    Omega(h) = trunc_256(P(h) XOR h) — the trailing 32 bytes (section 3.3)."""
    h = bytes(62) + (256).to_bytes(2, "big")
    padded = pad(msg)
    for i in range(0, len(padded), 64):
        h = compress(h, padded[i : i + 64])
    p = _from_matrix(_permutation(_to_matrix(h), is_q=False))
    return _xor64(p, h)[-32:]


# ---------------------------------------------------------------------------
# Final-round Grøstl-256 KAT vectors, (message, digest-hex), spanning the
# padding boundaries: empty, one byte, 56 bytes (the one-vs-two-block cutoff —
# 56 + the 80 byte + the 8-byte count is 65 > 64), 63/64 (the block edge; 64
# gets a whole extra padding block), 65 (one past it) and 128 (a two-block
# message, three once padded).
#
# Provenance — three mutually independent sources agree on every row:
# - the designers' final-round NIST submission package,
#   http://www.groestl.info/Groestl.zip, KAT_MCT/ShortMsgKAT_256.txt (the
#   messages are the shared NIST SHA-3-competition KAT inputs, indexed there
#   by bit length: Len = 8 * the byte count below);
# - sphlib 3.0's independent C implementation (test_groestl.c,
#   `nist_vec256`, same indexing);
# - ironclad's independent Common Lisp implementation
#   (testing/test-vectors/groestl-256.testvec).
# ---------------------------------------------------------------------------
KAT_VECTORS: tuple[tuple[bytes, str], ...] = (
    (
        b"",
        "1a52d11d550039be16107f9c58db9ebcc417f16f736adb2502567119f0083467",
    ),
    (
        bytes.fromhex("cc"),
        "15e2671f0eaf66c0de3093ab7b1e39dc68f945d7002fc5dfd52d60527e7228d1",
    ),
    (
        bytes.fromhex(
            "eebcc18057252cbf3f9c070f1a73213356d5d4bc19ac2a411ec8cdeee7a571e2"
            "e20eaf61fd0c33a0ffeb297ddb77a97f0a415347db66bcaf"
        ),
        "1db8b9b50186a8a8a122d9a861ed45b9d01aaf8256bf2c8956b09fb166c92f6d",
    ),
    (
        bytes.fromhex(
            "f57c64006d9ea761892e145c99df1b24640883da79d9ed5262859dcda8c3c32e"
            "05b03d984f1ab4a230242ab6b78d368dc5aaa1e6d3498d53371e84b0c1d4ba"
        ),
        "bcf360ae1494dfd755471adfb62feb4a68415ee501620b278e99cd60955ff3f9",
    ),
    (
        bytes.fromhex(
            "e926ae8b0af6e53176dbffcc2a6b88c6bd765f939d3d178a9bde9ef3aa131c61"
            "e31c1e42cdfaf4b4dcde579a37e150efbef5555b4c1cb40439d835a724e2fae7"
        ),
        "5adebbfdf6fd6178892b39a97a32b29fb605f97e1e5c3bbcf624a0e9cd72d145",
    ),
    (
        bytes.fromhex(
            "16e8b3d8f988e9bb04de9c96f2627811c973ce4a5296b4772ca3eefeb80a652b"
            "df21f50df79f32db23f9f73d393b2d57d9a0297f7a2f2e79cfda39fa393df1ac"
            "00"
        ),
        "5f27dd5b62ba1301867becd5ae6347790e4c56d5526554903824b1717c3b3065",
    ),
    (
        bytes.fromhex(
            "2b6db7ced8665ebe9deb080295218426bdaa7c6da9add2088932cdffbaa1c141"
            "29bccdd70f369efb149285858d2b1d155d14de2fdb680a8b027284055182a0ca"
            "e275234cc9c92863c1b4ab66f304cf0621cd54565f5bff461d3b461bd40df281"
            "98e3732501b4860eadd503d26d6e69338f4e0456e9e9baf3d827ae685fb1d817"
        ),
        "b8a871928fcc39ab286e5a768b0ae61ddbd765fbc55c2dd2f3d10477d362a08f",
    ),
)

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A plain-Python Ascon-Hash256, as the oracle the frx one is held to.

Transcribed from NIST SP 800-232 (final, August 2025, "Ascon-Based Lightweight
Cryptography Standards for Constrained Devices",
https://doi.org/10.6028/NIST.SP.800-232; section references below are to that
document). Everything runs over plain Python ints and bytes, sharing neither
the (lo, hi) uint32 half-pair arithmetic nor the no-gather authoring rules
that shape `ascon.py` — an oracle that shared those would fail the same way
the thing it checks does. In particular the substitution layer here is the
column-wise 5-bit lookup of Table 6, where the frx spelling is the bitsliced
Figure 3 circuit across the five words; the two meeting on all 32 inputs is
one of `ascon_test`'s cases.

The word convention is the final SP's, NOT the CAESAR-era Ascon papers': a
64-bit state word loads from its eight bytes **little-endian** (§A.1, "the
first eight bytes are mapped to the first 64-bit unsigned integer word S0 in
little-endian notation"), and writes back the same way. The classic Ascon
submissions were big-endian, so a transcription from pre-NIST material fails
every vector below.

The oracle is itself anchored: `reference_test` holds `ascon_hash256` to the
KAT vectors transcribed below (`KAT_VECTORS`) and `INITIAL_STATE` — derived
here by permuting IV ‖ 0^256 rather than copied — to the precomputed values
the SP publishes (Table 12), so agreement with the oracle means agreement
with Ascon-Hash256 rather than with a second copy of the same misreading.
"""

from __future__ import annotations

_MASK64 = (1 << 64) - 1

RATE = 8  # bytes: §5.1, "the rate and capacity of Ascon-Hash256 are 64 and 256 bits"
ROUNDS = 12  # §5.1: initialization, absorbing and squeezing all use Ascon-p[12]

# IV = v ‖ 0^8 ‖ a ‖ b ‖ t ‖ r/8 ‖ 0^16 with v=2, a=b=12, t=256, r/8=8
# (Appendix B, eq. 79 and Table 13); the assembled value is Table 14's.
IV = 0x0000080100CC0002

# const_0..const_15 (Table 5); round i of Ascon-p[rnd] adds c_i =
# const_{16-rnd+i} to S2 (§3.2, eqs. 3-4), so Ascon-p[12] runs const_4..15.
CONSTANTS: tuple[int, ...] = (
    0x3C, 0x2D, 0x1E, 0x0F, 0xF0, 0xE1, 0xD2, 0xC3,
    0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B,
)  # fmt: skip

# The 5-bit substitution box as the Table 6 lookup, index x0‖x1‖x2‖x3‖x4 with
# x0 (the S0 bit plane) the most significant bit — the table's note: x = 1
# corresponds to (0, 0, 0, 0, 1).
SBOX: tuple[int, ...] = (
    0x04, 0x0B, 0x1F, 0x14, 0x1A, 0x15, 0x09, 0x02,
    0x1B, 0x05, 0x08, 0x12, 0x1D, 0x03, 0x06, 0x1C,
    0x1E, 0x13, 0x07, 0x0E, 0x00, 0x0D, 0x11, 0x18,
    0x10, 0x0C, 0x01, 0x19, 0x16, 0x0A, 0x0F, 0x17,
)  # fmt: skip

# The Σ_i rotation pairs (§3.4, eqs. 8-12): S_i ^= (S_i ⋙ r1) ^ (S_i ⋙ r2).
SIGMA_ROTATIONS: tuple[tuple[int, int], ...] = (
    (19, 28),
    (61, 39),
    (1, 6),
    (10, 17),
    (7, 41),
)


def rotr(x: int, n: int) -> int:
    """Rotate a 64-bit word right by n (the ⋙ of §2.3)."""
    return ((x >> n) | (x << (64 - n))) & _MASK64


def permutation(s: list[int], rounds: int = ROUNDS) -> list[int]:
    """Ascon-p[rounds] (§3): rounds of p_C -> p_S -> p_L on five 64-bit words."""
    for i in range(rounds):
        # Constant-addition layer p_C (§3.2).
        s = [s[0], s[1], s[2] ^ CONSTANTS[16 - rounds + i], s[3], s[4]]
        # Substitution layer p_S (§3.3): 64 parallel SBOX lookups, column j
        # being (s_(0,j), …, s_(4,j)) — spelled as the lookup, not the circuit.
        out = [0, 0, 0, 0, 0]
        for j in range(64):
            x = 0
            for w in range(5):
                x |= ((s[w] >> j) & 1) << (4 - w)
            y = SBOX[x]
            for w in range(5):
                out[w] |= ((y >> (4 - w)) & 1) << j
        # Linear diffusion layer p_L (§3.4).
        s = [w ^ rotr(w, r1) ^ rotr(w, r2) for w, (r1, r2) in zip(out, SIGMA_ROTATIONS)]
    return s


# The §5.1 initialization, S = Ascon-p[12](IV ‖ 0^256) (eq. 54) — derived by
# running the permutation rather than copied, so `reference_test`'s anchor to
# the SP's precomputed Table 12 values checks the whole permutation at once.
INITIAL_STATE: tuple[int, ...] = tuple(permutation([IV, 0, 0, 0, 0]))


def pad(length: int) -> bytes:
    """What pad(·, 64) (Algorithm 2) appends to a `length`-byte message: the
    bit 1 then zeros to the next 64-bit boundary — in the little-endian byte
    convention the byte 0x01 then zero bytes (§A.2: y = x ⊕ (1 ≪ 8n)), always
    at least one byte, so a rate-aligned message gains a whole padding block.
    """
    return b"\x01" + b"\x00" * (RATE - length % RATE - 1)


def ascon_hash256(msg: bytes) -> bytes:
    """Ascon-Hash256 of one message, per §5.1 / Algorithm 5: absorb the padded
    64-bit blocks into S0 with Ascon-p[12] between blocks, then squeeze the
    256-bit digest as four rate blocks with Ascon-p[12] between reads."""
    s = list(INITIAL_STATE)
    padded = msg + pad(len(msg))
    blocks = [padded[i : i + RATE] for i in range(0, len(padded), RATE)]
    # Absorbing (Alg. 5): every block XORs into S[0:63]; each but the last is
    # followed by the permutation. Blocks load little-endian (§A.2's
    # "64-bit block absorption").
    for block in blocks[:-1]:
        s[0] ^= int.from_bytes(block, "little")
        s = permutation(s)
    s[0] ^= int.from_bytes(blocks[-1], "little")
    # Squeezing (Alg. 5): H_i reads S[0:63] after a permutation, four times.
    digest = b""
    for _ in range(4):
        s = permutation(s)
        digest += s[0].to_bytes(8, "little")
    return digest


# ---------------------------------------------------------------------------
# Ascon-Hash256 KAT vectors, (message, digest-hex), spanning the rate
# boundaries: empty, one byte, 7/8/9 (the one-block edge — 8 aligned bytes
# gain a whole padding block), 15/16 (the two-block edge) and 64 (an
# eight-block message, nine once padded). Messages follow the LWC KAT
# convention the sources share: byte i of an n-byte message has value i.
#
# Provenance — three mutually independent sources agree on every row:
# - the designers' reference C implementation,
#   https://github.com/ascon/ascon-c,
#   crypto_hash/asconhash256/LWC_HASH_KAT_128_256.txt (Count = n + 1 carries
#   the n-byte message);
# - pyascon, the SP 800-232 Python implementation
#   (https://github.com/meichlseder/pyascon), cross-checked against that KAT
#   file over all 1025 rows while transcribing;
# - NIST's ACVP reference vectors
#   (https://github.com/usnistgov/ACVP-Server, gen-val/json-files/
#   Ascon-Hash256-SP800-232), whose byte-aligned cases the same
#   implementations reproduce.
# ---------------------------------------------------------------------------
KAT_VECTORS: tuple[tuple[bytes, str], ...] = (
    (
        bytes(range(0)),
        "0b3be5850f2f6b98caf29f8fdea89b64a1fa70aa249b8f839bd53baa304d92b2",
    ),
    (
        bytes(range(1)),
        "0728621035af3ed2bca03bf6fde900f9456f5330e4b5ee23e7f6a1e70291bc80",
    ),
    (
        bytes(range(7)),
        "3e4d273ba69b3b9c53216107e88b75cdbeedbcbf8faf0219c3928ab62b116577",
    ),
    (
        bytes(range(8)),
        "b88e497ae8e6fb641b87ef622eb8f2fca0ed95383f7ffebe167acf1099ba764f",
    ),
    (
        bytes(range(9)),
        "94269c30e0296e1ec86655041841823efa1927f520fd58c8e9bce6197878c1a6",
    ),
    (
        bytes(range(15)),
        "6421330df99c05eb715415ee17b455f2674f862ae3cc5badffe43a4a3ed273e1",
    ),
    (
        bytes(range(16)),
        "3158c1940a2fbadbd68ab661777859b94a689e4efc375911467addd641835c38",
    ),
    (
        bytes(range(64)),
        "a6f241bea5d16405812c06019d9f72d60132bd7c089c60549b2e56bb01c64f48",
    ),
)

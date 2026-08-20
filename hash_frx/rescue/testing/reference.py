# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A plain-Python Rescue (RPO) permutation, as the oracle the frx one is held to.

Written from the Rescue-Prime Optimized specification
(https://eprint.iacr.org/2022/1577, Section 2) and its reference
implementation, ASDiscreteMathematics/rpo (MIT)
reference_implementation/rescue_prime_optimized.sage — the permutation and
half-round order from `rescue_XLIX_permutation` (the paper's Section 2.4
ordering), the constants derivation from `get_round_constants` (Section 2.2),
the circulant MDS convention from `mds_matrix_vector_multiplication`
(Section 2.3), and the sponge schedule (padding, overwrite mode, indexation)
from `rpo_hash` (Sections 2.5-2.7). Everything runs over arbitrary-precision
Python ints, sharing neither the dtype arithmetic nor the single-kernel
authoring rules that shape `rescue.py` — an oracle that shared those would
fail the same way the thing it checks does.

The oracle is itself anchored: `reference_test` holds `rpo_hash` over the
shipped RPO-128 tables to `RPO128_TEST_VECTORS` — the 19 digests printed in
the paper's Section 3.1, produced by the designers' Sage implementation and
independently reproduced by miden-crypto's Rust `Rpo256` (0xMiden/crypto
src/hash/rescue/rpo/tests.rs::EXPECTED @ v0.9.0, the paper's state layout;
current main relabeled the layout in its PR #755, so its vectors differ by
construction). The SHAKE256 constant derivation lives only here: the shipped
parameter surface carries the EXPANDED table, and `get_round_constants` is
what re-derives it.
"""

from __future__ import annotations

import hashlib


def circulant(row: list[int]) -> list[list[int]]:
    """The circulant matrix over `row`: `M[i][j] = row[(j - i) mod n]` — each
    row is the previous one rotated right (the RPO paper's Section 2.3; Sage's
    `matrix.circulant`, which the reference implementation applies)."""
    n = len(row)
    return [[row[(j - i) % n] for j in range(n)] for i in range(n)]


def matrix_vector(matrix: list[list[int]], vector: list[int], p: int) -> list[int]:
    return [sum(m * v for m, v in zip(row, vector)) % p for row in matrix]


def get_round_constants(
    p: int, m: int, capacity: int, security_level: int, rounds: int
) -> list[int]:
    """The Section 2.2 derivation (reference `get_round_constants`): SHAKE256
    expands the ASCII seed "RPO(p,m,c,level)" into 2*m*rounds chunks of
    ceil(|p|/8) + 1 bytes; each chunk, read least-significant-byte first, is
    reduced mod p. Returns the flat 2mN list, one m-run per half-round."""
    bytes_per_int = (p.bit_length() + 7) // 8 + 1
    num_bytes = bytes_per_int * 2 * m * rounds
    seed_string = "RPO(%i,%i,%i,%i)" % (p, m, capacity, security_level)
    byte_string = hashlib.shake_256(seed_string.encode("ascii")).digest(num_bytes)
    round_constants = []
    for i in range(2 * m * rounds):
        chunk = byte_string[bytes_per_int * i : bytes_per_int * (i + 1)]
        round_constants.append(int.from_bytes(chunk, "little") % p)
    return round_constants


def rescue_permutation(
    state: list[int],
    mds: list[list[int]],
    round_constants: list[list[int]],
    alpha: int,
    inv_alpha: int,
    p: int,
) -> list[int]:
    """The RPO permutation (reference `rescue_XLIX_permutation`): each round
    is two half-rounds of MDS -> constants -> power map, the alpha map in the
    first and the inv_alpha map in the second (the paper's Section 2.4 order —
    NOT the SoK's S-box-first order, see `rescue.py`). `round_constants` is
    the (2*rounds, m) row layout the parameter surface carries; rows 2r and
    2r+1 serve round r."""
    s = list(state)
    for r in range(len(round_constants) // 2):
        s = matrix_vector(mds, s, p)
        s = [(a + c) % p for a, c in zip(s, round_constants[2 * r])]
        s = [pow(a, alpha, p) for a in s]
        s = matrix_vector(mds, s, p)
        s = [(a + c) % p for a, c in zip(s, round_constants[2 * r + 1])]
        s = [pow(a, inv_alpha, p) for a in s]
    return s


def rpo_hash(
    input_sequence: list[int],
    mds: list[list[int]],
    round_constants: list[list[int]],
    alpha: int,
    inv_alpha: int,
    p: int,
    m: int,
    capacity: int,
) -> list[int]:
    """The RPO hash schedule the published vectors pin (reference `rpo_hash`;
    the paper's Sections 2.5-2.7): capacity in lanes 0..c-1 and rate in
    c..m-1; a non-multiple-of-rate input appends 1 then zeros and sets the
    first capacity lane to 1; absorption OVERWRITES the rate lanes; the digest
    is the first rate//2 rate lanes after the last permutation. The sponge is
    spelled here rather than through `hash_frx.sponge` on purpose: the anchor
    must not share code with anything it anchors."""
    rate = m - capacity
    state = [0] * m
    seq = list(input_sequence)
    if len(seq) % rate != 0:
        seq.append(1)
        while len(seq) % rate != 0:
            seq.append(0)
        state[0] = 1
    while seq:
        state[capacity:] = seq[:rate]
        state = rescue_permutation(state, mds, round_constants, alpha, inv_alpha, p)
        seq = seq[rate:]
    return state[capacity : capacity + rate // 2]


# --- RPO-128 anchors ---
#
# The shipped instance's coordinates (https://eprint.iacr.org/2022/1577,
# Section 2.1, Table 1): p = 2^64 - 2^32 + 1, m = 12, c = 4, level 128, N = 7,
# alpha = 7 with alpha^-1 mod (p-1) printed in the same section.
RPO128_P = 2**64 - 2**32 + 1
RPO128_M = 12
RPO128_CAPACITY = 4
RPO128_SECURITY_LEVEL = 128
RPO128_ROUNDS = 7
RPO128_ALPHA = 7
RPO128_INV_ALPHA = 10540996611094048183

# The published MDS first row (Section 2.3, 128-bit instance).
RPO128_MDS_ROW = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]

# The published digests (Section 3.1): rpo_hash([0..i]) for i = 0..18, in the
# paper's state layout. Transcribed from miden-crypto's machine-readable copy
# (0xMiden/crypto src/hash/rescue/rpo/tests.rs::EXPECTED @ v0.9.0) and
# mechanically cross-checked numeral-for-numeral against the paper's
# Section 3.1 text — two independent publications of one table, so a
# transcription slip here cannot look like agreement.
# fmt: off
RPO128_TEST_VECTORS = (
    (1502364727743950833, 5880949717274681448, 162790463902224431,
     6901340476773664264),
    (7478710183745780580, 3308077307559720969, 3383561985796182409,
     17205078494700259815),
    (17439912364295172999, 17979156346142712171, 8280795511427637894,
     9349844417834368814),
    (5105868198472766874, 13090564195691924742, 1058904296915798891,
     18379501748825152268),
    (9133662113608941286, 12096627591905525991, 14963426595993304047,
     13290205840019973377),
    (3134262397541159485, 10106105871979362399, 138768814855329459,
     15044809212457404677),
    (162696376578462826, 4991300494838863586, 660346084748120605,
     13179389528641752698),
    (2242391899857912644, 12689382052053305418, 235236990017815546,
     5046143039268215739),
    (9585630502158073976, 1310051013427303477, 7491921222636097758,
     9417501558995216762),
    (1994394001720334744, 10866209900885216467, 13836092831163031683,
     10814636682252756697),
    (17486854790732826405, 17376549265955727562, 2371059831956435003,
     17585704935858006533),
    (11368277489137713825, 3906270146963049287, 10236262408213059745,
     78552867005814007),
    (17899847381280262181, 14717912805498651446, 10769146203951775298,
     2774289833490417856),
    (3794717687462954368, 4386865643074822822, 8854162840275334305,
     7129983987107225269),
    (7244773535611633983, 19359923075859320, 10898655967774994333,
     9319339563065736480),
    (4935426252518736883, 12584230452580950419, 8762518969632303998,
     18159875708229758073),
    (14871230873837295931, 11225255908868362971, 18100987641405432308,
     1559244340089644233),
    (8348203744950016968, 4041411241960726733, 17584743399305468057,
     16836952610803537051),
    (16139797453633030050, 1090233424040889412, 10770255347785669036,
     16982398877290254028),
)
# fmt: on

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Plain-Python Ascon-Hash256, Ascon-XOF128 and Ascon-CXOF128, as the oracles
the frx ones are held to.

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

The oracles are themselves anchored: `reference_test` holds each to the KAT
vectors below (provenance on each tuple) and each initial state — derived here
by permuting IV ‖ 0^256 rather than copied — to the published record, so
agreement with an oracle means agreement with the standard rather than with a
second copy of the same misreading. For Ascon-Hash256 that anchor is the
precomputed state the SP publishes in Table 12; for the two XOFs it is the
Table 13 IV assembly, which reproduces all three IVs from their version and
output-length fields.

The three share everything but two constants and one stage: one rate, one
padding rule, one permutation. Ascon-XOF128 is Ascon-Hash256 at a second IV
with the output length moved out of the IV and into the caller's request, and
Ascon-CXOF128 is that at a third IV with a length-encoded customization string
absorbed ahead of the message.
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


# ---------------------------------------------------------------------------
# Ascon-XOF128 (§5.2): the same rate, padding and absorb as Ascon-Hash256, at a
# different IV and with an arbitrary output length. The IV layout is Table 13's
# with version 3 and an output length of 0 — "arbitrary" is spelled as a zero
# length field — so the two differ in this constant and in nothing else.
# ---------------------------------------------------------------------------
XOF_IV = 0x0000080000CC0003

# Derived here rather than copied, exactly as `INITIAL_STATE` is, so the device
# module's transcription meets an independent derivation in `ascon_test`.
XOF_INITIAL_STATE: tuple[int, ...] = tuple(permutation([XOF_IV, 0, 0, 0, 0]))


def ascon_xof128(msg: bytes, output_size: int) -> bytes:
    """Ascon-XOF128 of `msg`, read out to `output_size` bytes (§5.2).

    Absorb and pad are `ascon_hash256`'s; only the initial state and the number
    of squeeze blocks differ, and the last block is truncated where the request
    is not a multiple of the 8-byte rate.
    """
    state = list(XOF_INITIAL_STATE)
    padded = msg + pad(len(msg))
    for i in range(0, len(padded), RATE):
        state[0] ^= int.from_bytes(padded[i : i + RATE], "little")
        state = permutation(state)
    out = b""
    while True:
        out += (state[0] & _MASK64).to_bytes(RATE, "little")
        if len(out) >= output_size:
            return out[:output_size]
        state = permutation(state)


# ---------------------------------------------------------------------------
# Ascon-XOF128 KAT, from the same published record as `KAT_VECTORS` above:
# the Ascon team's reference implementation
# (https://github.com/ascon/ascon-c), crypto_hash/asconxof128/
# LWC_XOF_KAT_128_512.txt, whose `Count = n + 1` row carries the n-byte
# message `bytes(range(n))` and 64 bytes of output. The rows below are the ones
# that straddle the 8-byte rate boundary, transcribed from that file; the
# oracle above was checked against ALL 1025 of its rows while transcribing.
# ---------------------------------------------------------------------------
XOF_KAT_VECTORS: tuple[tuple[bytes, str], ...] = (
    (
        bytes(range(0)),
        "473d5e6164f58b39dfd84aacdb8ae42ec2d91fed33388ee0d960d9b3993295c6"
        "ad77855a5d3b13fe6ad9e6098988373af7d0956d05a8f1665d2c67d1a3ad10ff",
    ),
    (
        bytes(range(1)),
        "51430e0438ecdf642b393630d977625f5f337656ba58ab1e960784ac32a16e0d"
        "446405551f5469384f8ea283cf12e64fa72c426bfebaea3aa1529e2c4ab23a2f",
    ),
    (
        bytes(range(7)),
        "7ae562db37212a9acd2673ecfd5b4f1c5cb2e6f64ebf00aa7f6ef8dc82c448d5"
        "fe11cd91f4368c37690d79e5de0ca8ad419e1918ce8dab2d42363e9476638a7b",
    ),
    (
        bytes(range(8)),
        "8d1886f5d3ec4af8d15b44bc62b74da6ea91bc28fb82f9c34079b5ed6e38b6c9"
        "51803d7dfb3c5e512a0ef5e4060062a6fd067f9c73ef9bee527411bda67fc896",
    ),
    (
        bytes(range(9)),
        "db3013bfbbd132dc1d3152fd955ed48f7cbb675e9ad2a2fecf92b74c957592e0"
        "c89959e81c16fd07ead9eeb8e40359c497aa20258b43d87ec69ad0bb0993fd38",
    ),
    (
        bytes(range(15)),
        "7517d9b0383dc7742e9e1335d97d3f1c5a971416ca4e72bf504e962f80286862"
        "733ad8f5e60adcc1c5b21e8be99d32bc80d70277b81e709dc56579c37bebc080",
    ),
    (
        bytes(range(16)),
        "10bfedc5f6442d3e1d8c324878ce1ddf73b01cafc365589283ac4cbb98e48de3"
        "ceda8a41bb0983d539e4d90f6458c5c781724fad641ed3cdb4779931097440b3",
    ),
    (
        bytes(range(32)),
        "2e5f3403f4171471cc7934b51982cece8d6628435db70e89880f3be4e0b7b052"
        "32dfe63c44a836d771337c9c5a2688d1b71ecabe0d5c2006fef36ef3186138ad",
    ),
    (
        bytes(range(64)),
        "0865c2fa92c71058e79e5c4214f3a1505540411586920536ccee85fbf2940b9f"
        "0131385ffe92f15f35bd35373f14d8bf11f078d9850096016f857d27575da423",
    ),
)


# ---------------------------------------------------------------------------
# Ascon-CXOF128 (§5.3): Ascon-XOF128 with a customization string absorbed ahead
# of the message. Same rate, same padding, same absorb and squeeze; a third IV,
# and one prefix stage.
# ---------------------------------------------------------------------------
# The IV layout is Table 13's with version 4 and, like the XOF's, an output
# length of 0. Derived here from the field assembly rather than transcribed:
# v ‖ 0^8 ‖ a ‖ b ‖ t ‖ r/8 ‖ 0^16 with v=4, a=b=12, t=0, r/8=8, which
# reproduces `IV` and `XOF_IV` exactly when given their own version and output
# length — `reference_test` pins all three against that one assembly, so a
# mistyped nibble here is a mistyped nibble in the two shipped constants too.
CXOF_IV = 0x0000080000CC0004

# Derived, not copied, on the same terms as `INITIAL_STATE` and
# `XOF_INITIAL_STATE`.
CXOF_INITIAL_STATE: tuple[int, ...] = tuple(permutation([CXOF_IV, 0, 0, 0, 0]))

# A customization string is at most 256 bytes (§5.3: |Z| <= 2048 bits), which
# is what makes its length field eight bytes and not a variable-width encoding.
MAX_CUSTOMIZATION_BYTES = 256


def customization_prefix(customization: bytes) -> bytes:
    """The blocks absorbed ahead of the message: `Z0 ‖ Z ‖ pad`.

    `Z0` is the customization's length **in BITS**, as eight little-endian
    bytes — a full rate block on its own — then the string, then the same
    Algorithm 2 padding a message gets.

    **This is not cSHAKE's `bytepad`/`encode_string`.** The two solve the same
    problem and encode it differently, and reading SP 800-232's rule off the
    SHA-3 family's is the error a self-consistent implementation hides: a fixed
    eight-byte little-endian BIT count here, against cSHAKE's left-encoded byte
    count inside a rate-padded envelope. `CXOF_KAT_VECTORS` is what tells the
    two apart.

    Rate-aligned by construction, since `Z0` is exactly one block and the pad
    closes `Z` to a boundary.
    """
    if len(customization) > MAX_CUSTOMIZATION_BYTES:
        raise ValueError(
            f"customization is {len(customization)} bytes; "
            f"§5.3 allows at most {MAX_CUSTOMIZATION_BYTES}"
        )
    z0 = (len(customization) * 8).to_bytes(8, "little")
    return z0 + customization + pad(len(customization))


def ascon_cxof128(msg: bytes, customization: bytes, output_size: int) -> bytes:
    """Ascon-CXOF128 of `msg` under `customization`, read out to `output_size`
    bytes (§5.3).

    **The prefix is not a separate schedule.** Both stages absorb rate blocks
    with a permutation after each, and both are rate-aligned, so the whole hash
    is `ascon_xof128`'s loop over `customization_prefix(Z) ‖ msg ‖ pad(msg)` at
    a third initial state. Spelling it as one stream is what the frx side does
    too, and `CXOF_KAT_VECTORS` holds the equivalence to the published record
    rather than to an argument.
    """
    stream = customization_prefix(customization) + msg + pad(len(msg))
    state = list(CXOF_INITIAL_STATE)
    for i in range(0, len(stream), RATE):
        state[0] ^= int.from_bytes(stream[i : i + RATE], "little")
        state = permutation(state)
    out = b""
    while len(out) < output_size:
        out += (state[0] & _MASK64).to_bytes(RATE, "little")
        state = permutation(state)
    return out[:output_size]


# ---------------------------------------------------------------------------
# Ascon-CXOF128 KAT, from the designers' reference implementation
# (https://github.com/ascon/ascon-c), crypto_cxof/asconcxof128/
# LWC_CXOF_KAT_128_512.txt, which carries 1089 rows: every (message,
# customization) length pair up to 32 bytes each, message byte i = i and
# customization byte i = 0x10 + i, with 64 bytes of output.
#
# `ascon_cxof128` above was checked against ALL 1089 rows while the ones below
# were extracted; these are the pairs that straddle the 8-byte rate on each
# axis independently and on both at once, which is where a prefix that mis-pads
# or a length field written big-endian first disagrees.
# ---------------------------------------------------------------------------
CXOF_KAT_VECTORS: tuple[tuple[bytes, bytes, str], ...] = (
    (
        bytes(range(0)),
        bytes(range(0x10, 0x10 + 0)),
        "4f50159ef70bb3dad8807e034eaebd44c4fa2cbbc8cf1f05511ab66cdcc52990"
        "5ca12083fc186ad899b270b1473dc5f7ec88d1052082dcdfe69fb75d269e7b74",
    ),
    (
        bytes(range(0)),
        bytes(range(0x10, 0x10 + 7)),
        "717a9ba3b3c00f4078572b2d3fb3f0a86d45f70bc4e1cd89cb7a952bfa641623"
        "83735534ecbe0a7e62e7592cf447404db0361d98c2237245688ead15c05ae59b",
    ),
    (
        bytes(range(0)),
        bytes(range(0x10, 0x10 + 8)),
        "61324766441dd6c11e1736bad1d2185820885ed76fe2ce537775a6e855eeafd2"
        "a6651b5e862a44982765f8b4c7cbe9c8b354f569ead6abc62cc9b7cdd72e0cb3",
    ),
    (
        bytes(range(0)),
        bytes(range(0x10, 0x10 + 9)),
        "32fde6b9d290f56fc74aac9368f32c69973e1bab35d96118db7181aae5776876"
        "73c01a9e35327aded556987eed3441d4f42ec36b0c198498d9e7f357b948d560",
    ),
    (
        bytes(range(1)),
        bytes(range(0x10, 0x10 + 1)),
        "63fa8ba86382f2d544580f51322d080424b42c556eb74503cd73cf052bb993bd"
        "6f5210984c71c9c445f43ccc5b158226e509bd339cd634414377f79411aa8d5c",
    ),
    (
        bytes(range(7)),
        bytes(range(0x10, 0x10 + 0)),
        "aa04b2e280d626f649ebc9e6e09bdcb1ed4b4669647fa727064ece4c913e2d62"
        "f380f31b1b4eb19ab61c77bb49f533df1e4fddba97cde8315411a85a178e940b",
    ),
    (
        bytes(range(8)),
        bytes(range(0x10, 0x10 + 0)),
        "2c076d8a559299e39d9c42d271b40cfd1072bebfac53c939b931508885887440"
        "36579fb25bf87a8a08924bc6194a6a6349dbf3d0046b03661e36466f46002532",
    ),
    (
        bytes(range(9)),
        bytes(range(0x10, 0x10 + 0)),
        "f4bde749129c676dc47b76060ac2eecb8e42b169c22783df441dd351ed944a80"
        "6f30ba8e3d5210927e332459692a40969708183b1e50adcd88c42d664476808e",
    ),
    (
        bytes(range(8)),
        bytes(range(0x10, 0x10 + 8)),
        "7c2fc5904cc9ac514902e50747e36f993dbde034cb05587af1432bf81c74b1ec"
        "87ecf6179701064494487476f607715853d74c5727925ebf4974e25eb8878919",
    ),
    (
        bytes(range(15)),
        bytes(range(0x10, 0x10 + 16)),
        "232f99539efc1c4c69cb55cd0699e42f913694314212c0111872515e9d931c23"
        "30ab8eee54b67d37d05693020c5e777844f4f0f7b214f1cee8a08107eab5026d",
    ),
    (
        bytes(range(16)),
        bytes(range(0x10, 0x10 + 15)),
        "1792046c4b783cf55d72f6e4a013de56db4c152cc7a004df1c7e65e56c4587b4"
        "726f376942b80842ab3b12754958d40ee3242d45aaf994d4ed281807e6a3db1f",
    ),
    (
        bytes(range(32)),
        bytes(range(0x10, 0x10 + 32)),
        "f22e68afdbc5ced421b3679e8606666b1e9f080ef42a70ef9bc32d764dddf27f"
        "cd879490428977c5ac7d7db6cb16c9e26a2cea1586ec0ae2aa5c8b0135cfc38c",
    ),
)

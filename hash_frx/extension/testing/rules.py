# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The seven shipped padding rules, as one table both pad suites walk.

`pad_test` and `pad_traced_test` are split so the first can stay frx-free —
that is the whole point of the split, and it is a real property, not a
preference. What the split does not require is two copies of the rules, and
until this file existed there were two: the traced suite hand-copied the four it
needed, so `trailer_field` was held against a transcription rather than against
the rule a family ships.

That is the drift class the traced field was extracted to end, one level up:
change `GROSTL`'s `reserve` in one file and the other keeps passing. Deps `:pad`
alone, so the frx-free suite can take it too.

These still mirror the families rather than being imported from them:
`extension/` sits below the family packages, so reaching into
`hash_frx.grostl.grostl._PAD` would invert the layering. What holds the two in
agreement is `pad_test`'s vector table, which was read off the seven
hand-written `_padding_tail` functions before they were deleted.
"""

from __future__ import annotations

from hash_frx.extension.pad import PadRule, SpongePad, Trailer

SHA256 = PadRule(64, Trailer.BIT_LENGTH)
SHA512 = PadRule(128, Trailer.BIT_LENGTH, reserve=16)
RIPEMD160 = PadRule(64, Trailer.BIT_LENGTH, big_endian=False)
SM3 = PadRule(64, Trailer.BIT_LENGTH)
GROSTL = PadRule(64, Trailer.BLOCK_COUNT)
BLAKE2S = PadRule(64, Trailer.NONE)
BLAKE2B = PadRule(128, Trailer.NONE)

ASCON_PAD = SpongePad(rate=8, head=0x01, final_bit=False)
SHA3_256_PAD = SpongePad(rate=136, head=0x06)

# Every rule with a trailer, which is every rule the runtime-length paths serve.
# Both axes no SHA-2 rule exercises are here: Grostl counts blocks where the
# rest count bits, and RIPEMD-160 writes the field little-endian.
TRAILER_RULES: tuple[tuple[str, PadRule], ...] = (
    ("sha256", SHA256),
    ("sha512", SHA512),
    ("grostl", GROSTL),
    ("ripemd160", RIPEMD160),
)

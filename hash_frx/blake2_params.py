# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""RFC 7693 §2.8's parameter block — the bytes both BLAKE2 families XOR into
their IV to make an initial state.

The block is where BLAKE2 puts everything that is a property of *which hash
this is* rather than of the message: the digest length, the key length, the
salt, the personalization, and the tree-mode geometry. Both families lay the
same fields out in the same order; they differ only in the widths, which is
what makes this one module rather than two copies with transposed offsets.

`digest_size` already travelled this path — `0x01010000 ^ digest_size` was the
whole parameter block when the families shipped unkeyed, being byte 0 (the
digest length) beside the sequential-mode fanout and depth of 1. This module
spells the other sixty-three bytes.

**The layout, from §2.8.** Offsets are into the block; a field not named is
zero in sequential mode:

| field | BLAKE2b | BLAKE2s |
|---|---|---|
| digest length | 0 (1 B) | 0 (1 B) |
| key length | 1 (1 B) | 1 (1 B) |
| fanout | 2 (1 B) | 2 (1 B) |
| depth | 3 (1 B) | 3 (1 B) |
| leaf length | 4 (4 B) | 4 (4 B) |
| node offset | 8 (8 B) | 8 (6 B) |
| node depth | 16 (1 B) | 14 (1 B) |
| inner length | 17 (1 B) | 15 (1 B) |
| reserved | 18 (14 B) | — |
| salt | 32 (16 B) | 16 (8 B) |
| personalization | 48 (16 B) | 24 (8 B) |

**Sequential mode is not a parameter here, it is the shape of the function.**
Fanout and depth are 1 and everything else in the tree group is 0 — for every
hash this package computes, because the tree chaining that would make them vary
is not implemented. So they are written as the constants they are rather than
lifted into arguments no caller could usefully move, which is the axis rule
`extension/pad.py` states: an axis changes an outcome, or it is a parameter
nobody needs. The offsets above are the whole record a tree mode would need,
and it can have them when there is one.

Everything is little-endian (§2.4), and everything is host `bytes` — this
module builds a constant *from* parameters and never touches a message, so it
pulls no frx and needs no device, the property `extension/pad.py` keeps for the
same reason.
"""

from __future__ import annotations

# The two families' word widths, in bytes. Every other width in the block is a
# function of this one, which is why it is the parameter the builders take:
# the block is eight words, the salt and the personalization two each.
BLAKE2B_WORD_BYTES = 8
BLAKE2S_WORD_BYTES = 4


def block_size(word_bytes: int) -> int:
    """The parameter block's length: eight words. 64 bytes for BLAKE2b, 32 for
    BLAKE2s — the same eight-word count as the state it is XORed into."""
    return 8 * word_bytes


def salt_size(word_bytes: int) -> int:
    """The salt's length, and the personalization's: two words each. 16 bytes
    for BLAKE2b, 8 for BLAKE2s (§2.8)."""
    return 2 * word_bytes


def max_field_size(word_bytes: int) -> int:
    """The longest digest and the longest key the standard admits — the same
    number, 64 bytes for BLAKE2b and 32 for BLAKE2s.

    §2.1 gives `nn` and `kk` the same range, `1..bb/2` and `0..bb/2`, over the
    *message* block `bb` — 128 bytes for BLAKE2b, 64 for BLAKE2s. Half of that
    is `8 * word_bytes`, which is also `block_size` above: the parameter block
    and the maximum digest are the same length because both are eight words.
    They are one expression here rather than three because they are one
    quantity in the standard, not because the equality happened to hold.
    """
    return 8 * word_bytes


def param_block(
    word_bytes: int,
    digest_size: int,
    key_size: int = 0,
    salt: bytes = b"",
    person: bytes = b"",
) -> bytes:
    """The §2.8 parameter block for a sequential-mode hash, little-endian.

    `salt` and `person` are zero-padded to `salt_size(word_bytes)` and rejected
    above it — the standard fixes the field width, so a longer value is a
    caller error rather than something to truncate. An empty value is the
    all-zero field, which is what an unsalted, unpersonalized hash has, so
    `param_block(8, 32)` reproduces the `0x01010000 ^ 32` that BLAKE2b's
    initial state carried before any of this existed. `blake2_params_test`
    pins that equality in both families, which is what holds this module to
    the constant it generalizes.
    """
    salt_len = salt_size(word_bytes)
    if not 1 <= digest_size <= max_field_size(word_bytes):
        raise ValueError(
            f"digest_size must be in 1..{max_field_size(word_bytes)} (RFC 7693), "
            f"got {digest_size}"
        )
    if not 0 <= key_size <= max_field_size(word_bytes):
        raise ValueError(
            f"key_size must be in 0..{max_field_size(word_bytes)} (RFC 7693), "
            f"got {key_size}"
        )
    if len(salt) > salt_len:
        raise ValueError(
            f"salt is {len(salt)} bytes; RFC 7693 §2.8 gives it {salt_len}"
        )
    if len(person) > salt_len:
        raise ValueError(
            f"person is {len(person)} bytes; RFC 7693 §2.8 gives it {salt_len}"
        )

    block = bytearray(block_size(word_bytes))
    block[0] = digest_size
    block[1] = key_size
    block[2] = 1  # fanout, sequential mode
    block[3] = 1  # depth, sequential mode
    # Bytes 4 .. salt_off-1 stay zero: leaf length, node offset, node depth,
    # inner length and (BLAKE2b only) the reserved run are all 0 in sequential
    # mode, in both families. That the two layouts differ inside this run —
    # BLAKE2b's node offset is 8 bytes and BLAKE2s's 6 — costs nothing here
    # precisely because every field in it is zero.
    salt_off = block_size(word_bytes) // 2
    block[salt_off : salt_off + len(salt)] = salt
    block[salt_off + salt_len : salt_off + salt_len + len(person)] = person
    return bytes(block)


def param_words(
    word_bytes: int,
    digest_size: int,
    key_size: int = 0,
    salt: bytes = b"",
    person: bytes = b"",
) -> tuple[int, ...]:
    """`param_block` read as the eight little-endian words a family XORs into
    its IV, one per state word (§2.5: h = IV ^ p)."""
    block = param_block(word_bytes, digest_size, key_size, salt, person)
    return tuple(
        int.from_bytes(block[i * word_bytes : (i + 1) * word_bytes], "little")
        for i in range(8)
    )

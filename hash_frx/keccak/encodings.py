# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SP 800-185 §2.3's encodings — the length prefixes every SHA-3 derived
function is built from.

cSHAKE, KMAC and TupleHash all work the same way: they put something that is
*not* the message in front of the message and rely on the reader being able to
tell where one ends and the other begins. These four functions are that
ability, and they are the entire correctness surface of the three constructions
above them — a wrong `left_encode` is three wrong hashes with no indication of
a common cause, which is why they live in their own module with their own
tests rather than inside the first construction that needed them.

| function | §     | shape                                    |
|---|---|---|
| `left_encode`   | 2.3.1 | `n ‖ x`, length prefix **first**          |
| `right_encode`  | 2.3.1 | `x ‖ n`, length prefix **last**           |
| `encode_string` | 2.3.2 | `left_encode(bit length of S) ‖ S`        |
| `bytepad`       | 2.3.3 | `left_encode(w) ‖ X ‖ 0*`, to a `w` multiple |

**The two prefixes exist for the two directions a parser can run.**
`left_encode` is read from the front — a reader takes one byte, learns how many
follow, and knows where `x` ends. `right_encode` is read from the back, which is
what a construction needs when it appends the length *after* a message whose end
it has just reached (KMAC's `right_encode(L)`, TupleHash's). They encode the same
integer the same way and differ only in which end the count sits at.

**`encode_string` counts BITS; `bytepad`'s `w` counts BYTES.** This is the one
place in the module where two adjacent functions read the same-looking integer
in different units, and getting them crossed is not a loud failure: the block
counts and the shapes come out identical, every structural test passes, and only
a published vector disagrees. `encode_string(b"KMAC")` is
`left_encode(32) ‖ "KMAC"` — thirty-two, not four — because §2.3.2 defines the
function over bit strings and `len(S)` there is a bit count. `bytepad(X, 168)`
pads to 168 *bytes*, because §2.3.3 step 3 divides by 8 before taking the
modulus. Both units are asserted in `testing/encodings_test.py` rather than left
to the reader.

The neighbouring family makes the same trap concrete from the other side:
Ascon-CXOF128 prefixes its customization string with a fixed eight-byte
little-endian *bit* count, where cSHAKE uses this module's `bytepad` /
`encode_string` envelope over a *byte*-oriented value. The two produce the same
block count and the same shapes, so a cross-wiring survives everything but a
vector.

**Why the limit is 2^2040 and not a round number.** §2.3.1 admits
`0 ≤ x < 2^2040`, which looks arbitrary until you notice that the count `n` is
itself encoded in exactly one byte: `n ≤ 255` bytes of `x`, and 8 × 255 = 2040
bits. The bound is not a policy choice about how long a string may be, it is the
largest integer the encoding can describe — so it is checked here rather than
assumed, and the two boundary values are pinned by test.

Host `bytes` throughout, and no frx import: these operate on lengths and
parameters, never on batch data, so the whole module resolves at trace time and
puts nothing on a backend. That is `blake2_params.py`'s property, kept here for
the same reason — a caller inside a `@jit` gets a constant.
"""

from __future__ import annotations

# §2.3.1's validity condition. The count `n` prepended by `left_encode` (and
# appended by `right_encode`) is a single byte, so `x` may occupy at most 255
# bytes: 8 * 255 = 2040 bits. Spelled as the shift it is rather than as a
# literal, so the derivation stays visible.
MAX_ENCODE_BITS = 2040
MAX_ENCODE_EXCLUSIVE = 1 << MAX_ENCODE_BITS


def _encode_length(x: int) -> int:
    """§2.3.1 step 1's `n`: the smallest **positive** integer with `2^(8n) > x`.

    Positive is what makes `left_encode(0)` two bytes rather than one — zero
    needs no base-256 digits, and the standard spends one on it anyway so that
    every encoding has a body to point at.
    """
    if x < 0 or x >= MAX_ENCODE_EXCLUSIVE:
        raise ValueError(
            f"x ({x}) must satisfy 0 <= x < 2**{MAX_ENCODE_BITS}: the byte "
            "count that prefixes the encoding is itself one byte (§2.3.1)"
        )
    return max(1, (x.bit_length() + 7) // 8)


def left_encode(x: int) -> bytes:
    """§2.3.1's `left_encode(x)` — `enc8(n) ‖ x` in base 256, big-endian.

    Parsed from the *beginning*: the leading byte is how many follow. Used
    wherever a length precedes the thing it measures — `encode_string`'s prefix,
    and `bytepad`'s leading `w`.

    `left_encode(0)` is `01 00`, the standard's own example.
    """
    n = _encode_length(x)
    return bytes([n]) + x.to_bytes(n, "big")


def right_encode(x: int) -> bytes:
    """§2.3.1's `right_encode(x)` — `x` in base 256, big-endian, `‖ enc8(n)`.

    Parsed from the *end*, which is what a construction appending a length after
    a message needs: KMAC's `right_encode(L)` and TupleHash's are the reason this
    exists beside `left_encode` rather than instead of it.

    `right_encode(0)` is `00 01`, the standard's own example.
    """
    n = _encode_length(x)
    return x.to_bytes(n, "big") + bytes([n])


def encode_string(s: bytes) -> bytes:
    """§2.3.2's `encode_string(S)` — `left_encode(len(S)) ‖ S`, where **`len(S)`
    is a bit count**.

    The standard defines this over bit strings; this package is byte-oriented
    throughout, so the length is `8 * len(s)` and the result is always
    byte-aligned (§2.3.2's closing note: a byte-oriented `S` gives a
    byte-oriented output).

    The bit count is the whole trap. `encode_string(b"KMAC")` is
    `01 20 4B 4D 41 43` — `left_encode(32)` — and a byte count would make it
    `01 04 ...`, which is the same length, the same shape, and a different hash.

    `encode_string(b"")` is `01 00`, the standard's own example.
    """
    return left_encode(8 * len(s)) + s


def bytepad(x: bytes, w: int) -> bytes:
    """§2.3.3's `bytepad(X, w)` — `left_encode(w) ‖ X`, zero-filled to a multiple
    of `w` **bytes**.

    `w` is a byte count, and it is the rate of the sponge the result is about to
    be absorbed into (cSHAKE's 168 or 136, KMAC's the same), so the padded
    prefix lands exactly on a block boundary and the message that follows starts
    a fresh block.

    Step 2 of the standard's definition — pad to a whole byte — is a no-op on a
    byte-oriented `X`, which is every input this package has. Only step 3, the
    fill to a `w` multiple, does anything here.

    An `X` that already fills a whole number of blocks after the prefix gets no
    extra block: the fill is `-len(z) % w`, which is zero exactly then.
    """
    if w <= 0:
        raise ValueError(f"w ({w}) must be positive (§2.3.3)")
    z = left_encode(w) + x
    return z + bytes(-len(z) % w)

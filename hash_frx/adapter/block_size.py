# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Input block sizes, for the constructions that are keyed by a block.

FIPS 198-1's `B` — the byte width of the underlying hash's input block. HMAC
needs it to build `K0`, and PBKDF2 needs it because it drives HMAC; nothing else
in the package does.

**This is a table and not a `ByteHash` attribute, deliberately.** The obvious
alternative is `ByteHash.block_size`, and it was declined: `hash_frx/adapter/hmac.py`
already argues why, and the argument is about what the seam means rather than
about convenience —

    the seam carries the intersection of byte hashes (`digest` alone), and a
    block size only means something to block-keyed constructions — BLAKE3, whose
    keyed mode is native, has no block size for HMAC to read.

So the width lives with the constructions that read it. `pbkdf2.py` makes the
same call for its midstate profiles ("a shape with one caller stays with the
caller"), and `duplex_sponge.py` for its `+`-merge, which stays on the
construction rather than on `Permutation`.

**A family with no entry is a statement, not a gap.** `BLAKE3` is absent because
HMAC-BLAKE3 is a construction nobody should reach for: BLAKE3 keys natively
(`Blake3Keyed`, spec section 2.3), and its 64-byte compression block is not an
HMAC `B` just because the number exists. A caller who genuinely wants it can
still pass 64 explicitly — `Hmac` takes the width as an argument, so the table
supplies a default rather than a permission.

The keys are row TYPES rather than instances, because the width is a property of
the family and not of an instance's output length: `Sha512` and `Sha512_256`
share a 128-byte block, and looking up by type is what makes that one entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

# FIPS 180-4 §5.1 (SHA-2), FIPS 202 §6 (the SHA-3 rates), RFC 7693 §2.1
# (BLAKE2), GB/T 32905-2016 §5.2 (SM3), RIPEMD-160's original submission.
#
# The SHA-3 entries are the sponge RATE, which is what a block-keyed
# construction over a sponge has to use as `B` — FIPS 198-1 is written for a
# Merkle-Damgard hash, and the rate is the width that plays its block's role.
# HMAC over SHA-3 is defined but rarely the right tool (KMAC exists, FIPS 202
# §5.3.1), which is worth knowing before reaching for one of these.
_BLOCK_SIZES: dict[str, int] = {
    "Sha256": 64,
    "Sha512": 128,
    "Sha384": 128,
    "Sha512_256": 128,
    "Sm3": 64,
    "Ripemd160": 64,
    "Blake2s": 64,
    "Blake2b": 128,
    "Grostl256": 64,
    "Sha3_256": 136,
    "Sha3_512": 72,
    "Keccak256": 136,
    "Shake128": 168,
    "Shake256": 136,
    # Deliberately absent, and each for its own reason:
    #
    # - the BLAKE3 rows — keyed natively (spec section 2.3), so HMAC is the
    #   wrong construction rather than an unsupported one (module docstring).
    # - the Ascon rows — an 8-byte rate against a 32-byte digest, so FIPS
    #   198-1 §4's "replace a longer-than-block key by its digest" has nowhere
    #   to put the result. `Hmac.__init__` rejects `block_size < digest_size`
    #   for exactly this reason, so an entry here would only move the error.
}


def block_size(byte_hash: ByteHash) -> int:
    """`byte_hash`'s input block in bytes, for a construction that needs one.

    Raises `LookupError` for a hash with no entry rather than guessing — not
    `KeyError`, whose `__str__` renders through `repr()` and would print this
    explanation quote-wrapped with its punctuation escaped. A wrong `B` does not
    error anywhere downstream — HMAC would produce well-formed bytes under the
    wrong key schedule, which is a silent interoperability failure rather than a
    crash — so declining to answer is the only safe default.

    The host and device rows of one family share an entry: they are the same
    hash, and `HostSha256` differs from `Sha256` in where it runs rather than in
    what it computes.
    """
    name = type(byte_hash).__name__
    width = _BLOCK_SIZES.get(name.removeprefix("Host"))
    if width is None:
        raise LookupError(
            f"no block size is registered for {name}. A block-keyed "
            "construction (HMAC, PBKDF2) needs FIPS 198-1's B, and guessing it "
            "produces well-formed bytes under the wrong key schedule rather "
            "than an error. If this hash genuinely has one, add it to "
            "`hash_frx/adapter/block_size.py`; if it is a BLAKE3 row, its keyed "
            "mode is native and HMAC is the wrong construction (see that "
            "module's docstring)."
        )
    return width

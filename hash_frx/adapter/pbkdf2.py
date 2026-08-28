# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""PBKDF2 (RFC 8018 §5.2 / NIST SP 800-132) — the iterated construction over
the seam's HMAC, `hkdf.py`'s sibling at the other RFC:
`U_1 = HMAC(P, S ‖ INT(i))`, `U_j = HMAC(P, U_{j-1})`,
`T_i = U_1 ⊕ … ⊕ U_c`. A single derivation is a serial chain — no
parallelism claim inside one row — but the BATCH axis is data-parallel
exactly like every digest here, and the concrete consumer is batched:
BIP-39 seed derivation is PBKDF2-HMAC-SHA512 with c = 2048, which
wallet-recovery / proof-of-reserves workloads prove many mnemonics at a
time.

**The iteration chain is a traced loop, not an unrolled one.** c is
2048..600k, so the `U_j` chain rides a `lax.fori_loop` carrying the XOR
accumulator — the loop sits OUTSIDE every marker, and each iteration's two
hash applications ride the underlying hash's existing machinery. No new wire
name exists for PBKDF2 (a dedicated marker would add ABI surface with no
recognizer story beyond what the digest markers already give); the lowered
module's marker count is a small constant, independent of c, which the tests
pin.

**Two compressions per iteration, not four, where the hash ships midstate
machinery.** Inside the chain every HMAC hashes `pad-block ‖ (digest-size
bytes)`: the first block of each side is `K0 ⊕ ipad` / `K0 ⊕ opad`, FIXED
across all c iterations, so the fast path precomputes the two midstates once
(`compress(INITIAL_STATE, K0 ⊕ pad)`) and each iteration resumes the
Merkle–Damgård chain from them — one compression per side. The per-hash
wiring lives in `_MIDSTATE_PROFILES`, populated for the full-length SHA-2
rows (`sha256` / `sha512`: `compress`, `block_to_words`, `serialize_digest`,
`INITIAL_STATE` are public exactly for callers building their own blocks);
the profiles use the UNMARKED chain because the batched `K0` makes the
midstates per-row, and the `*_merkle_damgard` markers take a batch-shared
`h0`. The table lives here rather than on the seam, for the reason
`hmac.py` keeps `block_size` off the `ByteHash` Protocol: midstate
resumption means something only to this construction's hot loop, and a
shape with one caller stays with the caller. The truncated variants
(`Sha224`, `Sha384`, `Sha512_224`, `Sha512_256`) export the same machinery
but carry no entry —
their digest is a SLICE of the serialized state, which this profile shape
does not model — and a hash without an entry runs the seam-generic HMAC
body in the loop instead: identical bytes, four compressions per iteration
(the pad blocks re-hashed each time), the §4 key processing still hoisted
outside the trip count.

**A strictly-sequential caller wants `hashlib.pbkdf2_hmac` instead**, which is
also this module's differential oracle: the c-chain traces (`fori_loop`), so it
buys nothing for a caller deriving one key from one password.

Deliberately NOT here (the #202 scope line): scrypt and Argon2. Their
memory-hardness — per-instance MB..GB tables with data-dependent indexing —
is purpose-built to defeat exactly the batched fused-kernel model this
package exists for; admitting them would need an explicitly-unfused
primitive class, a separate scope decision gated on a named consumer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from hash_frx.adapter.block_size import block_size
from hash_frx.adapter.hmac import Hmac
from hash_frx.sha256 import sha256
from hash_frx.sha512 import sha512

# The FIPS 198-1 §4 pad bytes, restated from `hmac.py`'s private constants:
# two spec literals a reviewer checks against the standard either way (the
# `blake2b._IV64` restatement argument).
_IPAD = 0x36
_OPAD = 0x5C


class _MidstateProfile(NamedTuple):
    """The midstate machinery one hash module exports, as the fast path
    consumes it. Every field is the module's own public surface; the
    signatures agree across the SHA-2 pair by construction (`sha512.py`
    mirrors `sha256.py` section for section)."""

    initial_state: Array  # the module's INITIAL_STATE device constant
    compress: Callable[[Array, Array], Array]  # (state [B,W], words) -> state
    block_to_words: Callable[[Array], Array]  # uint8 [B, n*block] -> words
    serialize_digest: Callable[[Array], Array]  # state [B, W] -> uint8 digest
    padding_tail: Callable[[int], np.ndarray]  # the module's own MD tail


# Keyed by the DEVICE row type: the fast path needs traced per-row states, so
# only device rows qualify, and only the modules that ship the machinery have
# entries. Everything else takes the generic seam path below. `padding_tail`
# is each module's own (module-private) tail builder rather than a
# restatement here: the fast path finishes a resumed block with the hash's
# padding at the TOTAL length — the length field counts the already-absorbed
# pad block, which is the whole trick — and re-deriving the rule at this
# altitude would read as generic over any MD hash when it is SHA-2's
# (RIPEMD-160's length field is little-endian).
_MIDSTATE_PROFILES: dict[type, _MidstateProfile] = {
    sha256.Sha256: _MidstateProfile(
        sha256.INITIAL_STATE,
        sha256.compress,
        sha256.block_to_words,
        sha256.serialize_digest,
        sha256._PAD.tail,
    ),
    sha512.Sha512: _MidstateProfile(
        sha512.INITIAL_STATE,
        sha512.compress,
        sha512.block_to_words,
        sha512.serialize_digest,
        sha512._PAD.tail,
    ),
}


def pbkdf2(
    mac: Hmac, password: ArrayLike, salt: ArrayLike, iterations: int, dk_len: int
) -> Array:
    """`DK = PBKDF2(P, S, c, dkLen)` — RFC 8018 §5.2, over `mac`'s HMAC.

    password : uint8 `[P]` (shared) or `[B, P]` (per row).
    salt : uint8 `[S]` (shared) or `[B, S]` (per row).
    iterations : the cost parameter c ≥ 1. Static — the chain is a traced
        loop of static trip count.
    dk_len : bytes of derived key, at most `(2^32 − 1) · hLen` (§5.2);
        `n = ceil(dk_len / hLen)` blocks unroll at trace time (n is small —
        BIP-39 is one SHA-512 block exactly).
    Returns uint8 `[B, dk_len]` — B the broadcast of the operands' leading
    axes (both unbatched -> a batch of one).

    Rows are independent derivations: batching B mnemonics is B separate
    PBKDF2 runs advancing in lockstep, byte-identical per row to
    `hashlib.pbkdf2_hmac`.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if not 1 <= dk_len <= (2**32 - 1) * mac.digest_size:
        raise ValueError(f"dk_len must be in [1, (2^32-1)*hLen], got {dk_len}")
    password = fnp.asarray(password, dtype=fnp.uint8)
    salt = fnp.asarray(salt, dtype=fnp.uint8)
    if password.ndim == 1:
        password = password[None, :]
    if salt.ndim == 1:
        salt = salt[None, :]
    # The batch is whatever the operands' leading axes broadcast to; a
    # mismatched pair fails here, at the boundary (the hkdf_expand rule).
    batch = int(np.broadcast_shapes(password.shape[:1], salt.shape[:1])[0])
    salt = fnp.broadcast_to(salt, (batch, salt.shape[1]))

    h_len = mac.digest_size
    # FIPS 198-1 §4's key processing, hoisted OUTSIDE the trip count for both
    # bodies — inside the loop it would run per iteration, ×c at runtime
    # (including a full H(K) digest for a longer-than-block password).
    k0 = fnp.broadcast_to(mac.block_key(password), (batch, mac.block_size))
    k_ipad = k0 ^ fnp.uint8(_IPAD)
    k_opad = k0 ^ fnp.uint8(_OPAD)

    profile = _MIDSTATE_PROFILES.get(type(mac.byte_hash))
    if profile is not None and block_size(mac.byte_hash) != mac.block_size:
        # A mis-parameterized Hmac (wrong block for its hash) must not
        # silently take the fast path at the right block size: the generic
        # path is the one that reproduces the (non-standard) bytes asked for.
        profile = None

    if profile is None:
        # Seam-generic body: the two digest calls ride the loop with the pad
        # blocks re-hashed every round — four compressions per iteration
        # instead of two, the price of staying behind the seam.
        def hmac_u(u: Array) -> Array:
            inner = fnp.asarray(
                mac.byte_hash.digest(fnp.concatenate([k_ipad, u], axis=1)),
                dtype=fnp.uint8,
            )
            return fnp.asarray(
                mac.byte_hash.digest(fnp.concatenate([k_opad, inner], axis=1)),
                dtype=fnp.uint8,
            )

    else:
        # Midstate fast path: the two pad blocks are fixed across the chain,
        # so absorb each ONCE into a per-row midstate and resume from it
        # every iteration — one compression per side. The in-chain message
        # is always exactly hLen bytes after its pad block, so the MD tail
        # is one host constant.
        state0 = fnp.broadcast_to(
            profile.initial_state, (batch, profile.initial_state.shape[0])
        )
        inner_mid = profile.compress(state0, profile.block_to_words(k_ipad))
        outer_mid = profile.compress(state0, profile.block_to_words(k_opad))
        tail = fnp.asarray(profile.padding_tail(mac.block_size + h_len))
        tail_rows = fnp.broadcast_to(tail, (batch, tail.shape[0]))

        def resumed(mid: Array, payload: Array) -> Array:
            words = profile.block_to_words(
                fnp.concatenate([payload, tail_rows], axis=1)
            )
            return profile.serialize_digest(profile.compress(mid, words))

        def hmac_u(u: Array) -> Array:
            return resumed(outer_mid, resumed(inner_mid, u))

    def body(_j: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        u, t = carry
        u = hmac_u(u)
        return u, t ^ u

    blocks = []
    n = -(-dk_len // h_len)
    for i in range(1, n + 1):  # §5.2's block index, static and small
        idx_rows = fnp.broadcast_to(
            fnp.asarray(np.frombuffer(i.to_bytes(4, "big"), dtype=np.uint8)),
            (batch, 4),
        )
        u1 = fnp.asarray(
            mac.mac(password, fnp.concatenate([salt, idx_rows], axis=1)),
            dtype=fnp.uint8,
        )
        _, t = frx.lax.fori_loop(1, iterations, body, (u1, u1))
        blocks.append(t)
    return fnp.concatenate(blocks, axis=1)[:, :dk_len]

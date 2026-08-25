# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`Dual` — a hash family's two implementations, picked per call.

Most families here ship twice: a device row that takes a tracer and returns an
`Array`, and a host row over a native library that reads the message bytes and
returns an `np.ndarray`. Which one a call wants is a property of **the values**,
not of the consumer that holds them — one scheme instance verifies a batch under
a tracer and signs one message concretely — so it cannot be fixed when a
consumer is built. `Dual` is that choice as a value: hand it the values, get
back the row type to build.

    SHAKE128 = Dual(Shake128)
    SHAKE128(seeds)(32).digest(seeds)   # Shake128 if `seeds` is on a device,
                                        # HostShake128 if it is a numpy array

**Which rows pair is a hash-frx fact**, which is the whole reason this ships.
sig-frx hand-rolls the same dispatch in `sig_frx/hashes.py` (measured at
`f344dc4`) for SHAKE-128, SHAKE-256 and SHA-256, and spends four paragraphs of
module docstring re-deriving the rule — that the pairing is real and that the
return type is the authority on traceability. A second scheme repo would derive
it a third time.

**It hands back the row type rather than being a hash itself.** A `Dual` that
implemented `digest` would have to publish a `digest_size` and a `fusion_path`
before it had seen a message, and `fusion_path` is the one it could not answer
honestly: [`byte_hash.py`](../byte_hash.py) holds it to agree with the return
type by construction, and a dispatching wrapper has no single return type until
the call. Handing back a real row keeps that invariant where it belongs — on a
row that answers for itself — and keeps the parameterization the caller's, which
matters because the two ends of a pair take the same constructor arguments but
each family takes its own (`Shake256(64)`, `Blake3Keyed(key, 16)`, `Sha256()`).

**Why the table names types rather than keying on the `Host` prefix.** Every
pair does spell the host row `Host` + the device row's name, so a name-keyed
table would work and would cost less to import. The argument is about where each
form fails: naming the types breaks at IMPORT the moment a row is renamed, while
a lookup keyed on a name answers the renamed row with "this family has no host
sibling" — a wrong answer at a consumer's call site rather than an error at the
table. `adapter/testing/dual_test.py` holds the table to the registry's device
and host buckets in both directions either way, so the difference is how early it
is caught, not whether.
[`block_size.py`](block_size.py) accepts the name-keyed form, and the later
failure with it, because staying dep-free is worth more to a table that answers
with an `int`; this one has to produce the class.

**Why this is in `adapter/` and not on the seam.** #231 left the choice open,
and the pull toward [`byte_hash.py`](../byte_hash.py) is real: it defines the
seam and both row bases, so the pairing is a statement about things it already
owns. It is here because the table has to name row **types** in order to hand
one back, and every row module imports `byte_hash` — the table cannot live in
the module they all import without a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar, cast

from frx import Array

from hash_frx.blake2b.blake2b import Blake2b
from hash_frx.blake2b.byte_hashes import HostBlake2b
from hash_frx.blake2s.blake2s import Blake2s, HostBlake2s
from hash_frx.blake3.rows import (
    Blake3,
    Blake3DeriveKey,
    Blake3Keyed,
    HostBlake3,
    HostBlake3DeriveKey,
    HostBlake3Keyed,
)
from hash_frx.keccak.byte_hashes import (
    HostSha3_256,
    HostSha3_512,
    HostShake128,
    HostShake256,
    Sha3_256,
    Sha3_512,
    Shake128,
    Shake256,
)
from hash_frx.sha256.sha256 import HostSha256, Sha256
from hash_frx.sha512.sha512 import (
    HostSha384,
    HostSha512,
    HostSha512_256,
    Sha384,
    Sha512,
    Sha512_256,
)
from hash_frx.sm3.sm3 import HostSm3, Sm3

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

# Device row type -> its host sibling. Every row that HAS one, and nothing else.
_HOST_SIBLINGS: dict[type, type] = {
    Sha256: HostSha256,
    Sha512: HostSha512,
    Sha384: HostSha384,
    Sha512_256: HostSha512_256,
    Sm3: HostSm3,
    Blake2s: HostBlake2s,
    Blake2b: HostBlake2b,
    Sha3_256: HostSha3_256,
    Sha3_512: HostSha3_512,
    Shake128: HostShake128,
    Shake256: HostShake256,
    Blake3: HostBlake3,
    Blake3Keyed: HostBlake3Keyed,
    Blake3DeriveKey: HostBlake3DeriveKey,
    # The families absent from this table ship no host row at all; `__init__`
    # below declines for them and carries the per-family reason. Each has a
    # pure-Python oracle under `testing/`, which is a differential partner for
    # the device row and not a row a consumer can reach.
}

# One family's row type — `Shake256`, `Blake3Keyed`, `Sha256`. A TypeVar and not
# a plain `type[ByteHash]` so a consumer's `Dual(Shake256)(vals)(64)` type-checks
# against the family's OWN constructor: collapsing to the seam loses the arity
# and the parameters with it, and `type[ByteHash](64)` is an error.
Family = TypeVar("Family", bound="type[ByteHash]")


def _traced(*values: object) -> bool:
    """Whether any of `values` is on a device, and so whether the device row is
    the one to build.

    A tracer and a committed device buffer both answer yes, and both should: the
    row that takes either without a round trip is the device row. The reverse
    needs no rule and gets none — a host row cannot be called on a tracer at all,
    because it reads the message bytes, and `ByteHash`'s return type is what says
    so.
    """
    return any(isinstance(value, Array) for value in values)


class Dual(Generic[Family]):
    """The device/host pair for one family, as a value that selects per call.

    Built from the device row type; the host sibling comes from the table above,
    because which rows pair is this package's fact rather than a caller's choice.

    device : the family's device row type — `Shake128`, `Sha256`, `Blake3Keyed`.

    Placement note: under `adapter/` and not on the seam because this table
    names row types, and every row module imports `byte_hash` (module docstring).
    """

    def __init__(self, device: Family) -> None:
        host = _HOST_SIBLINGS.get(device)
        if host is None:
            # `LookupError` rather than `KeyError`, whose `__str__` renders
            # through `repr()` and would print this quote-wrapped with its
            # punctuation escaped — `block_size` declines it for the same reason.
            raise LookupError(
                f"{device.__name__} has no host sibling to pair with. A family "
                "ships one only where a native library implements it: "
                "`hashlib` has no Grøstl, no Ascon and no pre-FIPS Keccak "
                "(its `sha3_*` carry "
                "the FIPS domain suffix), and OpenSSL 3 moved RIPEMD-160 to "
                "the legacy provider. There is nothing to dispatch to, so the "
                "device row is the only implementation and a consumer names it "
                "directly. If a host sibling has since shipped, pair it in "
                "`hash_frx/adapter/dual.py`."
            )
        self.device = device
        # Cast because the host row is not a subtype of the device row — the two
        # are siblings. It is typed as the device's own `Family` deliberately:
        # that is exact about the CONSTRUCTOR, which is all a caller uses this
        # for, and the identity it blurs is the one thing a `Dual` exists to
        # stop a caller from depending on.
        self.host = cast("Family", host)

    def __call__(self, *values: object) -> Family:
        """The row type to build for a call over `values`: the device row if any
        of them is on a device, the host row otherwise.

        `values` are what the call is *about* — the seed a stream is expanded
        from, the parts a digest is taken over — and need not be the message
        itself, which a caller often builds afterwards from exactly these. With
        none passed, nothing is on a device and the host row is the answer.
        """
        return self.device if _traced(*values) else self.host

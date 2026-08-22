# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A host Grøstl-256, as the differential partner for the device one.

**Testonly, and deliberately so — the `HostKeccak256` asymmetry.** The shipped
host rows exist because a C library implements them (`hashlib`, the `blake3`
wheel) at microseconds per message; Python's standard library has no Grøstl —
it lost the SHA-3 competition to Keccak, and `pycryptodome` does not carry it
either — so a host row here means the pure-Python oracle at three-plus orders
of magnitude off what a `Host*` name promises. A `testonly` target cannot be
depended on by shipped code, so a sequential Grøstl caller has no host path;
giving it one is a dependency decision, not a reason to promote this.

Built on `reference.grostl256`, which `reference_test` anchors to the
final-round KAT across every padding boundary — so this row is exactly as
trustworthy as that anchor, and `grostl_test`'s differential sweep holds the
device digest to it at lengths the KAT transcription does not carry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from frx.typing import ArrayLike

from hash_frx.byte_hash import HostRow, host_digest
from hash_frx.grostl.grostl import GROSTL256_DIGEST_SIZE
from hash_frx.grostl.testing.reference import grostl256

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from hash_frx.byte_hash import ByteHash


class HostGrostl256(HostRow):
    """`ByteHash` for host Grøstl-256 over the plain-Python oracle.

    The loop it runs under is [`byte_hash.host_digest`](../../byte_hash.py),
    shared with every host row in the package; the oracle asks for `bytes`,
    and one copy is nothing beside a pure-Python compression.
    """

    digest_size = GROSTL256_DIGEST_SIZE
    # The one legitimate class constant of the taxonomy: a host row is HOST on
    # every backend, so nothing here varies with the pin.

    def _hash_one(self, data: ReadableBuffer) -> bytes:
        return grostl256(bytes(data))

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(self._hash_one, self.digest_size, msg)


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_host_grostl256: type[ByteHash] = HostGrostl256

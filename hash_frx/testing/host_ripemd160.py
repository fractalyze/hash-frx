# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A host RIPEMD-160, as the differential partner for the device one.

**Testonly, and deliberately so — the `host_grostl256` arrangement, for a
different reason.** Grøstl has no C binding at all; RIPEMD-160 nominally has
one, but OpenSSL 3 moved it to the legacy provider, so `hashlib.new(
"ripemd160")` raises on most current builds — a `Host*` row that fails by
default on the platforms this package ships to is worse than none, and a
`pycryptodome` dependency is a decision this row does not get to make (issue
#189 records both). So the row loops the pure-Python oracle instead, at an
oracle's speed: the differential partner only, never a shipped fast path. A
`testonly` target cannot be depended on by shipped code, which is what keeps
that promise structural.

Built on `ripemd160_reference.ripemd160`, which `ripemd160_reference_test`
anchors to the designers' published vectors — so this row is exactly as
trustworthy as that anchor, and `ripemd160_test`'s differential sweep holds
the device digest to it at lengths the published vectors do not carry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from frx.typing import ArrayLike

from hash_frx.byte_hash import HostRow, host_digest
from hash_frx.testing.ripemd160_reference import ripemd160

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from hash_frx.byte_hash import ByteHash


class HostRipemd160(HostRow):
    """`ByteHash` for host RIPEMD-160 over the plain-Python oracle.

    The loop it runs under is [`byte_hash.host_digest`](../byte_hash.py),
    shared with every host row in the package; the oracle asks for `bytes`,
    and one copy is nothing beside a pure-Python compression.
    """

    # 20 bytes is the standard's own definition (the 160 in the name), so the
    # literal lives here rather than importing the device module's constant —
    # the oracle side of the differential carries no frx hash to import.
    digest_size = 20
    # The one legitimate class constant of the taxonomy: a host row is HOST on
    # every backend, so nothing here varies with the pin.

    def _hash_one(self, data: ReadableBuffer) -> bytes:
        return ripemd160(bytes(data))

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return host_digest(self._hash_one, self.digest_size, msg)


if TYPE_CHECKING:
    # Seam-conformance pin (docs/reference/conventions.md).
    _bh_host_ripemd160: type[ByteHash] = HostRipemd160

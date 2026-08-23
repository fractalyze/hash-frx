# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`Xof` — a variable-output hash family, before a length is chosen.

A variable-output hash ships here as a *family* and hashes as a *row*:
`Shake256` is the family, `Shake256(64)` the hash. The two are genuinely
different things — `Shake256(32)` and `Shake256(64)` are two hashes rather than
one hash asked for more bytes, which is the rule
[`keccak/byte_hashes.py`](../keccak/byte_hashes.py) states and
[`mgf1.py`](mgf1.py) repeats — so a consumer that has chosen SHAKE-256 but not
yet how many bytes it wants is holding the family. `Xof` is the type of that:
the constructor an output length is handed to.

It ships because a consumer that does not get it here writes it: sig-frx
declares this exact alias in `sig_frx/hashes.py` (measured at `f344dc4`),
because a scheme reads its output length from its own parameters and so takes
the family rather than a row. A second scheme repo would declare it again.

**What satisfies it.** Every variable-output row and its host sibling —
`Shake128`, `Shake256`, `AsconXof128`, `Blake2s`, `Blake2b`, `Blake3` — whose
constructor takes the output length alone. So does
`functools.partial(Mgf1, byte_hash)`, and that is not a coincidence: `Mgf1` is a
row over `(hash, length)` rather than a free function of a length *because* a
free function could not fill this slot, which [`mgf1.py`](mgf1.py) records.

**What does not.** A fixed-output row — `Sha256()` takes no length, so a
consumer holding one holds the hash already and has nothing to hand a length to.
Nor `Blake3Keyed`, whose constructor takes `(key, output_size)`: it reaches this
type through a `partial` the same way `Mgf1` does, the key being a parameter of
the hash rather than a length the caller chooses per call.

It is an alias rather than a `Protocol` because there is nothing to declare
beyond the call itself. The rows that satisfy it are plain classes that predate
the name, so a Protocol would be a base to inherit or a registration to
remember, and would check nothing a structural `Callable` does not already.

It sits under `adapter/` for dep-freeness rather than because it is an adapter:
it builds nothing over a finished hash, it is a statement about the seam. The
seam's own module imports `frx` at run time, so hosting the alias there would
charge a typing-only consumer the whole backend — which is the one thing this
module is shaped to avoid.

`ByteHash` is a forward reference and the import is `TYPE_CHECKING`-only, so
reading this type costs no backend — the same reason
[`block_size.py`](block_size.py) keeps its own table dep-free, and it matters
more here: a purely-typing consumer imports this and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from hash_frx.byte_hash import ByteHash

Xof: TypeAlias = Callable[[int], "ByteHash"]

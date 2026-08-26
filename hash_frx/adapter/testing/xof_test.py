# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What `Xof` claims about the rows, checked against the rows.

The alias is checked by mypy wherever a consumer annotates with it, and that
catches nothing here: the rows predate the name and satisfy it structurally, so
the interesting question is not whether the annotation compiles but whether the
package's claim about which constructors fill the slot is TRUE — every
variable-output family, both ends of each pair, and `Mgf1` through a `partial`.

That last one is why this file exists rather than being a line in the docstring.
`Mgf1` was made a row over `(hash, length)` instead of a free function of a
length precisely so it could fill this slot (`mgf1.py` records the decision), and
nothing else holds those two modules to each other.

The output length is the whole surface, so each case is `family(n).digest_size
== n`: a constructor that took the length and then ignored it would satisfy the
type and break every consumer of it.
"""

from __future__ import annotations

from functools import partial

from absl.testing import absltest, parameterized

from hash_frx.adapter.mgf1 import Mgf1
from hash_frx.adapter.xof import Xof
from hash_frx.ascon.ascon import AsconXof128
from hash_frx.blake2b.blake2b import Blake2b
from hash_frx.blake2s.blake2s import Blake2s
from hash_frx.blake3.rows import Blake3, Blake3Keyed
from hash_frx.keccak.byte_hashes import (
    Shake128,
    Shake256,
)
from hash_frx.sha256.sha256 import Sha256
from hash_frx.testing.rows import BYTE_HASH_ROWS

# `_LENGTH` is inside every family's range — BLAKE2s caps at 32 and BLAKE2b at
# 64 — so a failure is the type's claim being wrong rather than the length being
# out of bounds for one row.
_LENGTH = 24

# Every family whose constructor takes the output length ALONE, both ends of
# each pair: the pairs are what would catch a host row that shipped with a
# different constructor from its device sibling.
_FAMILIES: list[tuple[str, Xof]] = [
    ("Shake128", Shake128),
    ("Shake256", Shake256),
    ("AsconXof128", AsconXof128),
    ("Blake2s", Blake2s),
    ("Blake2b", Blake2b),
    ("Blake3", Blake3),
]


class VariableOutputFamiliesTest(parameterized.TestCase):
    @parameterized.named_parameters(*_FAMILIES)
    def test_the_family_is_an_xof(self, family: Xof) -> None:
        self.assertEqual(family(_LENGTH).digest_size, _LENGTH)

    def test_the_list_covers_every_row_that_answers_to_a_length(self) -> None:
        # The completeness half, without which "every variable-output family"
        # is a claim about whichever rows happened to ship when this was
        # written — the drift `testing/rows.py` exists to end. Membership is
        # decided by TRYING the row rather than by inspecting its signature,
        # because answering to a length is exactly what `Xof` asserts and a row
        # that took one and ignored it would pass an introspective check.
        named = {name for name, _ in _FAMILIES}
        missing = []
        for case in BYTE_HASH_ROWS:
            row_type = type(case.make())
            try:
                fits = row_type(_LENGTH).digest_size == _LENGTH
            except (TypeError, ValueError):
                continue
            if fits and row_type.__name__ not in named:
                missing.append(row_type.__name__)
        self.assertEqual(
            missing,
            [],
            f"these rows satisfy `Xof` but are not covered here: {missing}",
        )


class ReachedThroughAPartialTest(absltest.TestCase):
    """The constructions that fill the slot with their other parameters bound."""

    def test_a_partial_over_mgf1_is_an_xof(self) -> None:
        # The claim `mgf1.py` was designed around: a free `mgf1(h, seed, n)`
        # could not fill this slot, so the construction ships as a row and the
        # underlying hash binds ahead of the length.
        xof: Xof = partial(Mgf1, Sha256())
        self.assertEqual(xof(_LENGTH).digest_size, _LENGTH)

    def test_mgf1_stretches_a_fixed_output_hash_past_its_digest(self) -> None:
        # What the slot is worth: SHA-256 is not a variable-output family, and
        # this is how a consumer that needs one gets it out of a hash that is
        # not. 100 bytes is four SHA-256 blocks and not a multiple of 32.
        xof: Xof = partial(Mgf1, Sha256())
        self.assertEqual(xof(100).digest_size, 100)

    def test_a_keyed_family_is_an_xof_once_its_key_is_bound(self) -> None:
        # `Blake3Keyed(key, output_size)` takes the key first, the key being a
        # parameter of the hash rather than something a caller picks per call.
        xof: Xof = partial(Blake3Keyed, bytes(range(32)))
        self.assertEqual(xof(_LENGTH).digest_size, _LENGTH)


class FixedOutputRowsTest(absltest.TestCase):
    def test_a_fixed_output_family_takes_no_length(self) -> None:
        # The other half of the claim, and the reason `Dual` hands back a row
        # type rather than an `Xof`: a fixed-output family is built with no
        # arguments, so there is nothing to hand a length to.
        with self.assertRaises(TypeError):
            Sha256(_LENGTH)  # type: ignore[call-arg]
        self.assertEqual(Sha256().digest_size, 32)


if __name__ == "__main__":
    absltest.main()

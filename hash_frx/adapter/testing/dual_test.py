# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The device/host pairing table, and the selection over it.

Two things are worth checking here and neither is a digest. **That the table is
complete** — it is written out rather than derived from the `Host` prefix
(`dual.py` says why), so the way it stays honest is being walked against
`testing/rows.py`, the registry `row_conformance_test` already holds equal to
the package. A family that gains a host row without an entry, or an entry that
stops naming a shipped row, fails here rather than at a consumer's call site.

**And that the selection reads the values.** Both directions, and the three
kinds of value that answer them: a tracer, a committed device buffer, and a
numpy array. A `Dual` that always returned the device row would pass a
byte-exactness suite perfectly while costing a concrete caller a device dispatch
per short message, which is the thing it exists to avoid.

Whether a pair agrees byte for byte is deliberately NOT here:
`row_conformance_test.test_device_and_host_agree` already walks every sibling
pair for that, and a second copy would drift rather than reinforce.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

from hash_frx.adapter.dual import _HOST_SIBLINGS, Dual
from hash_frx.blake3.rows import Blake3Keyed, HostBlake3Keyed
from hash_frx.keccak.byte_hashes import HostShake128, Shake128
from hash_frx.sha256 import HostSha256, Sha256
from hash_frx.testing.rows import DEVICE_ROWS, HOST_ROWS

_MSG = np.arange(3 * 40, dtype=np.uint8).reshape(3, 40)

# Shipped rows by class NAME, which is what the `Host` prefix is a statement
# about. The device/host SPLIT is not restated by that prefix: `rows.py` derives
# it from `DeviceRow`/`HostRow` and its own comment declines to restate it, so
# reading its two tuples keeps this test checking the law rather than the
# labelling — and a row whose name broke the convention would land on the right
# side here and fail the pairing assertions instead of quietly leaving the
# universe.
_DEVICE_TYPES = {t.__name__: t for t in (type(c.make()) for c in DEVICE_ROWS)}
_HOST_TYPES = {t.__name__: t for t in (type(c.make()) for c in HOST_ROWS)}

# Device rows whose family ships no host sibling. Derived rather than listed, so
# a host row that ships later moves the family into `PairingTableTest` instead
# of leaving a stale expectation behind.
_UNPAIRED = [
    (name, row)
    for name, row in _DEVICE_TYPES.items()
    if f"Host{name}" not in _HOST_TYPES
]


class PairingTableTest(absltest.TestCase):
    """The table against the registry, in both directions."""

    def test_every_family_with_a_host_row_is_paired(self) -> None:
        # The direction that catches a row added to one side only: a host
        # sibling ships, so a consumer can reasonably ask for the pair, and an
        # unpaired device row would answer "this family has no host row" —
        # wrong, and wrong at their call site rather than here.
        missing = [
            name
            for name, row in _DEVICE_TYPES.items()
            if f"Host{name}" in _HOST_TYPES and row not in _HOST_SIBLINGS
        ]
        self.assertEqual(
            missing,
            [],
            "these families ship a host row but are not paired in "
            f"`hash_frx/adapter/dual.py`: {missing}",
        )

    def test_every_entry_names_two_shipped_rows_of_one_family(self) -> None:
        # The other direction: an entry naming a row that no longer ships, or
        # pairing two rows that are not the same family. A rename catches here
        # rather than becoming a `Dual` that hands back a hash nobody asked for.
        #
        # The key must be a shipped DEVICE row and the value a shipped HOST one,
        # read off the class rather than the prefix. That is also what pins the
        # pair against being spelled the wrong way round — which would select
        # backwards in both directions, while every digest it produced stayed
        # right bytes.
        for device, host in _HOST_SIBLINGS.items():
            with self.subTest(device=device.__name__):
                self.assertIs(_DEVICE_TYPES.get(device.__name__), device)
                self.assertIs(_HOST_TYPES.get(host.__name__), host)
                self.assertEqual(host.__name__, f"Host{device.__name__}")


class DeliberateAbsencesTest(parameterized.TestCase):
    """A family with no host row must decline, and say why."""

    @parameterized.named_parameters(*_UNPAIRED)
    def test_an_unpaired_family_declines_and_says_why(self, device: type) -> None:
        # Both properties in one method, as `block_size_test` does for the same
        # table shape. The file name because a caller hitting this needs to know
        # it is a registry gap rather than a bug in their own code; the reason
        # because the absences carry an argument — no `hashlib` Grøstl or Ascon,
        # no pre-FIPS Keccak, RIPEMD-160 behind OpenSSL 3's legacy provider —
        # which is pinned rather than left to a comment on the table.
        with self.assertRaises(LookupError) as caught:
            Dual(device)
        message = str(caught.exception)
        self.assertIn("adapter/dual.py", message)
        self.assertIn("native library", message)


class SelectionTest(absltest.TestCase):
    """Which row a call gets, and what its digest returns."""

    def test_a_tracer_selects_the_device_row(self) -> None:
        # The case the return type is the authority on: under a trace there is
        # no host row that could run at all, since it would have to read the
        # message bytes.
        selected: list[type] = []

        @frx.jit
        def run(msg: Array) -> Array:
            row = Dual(Shake128)(msg)
            selected.append(row)
            return row(32).digest(msg)

        out = run(fnp.asarray(_MSG))
        self.assertEqual(selected, [Shake128])
        self.assertIsInstance(out, Array)

    def test_a_committed_device_buffer_selects_the_device_row(self) -> None:
        # Not a tracer, but already on the device: the row that takes it without
        # a round trip is still the device one.
        seeds = fnp.asarray(_MSG)
        self.assertIs(Dual(Shake128)(seeds), Shake128)
        self.assertIsInstance(Dual(Shake128)(seeds)(32).digest(seeds), Array)

    def test_a_host_value_selects_the_host_row(self) -> None:
        self.assertIs(Dual(Shake128)(_MSG), HostShake128)
        out = Dual(Shake128)(_MSG)(32).digest(_MSG)
        self.assertIsInstance(out, np.ndarray)
        self.assertNotIsInstance(out, Array)

    def test_one_device_value_among_host_ones_selects_the_device_row(self) -> None:
        # `any`, not `all`: the call these values are about will be traced, and
        # a host row could not run inside it.
        self.assertIs(Dual(Shake128)(_MSG, fnp.asarray(_MSG)), Shake128)

    def test_no_values_selects_the_host_row(self) -> None:
        # Nothing is on a device, so nothing forces one.
        self.assertIs(Dual(Shake128)(), HostShake128)

    def test_a_fixed_output_family_selects_a_nullary_constructor(self) -> None:
        # The pair is over the family, not over an output length, so a
        # fixed-output family goes through unchanged and is built with no
        # arguments.
        self.assertIs(Dual(Sha256)(fnp.asarray(_MSG)), Sha256)
        self.assertIs(Dual(Sha256)(_MSG), HostSha256)
        self.assertIsInstance(Dual(Sha256)(_MSG)().digest(_MSG), np.ndarray)

    def test_a_keyed_family_keeps_its_own_constructor_arguments(self) -> None:
        # What handing back the TYPE buys: the parameterization stays the
        # caller's, and each family's arguments are its own.
        self.assertIs(Dual(Blake3Keyed)(_MSG), HostBlake3Keyed)
        row = Dual(Blake3Keyed)(_MSG)(bytes(range(32)), 16)
        self.assertEqual(row.digest_size, 16)

    def test_the_pair_is_readable_without_a_call(self) -> None:
        # Both ends are attributes, so a consumer that has already decided can
        # name one without inventing a value to dispatch on.
        dual = Dual(Shake128)
        self.assertIs(dual.device, Shake128)
        self.assertIs(dual.host, HostShake128)


if __name__ == "__main__":
    absltest.main()

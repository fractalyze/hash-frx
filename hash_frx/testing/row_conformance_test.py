# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every shipped row, held to the seam's contract at once.

`byte_hash.py` states the contract; thirty-two rows implement it. Each family
tested its own, so a rule could hold everywhere it was checked and still be
missing from whichever row shipped last — which is what happened twice, when the
seam sweeps behind #211 and #215 both had to be re-applied to SM3 and BLAKE2s
after the fact. This walks the list instead.

The equality cases carry the most weight and read like the least. A row's
`__eq__`/`__hash__` is its **jit cache key**: two instances that compare equal
share a trace, and two that do not each get their own. Get it wrong in the
lenient direction and one key's compiled executable is served for another's —
silently, with the right shape and the wrong constants. That is why `variant`
exists in the registry, and why "equal" and "distinguishable" are both asserted
rather than just the first.
"""

from __future__ import annotations

import importlib
import pathlib

import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

import hash_frx
from hash_frx.byte_hash import ByteHash, DeviceRow, HostRow, Row
from hash_frx.fusion import FusionPath
from hash_frx.testing.rows import ALL_ROWS, DEVICE_ROWS, HOST_ROWS, RowCase

_MSG = np.arange(3 * 40, dtype=np.uint8).reshape(3, 40)


def _named(cases: tuple[RowCase, ...]) -> list[tuple[str, RowCase]]:
    return [(c.name, c) for c in cases]


_BY_NAME = {c.name: c for c in HOST_ROWS}
# (device, host) for every device row that ships a host sibling.
_SIBLINGS = [
    (d.name, d, _BY_NAME[f"Host{d.name}"])
    for d in DEVICE_ROWS
    if f"Host{d.name}" in _BY_NAME
]


class RowEqualityTest(parameterized.TestCase):
    """The jit cache key, row by row."""

    @parameterized.named_parameters(*_named(ALL_ROWS))
    def test_same_parameters_compare_equal(self, case: RowCase) -> None:
        # Freshly built same-parameter instances must share a trace. Identity
        # equality here does not error — it just re-traces every call.
        self.assertEqual(case.make(), case.make())

    @parameterized.named_parameters(*_named(ALL_ROWS))
    def test_equal_rows_hash_equal(self, case: RowCase) -> None:
        # Required of `__eq__`/`__hash__` generally, and required here in
        # particular because the pair is used as a dict key.
        self.assertEqual(hash(case.make()), hash(case.make()))
        self.assertIn(case.make(), {case.make(): "cached"})

    @parameterized.named_parameters(
        *[(c.name, c) for c in ALL_ROWS if c.variant is not None]
    )
    def test_different_parameters_compare_unequal(self, case: RowCase) -> None:
        # The lenient-direction failure: a row that adds a parameter and forgets
        # to compare on it serves one parameterization's trace for another's.
        assert case.variant is not None
        self.assertNotEqual(case.make(), case.variant())

    @parameterized.named_parameters(*_named(ALL_ROWS))
    def test_a_foreign_type_is_not_equal(self, case: RowCase) -> None:
        # `type(other) is not type(self)` rather than `isinstance`: asymmetric
        # under subclassing, and it blocks Python's reflected fallback.
        self.assertNotEqual(case.make(), object())
        self.assertNotEqual(case.make(), 42)

    def test_distinct_row_types_are_never_equal(self) -> None:
        # Two parameterless rows with the same digest_size (Sha256, Sha512_256,
        # Sm3, Keccak256 are all 32 bytes) must still key separately.
        built = [c.make() for c in ALL_ROWS]
        for i, a in enumerate(built):
            for b in built[i + 1 :]:
                if type(a) is not type(b):
                    self.assertNotEqual(a, b)


class RowSeamTest(parameterized.TestCase):
    """The parts of the `ByteHash` contract that hold for every row."""

    @parameterized.named_parameters(*_named(ALL_ROWS))
    def test_satisfies_the_protocol(self, case: RowCase) -> None:
        self.assertIsInstance(case.make(), ByteHash)

    @parameterized.named_parameters(*_named(ALL_ROWS))
    def test_digest_size_is_positive_and_matches_output(self, case: RowCase) -> None:
        row = case.make()
        self.assertGreater(row.digest_size, 0)
        self.assertEqual(np.asarray(row.digest(_MSG)).shape, (3, row.digest_size))

    @parameterized.named_parameters(*_named(ALL_ROWS))
    def test_rank_is_rejected_at_the_seam(self, case: RowCase) -> None:
        # A 1-D message is the common miss: one message is B=1, not a bare [L].
        # It must fail at the seam, not from inside a marked region's trace.
        with self.assertRaises((ValueError, TypeError)):
            case.make().digest(np.zeros(40, dtype=np.uint8))

    @parameterized.named_parameters(*_named(ALL_ROWS))
    def test_zero_row_batch_digests_to_zero_rows(self, case: RowCase) -> None:
        # B=0 is what #211 fixed everywhere; this is what keeps it fixed for the
        # row that ships next.
        row = case.make()
        out = np.asarray(row.digest(np.zeros((0, 40), dtype=np.uint8)))
        self.assertEqual(out.shape, (0, row.digest_size))

    @parameterized.named_parameters(*_named(DEVICE_ROWS))
    def test_device_rows_return_arrays_and_are_traceable(self, case: RowCase) -> None:
        # The return type is the authority for traceability, and `fusion_path`
        # is held to agree with it.
        row = case.make()
        self.assertIsInstance(row.digest(_MSG), Array)
        self.assertTrue(row.fusion_path.is_traceable)
        self.assertIsNot(row.fusion_path, FusionPath.HOST)

    @parameterized.named_parameters(*_named(HOST_ROWS))
    def test_host_rows_return_ndarrays_and_are_not_traceable(
        self, case: RowCase
    ) -> None:
        row = case.make()
        out = row.digest(_MSG)
        self.assertIsInstance(out, np.ndarray)
        self.assertNotIsInstance(out, Array)
        self.assertIs(row.fusion_path, FusionPath.HOST)
        self.assertFalse(row.fusion_path.is_traceable)

    @parameterized.named_parameters(*_SIBLINGS)
    def test_device_and_host_agree(self, device: RowCase, host: RowCase) -> None:
        # A device row and its host sibling are two implementations of one
        # function; the seam is only worth coding against if the same call means
        # the same thing through both. Only the pairs that exist are enumerated,
        # so the case count is the number actually compared rather than a device
        # row count padded with skips.
        np.testing.assert_array_equal(
            np.asarray(device.make().digest(_MSG)),
            np.asarray(host.make().digest(_MSG)),
        )


class RegistryTest(absltest.TestCase):
    def test_every_shipped_row_is_registered(self) -> None:
        # The list is only a contract if it cannot silently fall behind the
        # package, so the sweep walks the SOURCE TREE rather than the modules
        # the registry already names — deriving the search boundary from the
        # thing being checked would let a whole new row module ship with zero
        # rows registered and still pass. `markers_test` sweeps the same way for
        # the same reason.
        package = pathlib.Path(next(iter(hash_frx.__path__)))
        registered = {type(c.make()) for c in ALL_ROWS}
        missing = []
        for source in sorted(package.rglob("*.py")):
            if "testing" in source.relative_to(package).parts:
                continue
            name = ".".join(
                ("hash_frx", *source.relative_to(package).with_suffix("").parts)
            ).removesuffix(".__init__")
            module = importlib.import_module(name)
            for attr, obj in vars(module).items():
                if attr.startswith("_") or not isinstance(obj, type):
                    continue
                if obj.__module__ != name or obj in registered:
                    continue
                # The bases themselves are the contract, not rows on it.
                if obj in (Row, DeviceRow, HostRow):
                    continue
                if issubclass(obj, (DeviceRow, HostRow)):
                    missing.append(f"{name}.{attr}")
        self.assertEqual(missing, [], f"unregistered rows: {missing}")


if __name__ == "__main__":
    absltest.main()

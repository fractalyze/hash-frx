# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every shipped row, held to the seam's contract at once.

`byte_hash.py` states the contract; thirty-nine rows implement it. Each family
tested its own, so a rule could hold everywhere it was checked and still be
missing from whichever row shipped last — which is what happened twice, when the
seam sweeps behind #211 and #215 both had to be re-applied to SM3 and BLAKE2s
after the fact. This walks the list instead.

The equality cases carry the most weight and read like the least: a row's
`__eq__`/`__hash__` is its jit cache key, and `byte_hash.Row` states what that
costs when it is wrong. That is why the registry carries `variants`, and why
"equal" and "distinguishable" are both asserted rather than just the first.

**Which list a case walks says which contract it is asserting.** The equality
cases walk `ALL_ROWS`, because equality belongs to `Row`; the rest walk
`BYTE_HASH_ROWS`, because they ask about `digest`, which belongs to `ByteHash`.
"""

from __future__ import annotations

import importlib
import re

import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

from hash_frx.byte_hash import ByteHash, DeviceRow, Row
from hash_frx.testing.package_sweep import (
    declared,
    declared_anywhere,
    shipped_sources,
)
from hash_frx.testing.rows import (
    ALL_ROWS,
    BYTE_HASH_ROWS,
    RowCase,
)

_MSG = np.arange(3 * 40, dtype=np.uint8).reshape(3, 40)


def _named(cases: tuple[RowCase, ...]) -> list[tuple[str, RowCase]]:
    return [(c.name, c) for c in cases]


class RowEqualityTest(parameterized.TestCase):
    """The jit cache key, row by row — every row, byte hash or not."""

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
        *[(f"{c.name}_{i}", c, i) for c in ALL_ROWS for i in range(len(c.variants))]
    )
    def test_different_parameters_compare_unequal(
        self, case: RowCase, index: int
    ) -> None:
        # The lenient-direction failure: a row that adds a parameter and forgets
        # to compare on it serves one parameterization's trace for another's.
        # One case per parameter, because a row with two of them and one variant
        # leaves the other untested — which is how `_Blake3Hash`'s output-length
        # half went unexercised.
        self.assertNotEqual(case.make(), case.variants[index]())

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
    """The parts of the `ByteHash` contract that hold for every byte hash."""

    @parameterized.named_parameters(*_named(BYTE_HASH_ROWS))
    def test_satisfies_the_protocol(self, case: RowCase) -> None:
        self.assertIsInstance(case.make(), ByteHash)

    @parameterized.named_parameters(*_named(BYTE_HASH_ROWS))
    def test_digest_size_is_positive_and_matches_output(self, case: RowCase) -> None:
        row = case.make()
        self.assertGreater(row.digest_size, 0)
        self.assertEqual(np.asarray(row.digest(_MSG)).shape, (3, row.digest_size))

    @parameterized.named_parameters(*_named(BYTE_HASH_ROWS))
    def test_rank_is_rejected_at_the_seam(self, case: RowCase) -> None:
        # A 1-D message is the common miss: one message is B=1, not a bare [L].
        # It must fail at the seam, not from inside a marked region's trace.
        with self.assertRaises((ValueError, TypeError)):
            case.make().digest(np.zeros(40, dtype=np.uint8))

    @parameterized.named_parameters(*_named(BYTE_HASH_ROWS))
    def test_zero_row_batch_digests_to_zero_rows(self, case: RowCase) -> None:
        # B=0 is what #211 fixed everywhere; this is what keeps it fixed for the
        # row that ships next.
        row = case.make()
        out = np.asarray(row.digest(np.zeros((0, 40), dtype=np.uint8)))
        self.assertEqual(out.shape, (0, row.digest_size))

    @parameterized.named_parameters(*_named(BYTE_HASH_ROWS))
    def test_every_byte_hash_returns_a_device_array(self, case: RowCase) -> None:
        # The seam's own law (`byte_hash.ByteHash.digest`): every row is a
        # device row, so every `digest` hands back an `Array`. Asserted over
        # the byte-hash registry so it reaches the rows on no base too —
        # `Mgf1` is a byte hash on neither, and is the row most able to
        # disagree, its `fusion_path` being delegated rather than its own.
        row = case.make()
        self.assertIsInstance(row.digest(_MSG), Array)


class RegistryTest(absltest.TestCase):
    def test_every_shipped_row_is_registered(self) -> None:
        # The list is only a contract if it cannot silently fall behind the
        # package, so the sweep walks the SOURCE TREE rather than the modules
        # the registry already names — deriving the search boundary from the
        # thing being checked would let a whole new row module ship with zero
        # rows registered and still pass.
        registered = {type(c.make()) for c in ALL_ROWS}
        missing = []
        for name, _path in shipped_sources():
            module = importlib.import_module(name)
            for attr, obj in vars(module).items():
                if attr.startswith("_") or not isinstance(obj, type):
                    continue
                if obj.__module__ != name or obj in registered:
                    continue
                # The bases and the seam are the contract, not rows on them.
                if obj in (Row, DeviceRow, ByteHash):
                    continue
                # `Row`, not `DeviceRow`. Flagging only the
                # subclasses made the sweep blind to exactly the rows that
                # subclass `Row` directly — `Mgf1` and `Hmac`, both shipped and
                # both unregistered — while `rows.py` claimed completeness. So
                # the sweep matches on the base every row shares.
                #
                # `digest` is the second door, and it is what makes the base a
                # convention rather than the thing being relied on: a byte hash
                # written without `Row` at all would otherwise be invisible
                # here, and being pinned — the only other declaration — is a
                # second thing the same author has to remember.
                if issubclass(obj, Row) or hasattr(obj, "digest"):
                    missing.append(f"{name}.{attr}")
        self.assertEqual(missing, [], f"unregistered rows: {missing}")


# The pin, as `docs/reference/conventions.md` spells it. `_\w*` and not `_\w+`:
# a module pinning two rows must name them (mypy rejects re-annotating `_`), so
# every pin in the tree today carries a suffix — but the bare `_` of the doc's
# own example must count too, or a row following it reads as unpinned. Anchored
# so a `TypeVar` bound to `type[ByteHash]` is not read as one.
_PIN = re.compile(r"^\s*_\w*: type\[ByteHash\] = (\w+)$", re.M)


# Module -> the rows it pins as `ByteHash`.
_PINS = declared(_PIN)


class PinTest(absltest.TestCase):
    """Every shipped byte hash is pinned, in the module that defines it.

    `RegistryTest` above is the runtime half of the pair — a row is registered,
    so the suite runs it against a live instance. This is the static half, and
    `docs/reference/conventions.md` states it as a package-wide rule that was
    a hand-written instance per module with nothing checking the next one.

    Neither substitutes for the other, and `Mgf1` is why: it shipped
    `fusion_path` as a read-only `@property`, which the `ByteHash` Protocol's
    mutable-attribute declaration does not accept, so a consumer annotating a
    parameter `ByteHash` could not be passed one. The pin catches that.
    `RowSeamTest`'s `@runtime_checkable` `assertIsInstance` does not, because
    protocol `isinstance` is `hasattr`-based and a property passes it — and
    `Mgf1` was in neither at the time, which is how it reached `main`.
    """

    def test_every_byte_hash_is_pinned_in_its_own_module(self) -> None:
        missing = []
        for case in BYTE_HASH_ROWS:
            row = type(case.make())
            if row.__name__ not in _PINS.get(row.__module__, set()):
                missing.append(f"{row.__module__}.{row.__name__}")
        self.assertEqual(
            missing,
            [],
            f"no seam-conformance pin in the defining module: {missing}",
        )

    def test_every_pin_names_a_registered_byte_hash(self) -> None:
        # The other direction, which `RegistryTest` cannot cover: it sweeps for
        # `Row` subclasses, so a byte hash written without the base would be
        # invisible to it. Being pinned is the second way a module declares it
        # ships one.
        registered = {type(c.make()).__name__ for c in BYTE_HASH_ROWS}
        unregistered = sorted(declared_anywhere(_PIN) - registered)
        self.assertEqual(
            unregistered,
            [],
            f"pinned as `ByteHash` but in no registry: {unregistered}",
        )


if __name__ == "__main__":
    absltest.main()

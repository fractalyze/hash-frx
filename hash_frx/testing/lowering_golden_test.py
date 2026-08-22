# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The gate's own gate.

`assert_lowering_unchanged` is what the re-layering steps are verified against,
so an assertion that quietly passes on a changed wire surface would let the
whole sequence through unchecked. Every case below is a change the gate exists
to catch, made deliberately, against a marked region built here rather than a
shipped hash — the point is to move ONE thing at a time, which a real hash's
digest path does not allow.

The negative cases matter as much: a gate that fires on a function rename would
be abandoned within a week, so the module-name normalization is pinned too.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hash_frx.fusion import fused_region
from hash_frx.testing.lowering_golden import (
    assert_lowering_unchanged,
    lowering_text,
    normalize,
)

_NAME = "hash_frx.perm.fake_for_test"
_STATE = np.arange(4, dtype=np.uint32)


def _region(state, addend, factor, **_attrs):  # type: ignore[no-untyped-def]
    # `lax.composite` threads the marker's attrs into the decomposition, so a
    # body has to accept them even when, as here, it reads none.
    return (state + addend) * factor


def _marked(state, *, name=_NAME, version=1, swap=False, **attrs):  # type: ignore[no-untyped-def]
    """A marked region whose name, version, attrs and operand ORDER are all
    dials — one per thing the gate has to notice."""
    a, f = fnp.asarray(np.uint32(7)), fnp.asarray(np.uint32(3))
    ops = (f, a) if swap else (a, f)
    body = (lambda s, x, y, **kw: _region(s, y, x, **kw)) if swap else _region
    return fused_region(body, state, *ops, name=name, version=version, **attrs)


class GateBitesTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.state = fnp.asarray(_STATE)
        self.golden = lowering_text(lambda s: _marked(s), self.state)

    def test_identical_lowering_passes(self) -> None:
        # The gate is useless if it cannot recognize the unchanged program.
        assert_lowering_unchanged(
            self, lambda s: _marked(s), self.state, golden=self.golden
        )

    def test_a_renamed_entry_function_still_passes(self) -> None:
        # frx names the module after the traced callable, so a rename or a
        # re-wrap moves `module @jit_...` without moving the program. If the
        # gate fired here it would be abandoned as noise.
        def digest_under_a_new_name(s):  # type: ignore[no-untyped-def]
            return _marked(s)

        assert_lowering_unchanged(
            self, digest_under_a_new_name, self.state, golden=self.golden
        )

    def test_marker_rename_fails(self) -> None:
        # The failure the byte-exactness suites cannot see: a renamed marker
        # inlines unrecognized and computes identical bytes on a slower path.
        with self.assertRaises(self.failureException) as caught:
            assert_lowering_unchanged(
                self,
                lambda s: _marked(s, name="hash_frx.perm.renamed"),
                self.state,
                golden=self.golden,
            )
        self.assertIn("hash_frx.perm.renamed", str(caught.exception))

    def test_operand_reorder_fails(self) -> None:
        # A name-routed emitter reads its constants positionally, so swapping
        # two operands keeps the bytes and breaks the ABI.
        with self.assertRaises(self.failureException):
            assert_lowering_unchanged(
                self, lambda s: _marked(s, swap=True), self.state, golden=self.golden
            )

    def test_version_bump_fails(self) -> None:
        # `composite.version` is how a contract change stages; a silent bump
        # would ship an ABI change as a refactor.
        with self.assertRaises(self.failureException):
            assert_lowering_unchanged(
                self, lambda s: _marked(s, version=2), self.state, golden=self.golden
            )

    def test_dropped_attribute_fails(self) -> None:
        # Attrs carry which primitive runs inside a generic envelope, so an
        # attr that stops being emitted is a routing change.
        golden = lowering_text(lambda s: _marked(s, primitive="fake"), self.state)
        with self.assertRaises(self.failureException):
            assert_lowering_unchanged(
                self, lambda s: _marked(s), self.state, golden=golden
            )

    def test_lost_composite_fails(self) -> None:
        # The quietest failure of all: the region stops being marked. Right
        # bytes, no kernel, nothing else notices.
        with self.assertRaises(self.failureException):
            assert_lowering_unchanged(
                self,
                lambda s: _region(
                    s, fnp.asarray(np.uint32(7)), fnp.asarray(np.uint32(3))
                ),
                self.state,
                golden=self.golden,
            )

    def test_failure_message_is_a_diff(self) -> None:
        # A 4000-line inequality is unreadable; the diff is the whole value of
        # the message on a real row.
        with self.assertRaises(self.failureException) as caught:
            assert_lowering_unchanged(
                self, lambda s: _marked(s, version=2), self.state, golden=self.golden
            )
        message = str(caught.exception)
        self.assertIn("--- golden", message)
        self.assertIn("+++ actual", message)


class NormalizationTest(absltest.TestCase):
    def test_only_the_module_name_is_erased(self) -> None:
        # Keeping the normalization minimal is what keeps the gate strict, so
        # the substitution is held to exactly one line.
        text = lowering_text(lambda s: _marked(s), fnp.asarray(_STATE))
        self.assertIn("module @<jit>", text)
        self.assertNotIn("@jit__lambda", text)
        self.assertIn("stablehlo.composite", text)

    def test_normalize_is_idempotent(self) -> None:
        text = lowering_text(lambda s: _marked(s), fnp.asarray(_STATE))
        self.assertEqual(normalize(text), text)

    def test_lowering_is_deterministic_in_process(self) -> None:
        # The cross-process determinism this gate relies on was measured when
        # it was written; this keeps the in-process half honest if frx ever
        # starts numbering SSA values from a counter that outlives a trace.
        args = (fnp.asarray(_STATE),)
        self.assertEqual(
            lowering_text(lambda s: _marked(s), *args),
            lowering_text(lambda s: _marked(s), *args),
        )


if __name__ == "__main__":
    absltest.main()

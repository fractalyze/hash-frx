# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only: pin a family's emitter routing, so both marker arms are reachable
on whichever leg is running.

A family picks its marker from `_routes_to_dedicated_emitter()` — the pin AND a
backend that carries the arm (`fusion.routing`). That answer differs per leg,
and for a family whose emitter nobody has written yet it is `False` everywhere,
so a test that asserts on the DEDICATED arm has to pin the answer rather than
take it. Every permutation family that does so wrote its own copy of the patch;
this is the one home (#193).

**Patch the decision, not one of its inputs.** Patching
`_DEDICATED_EMITTER_AVAILABLE` alone leaves a leg absent from
`_EMITTER_BACKENDS` on the generic marker in BOTH arms — which does not fail,
it makes the dedicated assertions vacuous. Patching the combined answer holds
whatever the backend tuple says, which is the point of patching the decision
rather than one of its inputs.

**Wrap CONSTRUCTION, not just the call.** Every family reads the decision in
`__init__` to fix the marker name it will carry, so a patch around `permute`
alone lands too late.

Neither arm needs the emitter to exist: the cases these serve read the jaxpr or
the lowered module, never a compiled one, so nothing has to recognize the
marker for the assertion to mean something.

**Not for testing the gate itself.** That the routing is a *conjunction* — a
backend without an arm stays generic even when the pin is `True` — is a real
regression these families guard, and it can only be shown by patching
`_DEDICATED_EMITTER_AVAILABLE` and `_EMITTER_BACKENDS` separately. Those cases
patch the inputs deliberately; routing them through here would delete the only
thing they check.

`pinned_routing` rather than the `routing` this was filed as: `fusion.routing`
is production and answers the opposite question — what the routing *is*, not
what a case wants to pretend it is.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from types import ModuleType
from unittest import mock


@contextlib.contextmanager
def pinned_routing(module: ModuleType, dedicated: bool) -> Iterator[None]:
    """Pin `module`'s routing decision to `dedicated` for the duration.

    `module` is the one that DEFINES `_routes_to_dedicated_emitter`, which is
    not always the module under test: a sponge or a byte hash takes its marker
    from the permutation's decision, so `keccak.testing.sponge_test` pins
    `keccak.permutation` rather than `keccak.sponge`.
    """
    with mock.patch.object(module, "_routes_to_dedicated_emitter", lambda: dedicated):
        yield


def dedicated_emitter(module: ModuleType) -> contextlib.AbstractContextManager[None]:
    """Route as if the dedicated emitter were reachable, so the family carries
    its own marker."""
    return pinned_routing(module, True)


def generic_emitter(module: ModuleType) -> contextlib.AbstractContextManager[None]:
    """Route as if it were not, so the family falls back to
    `FUSED_REGION_MARKER`."""
    return pinned_routing(module, False)

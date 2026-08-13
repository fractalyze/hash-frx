# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Whether the backend under test has the BLAKE3 fusion emitter.

A marked `hash_frx.blake3` call produces the same bytes either way, so nothing
about correctness reads this. What it decides is *cost*: with no emitter the
marker inlines and XLA codegens the whole unrolled hash, and that compile is
super-linear in the compression count — measured on the wheel that first carried
the emitter (frx 0.10.2.dev20260812135255), a three-chunk `digest` compiles in
1.7s where the emitter is present and had not finished after 150s where it is
not.

So the suites that would pay it cap what they run on the legs that cannot afford
it — a shorter differential sweep, seam rows that stop below the tree layer,
value tables that read the unmarked decomposition instead of the shipped entry
points. Each site states what it gives up. This is the one place the condition
is spelled, so a leg gaining an emitter lifts every one of them at once.

Both legs now have one, so on `cpu` and `gpu` nothing is capped and the tables
read the shipped entry points. The list is explicit rather than `!= "unknown"`
because a leg is capped until an emitter is *written* for it: a backend absent
here — Metal today — still inlines the marker and still pays the cliff.
"""

from __future__ import annotations

import frx

# The Fractalyze XLA emitters: fractalyze/xla#499 registers the recognizer and
# rewriter on the GPU compiler, fractalyze/xla#507 on the CPU one. On CPU every
# entry point — `digest` at one block, one chunk and three chunks, `xof` at 64
# and 131 bytes, both keyed modes, `derive_key`, `non_root_digest` and
# `parent_digest` — compiles to exactly one custom fusion in ~1s, where the
# inline form had not finished in 150s.
HAS_BLAKE3_EMITTER = frx.default_backend() in ("cpu", "gpu")

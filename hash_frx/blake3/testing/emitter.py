# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Whether the backend under test has the BLAKE3 fusion emitter.

A marked `hash_frx.digest.blake3` call produces the same bytes either way, so nothing
about correctness reads this. What it decides is *cost*: with no emitter the
marker inlines and XLA codegens the whole unrolled hash, and that compile is
super-linear in the compression count — measured on the wheel that first carried
the emitter (frx 0.10.2.dev20260812135255), a three-chunk `digest` compiles in
1.7s where the emitter is present and had not finished after 150s where it is
not.

So the suites that would pay it cap what they run on the legs that cannot afford
it — a shorter differential sweep, seam rows that stop below the tree layer,
value tables that read the unmarked decomposition instead of the shipped entry
points. Each site states what it gives up.

Both legs now have one, so on `cpu` and `gpu` nothing is capped and the tables
read the shipped entry points. The condition is read off the production switch
in `byte_hashes` — the same pin+backend conjunction that derives the rows'
`fusion_path` — so the caps and the seam cannot disagree about which legs pay
the cliff, and a leg gaining an emitter lifts every cap when the production
tuple moves. A backend absent there — Metal today — still inlines the marker
and still pays it.
"""

from __future__ import annotations

from hash_frx.blake3.byte_hashes import _routes_to_dedicated_emitter

# Measured on the CPU leg when the emitters landed (fractalyze/xla#499 gpu,
# #507 cpu): every entry point — `digest` at one block, one chunk and three
# chunks, `xof` at 64 and 131 bytes, both keyed modes, `derive_key`,
# `non_root_digest` and `parent_digest` — compiles to exactly one custom fusion
# in ~1s, where the inline form had not finished in 150s.
HAS_BLAKE3_EMITTER = _routes_to_dedicated_emitter()

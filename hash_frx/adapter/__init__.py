# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Adapter: the modes that sit on top of an extension.

The model's third layer. A primitive is a permutation or a compression, an
extension is the schedule that feeds it a message, and an adapter is a
construction built *over* a finished hash rather than inside one — HMAC keys a
`ByteHash`, HKDF and PBKDF2 key HMAC.

The layer exists as a directory because that is what stops the next one landing
at the top level beside the primitives, which is where these three were. It is
an open set, unlike `extension/` — the epic's claim is that there are exactly
three extensions, and no such claim is made here.

Nothing is re-exported from this `__init__`. The package's public surface is
`hash_frx/__init__.py`'s lazy table, which names these modules directly; a
second re-export layer here would be a second place for a name to go stale.
"""

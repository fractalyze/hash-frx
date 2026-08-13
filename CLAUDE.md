# Project context for Claude Code

Everything load-bearing lives in the repo docs and the module docstrings they
point at. Treat those as the source of truth; this file is the map plus the two
rules every change must respect.

- **Project overview, install, dev quick-start:** [`README.md`](README.md)
- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md)
- **Coding conventions — what a primitive is written to, and why the code does
  not read the way a Python reviewer would prefer:**
  [`docs/reference/conventions.md`](docs/reference/conventions.md)
- **Design notes — why two seams, and what may live in this repo at all:**
  [`docs/blocks/hash.md`](docs/blocks/hash.md)
- **Dev environment — backend selection, the CUDA-12 requirement, running
  against an unreleased XLA, the compile cache:**
  [`docs/reference/development.md`](docs/reference/development.md)
- **Detailed design & open decisions:** tracked on GitHub — epic issue
  [fractalyze/hash-frx#1](https://github.com/fractalyze/hash-frx/issues/1).

## Two non-negotiables

- **Application-agnostic.** No proving scheme, signature scheme, or blockchain
  leaks in. A primitive here reads its own parameters and nothing else: domain
  separation, parameter choice, and padding conventions belong to the consumer.
  Two seams carry that — [`Permutation`](hash_frx/permutation.py) over a field
  dtype and [`ByteHash`](hash_frx/byte_hash.py) over raw bytes — and a consumer
  reads `width`/`dtype` or `digest_size` rather than naming the hash it runs on.
  If scheme-specific knowledge is creeping in, it belongs in the consumer.
- **Fusion is a correctness-of-design property.** A permutation call, a digest
  call, and a whole sponge hash each lower to **one device unit by
  construction** — a `lax.composite` marker a Fractalyze XLA rewriter turns into
  one kernel — never by a per-primitive compiler pattern-match. So a reduction,
  gather, dynamic index, or `while` inside a generically marked body is a **bug**
  rather than a missed optimization: it splits the kernel while still producing
  the right bytes, so only the GPU leg catches it. What the unit is, which
  markers exist, and how a body is held to the rule is stated once in the
  [fusion contract](docs/README.md#the-fusion-contract).

  **The unit comes from a *recognized* marker, never from the fallback.** A bare
  `zorch.fused_region` carries no live-width operand, and the rewriter declines
  those on purpose, so it inlines and ordinary fusion materializes the
  intermediates — a single permutation lands as ~70 fusions rather than one. So
  `has_dedicated_fusion = False` means *no kernel*, not *a slower kernel*, and
  the straight-line rule above is what keeps a body wrappable **for when** its
  emitter ships, not something that buys a unit on its own.

  **Reachable is two questions, not one**: does the pinned plugin carry the
  emitter, and does the backend being compiled for. Each switch tracks the
  `frx>=` floor in `pyproject.toml` for the first, and an explicit backend list
  for the second — the Keccak arms are GPU-only, so a CPU build takes the
  fallback no matter what the pin says. Routing to an emitter the backend lacks
  is not free the way an unrecognized name is: a whole-hash marker traces the
  entire chain into one composite, and where nothing honours it that trace is
  spent for nothing. Byte equality holds throughout, so the only case that can
  see any of this is one reading the *compiled* module for `kind=kCustom`, on a
  leg that has the emitter.

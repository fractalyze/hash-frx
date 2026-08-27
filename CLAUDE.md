# Project context for Claude Code

Everything load-bearing lives in the repo docs and the module docstrings they
point at. Treat those as the source of truth; this file is the map plus the two
rules every change must respect.

- **Project overview, install, dev quick-start:** [`README.md`](README.md)
- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md)
- **Coding conventions — what a primitive is written to, and why the code does
  not read the way a Python reviewer would prefer:**
  [`docs/reference/conventions.md`](docs/reference/conventions.md)
- **Design notes — the primitive / extension / adapter layering, why a schedule
  is written once per construction, and what may live in this repo at all:**
  [`docs/blocks/hash.md`](docs/blocks/hash.md)
- **Consuming this package from another repo — the import form, the Bazel dep
  that goes with it, and what to do about a missing name:**
  [`docs/reference/consuming.md`](docs/reference/consuming.md)
- **Dev environment — backend selection, the two test legs (`bazel test` runs
  only one), the lowering gate a wire-preserving refactor needs, the CUDA-12
  requirement, running against an unreleased XLA, the compile cache:**
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

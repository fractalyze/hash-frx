# hash-frx

FRX-native hash primitives — algebraic permutations and byte hashes, each written
to lower to a **single fused kernel**.

`hash-frx` sits between **FRX** — Fractalyze's fork of
[JAX](https://github.com/jax-ml/jax) — and everything that hashes: the proving
blocks in [`zorch`](https://github.com/fractalyze/zorch), the signature schemes in
`sig-frx`, and any other FRX consumer. FRX provides tracing and codegen, lowered
through **Fractalyze XLA**, its fork of stock [XLA](https://github.com/openxla/xla)
that adds native field and elliptic-curve types.

## Design philosophy

- **Two seams, no concrete hash in the consumer.** `Permutation` is a fixed-width
  permutation over a field dtype (Poseidon, Poseidon2); `ByteHash` maps a batch of
  equal-length byte messages to digests (SHA-256, BLAKE3, Keccak/SHA-3). A
  consumer reads `width`/`dtype` or `digest_size` and calls `permute`/`digest` —
  it never names the hash it runs on.
- **Fusion by construction.** A permutation call, a digest call, and each
  sponge `absorb`/`squeeze` lower to one fused kernel *by construction* — a
  `lax.composite` marker an XLA emitter recognizes — never by a per-primitive
  compiler pattern-match.
- **Application-agnostic.** No proving scheme, signature scheme, or blockchain
  leaks in. Domain separation, parameter choice, and padding conventions belong
  to the consumer.
- **Byte-exact with the standard.** A byte hash reproduces its specification
  exactly (SHA-256 = FIPS 180-4, BLAKE3 = the BLAKE3 spec, SHA-3/SHAKE = FIPS
  202), verified against the published test vectors.

## Status

Bootstrapping. The symmetric layer is being extracted from `zorch/hash` and
extended with BLAKE3 and the Keccak family; `zorch` then consumes this repo.
Work is tracked on the [issues](https://github.com/fractalyze/hash-frx/issues).

## Development

**Python 3.11.** `frxlib` publishes cp311 wheels only, and both the hermetic
Bazel toolchain and `.python-version` pin that version.

Bazel is the build, and the whole suite is one command — the same one CI's CPU
leg runs:

```sh
bazel test //...
```

Tests are backend-agnostic and default to CPU (`.bazelrc` sets
`FRX_PLATFORMS=cpu`), so a plain run is deterministic on any machine. Run them
on the device to exercise the fusion markers — that leg is the only one that
reports a lost marker, because an unrecognized marker still produces the right
bytes:

```sh
bazel test --test_env=FRX_PLATFORMS=cuda \
    --test_env=XLA_PYTHON_CLIENT_PREALLOCATE=false //...
```

`cuda` is strict — there is no CPU fallback — so a green run really did execute
on the device. Preallocation is off because Bazel runs the test actions
concurrently against the one device, and each process would otherwise claim most
of its memory.

For interactive work outside Bazel, the same pinned toolchain in a virtualenv:

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The extra index carries the `frx` builds and the CUDA plugin wheels, which are
too large for PyPI's per-file limit. `requirements.in` holds the pins;
regenerate the lock with `bazel run //:requirements.update` instead of editing
it by hand.

Install the git hooks with both stages named. Plain `pre-commit install` wires
only the `pre-commit` stage, which leaves the commit-message linter inactive —
formatting hooks fire while a malformed commit message sails through to CI:

```sh
pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org):
a valid type, a lowercase summary with no trailing period, a
header of at most 80 characters, and a body on everything but `docs`. Scope is
free-form. The same linter runs in CI over the pull request title and every
commit in it.

## Documentation

- **Task-indexed hub:** [`docs/README.md`](docs/README.md) — indexes the seams,
  constructions, and implementations by what you are trying to do, and states
  the [fusion contract](docs/README.md#the-fusion-contract) they all share.
- **Contributing with Claude Code:** [`CLAUDE.md`](CLAUDE.md) — the same map,
  plus the two rules every change must respect.

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).

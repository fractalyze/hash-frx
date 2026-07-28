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

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).

# hash-frx docs

Reference indexed by what you're trying to do. For the project overview, install,
and the dev quick-start, see [`../README.md`](../README.md).

The authoritative description of a primitive is its module docstring — every
seam, construction, and implementation states its own contract and the reasoning
behind its shape. This hub is the index into them, plus the one contract that
spans all of them: [fusion](#the-fusion-contract).

The tree mirrors the layering. `hash_frx/*.py` holds the two seams and the
constructions built over them; `hash_frx/<family>/` is one implementation family
(its parameter surface, its normal-form linear layers, its permutation); every
package carries a `testing/` subdir with its tests and the reusable helpers those
tests share.

## Working here

| Question                                                                                   | Where                                             |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| The rules a hash primitive is written to — fusion authoring, the seam and pin requirements, byte-exactness, and the assertions that must bite | [`reference/conventions.md`](reference/conventions.md) |
| Why the layer is shaped this way — two seams, what may live here at all, the SHA-256 pair, fusion as a design property | [`blocks/hash.md`](blocks/hash.md)                 |
| Getting a dev loop — backend selection, the CUDA version trap, an unreleased XLA, the compile cache | [`reference/development.md`](reference/development.md) |

## Seams — the surfaces a consumer codes against

| Question                                                                                   | Where                                             |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| Hashing with a fixed-width permutation, algebraic or bit-oriented, without naming one       | [`permutation.py`](../hash_frx/permutation.py)     |
| Hashing raw bytes byte-identically to a published standard, without naming one              | [`byte_hash.py`](../hash_frx/byte_hash.py)         |

## Constructions over a `Permutation`

| Question                                                                                   | Where                                               |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| One-shot sponge hash — padding-free or Merkle–Damgård, selected per call                    | [`sponge.py`](../hash_frx/sponge.py)                 |
| Interleaved absorb/squeeze with add-mode absorb, for a classic Fiat-Shamir prover            | [`duplex_sponge.py`](../hash_frx/duplex_sponge.py)   |
| n-to-1 truncated-permutation compression, for a hash tree                                   | [`compression.py`](../hash_frx/compression.py)       |

## Implementations

| Question                                                                                   | Where                                               |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| Poseidon2 — permutation, parameter surface, and its two linear layers                       | [`poseidon2/`](../hash_frx/poseidon2)                |
| Classic Poseidon — the naive Hades schedule and the optimized-sparse refactor of it          | [`poseidon/`](../hash_frx/poseidon)                  |
| SHA-256 — batched digest, incremental midstate, and the device / host `ByteHash` pair        | [`sha256.py`](../hash_frx/sha256.py)                 |
| Keccak-f[1600] — the permutation under SHA-3, SHAKE and Keccak-256, over uint32 lane halves  | [`keccak/`](../hash_frx/keccak)                      |
| BLAKE3 — the compression function every mode is built on, and the chunk over it that hashes a one-chunk message | [`blake3/`](../hash_frx/blake3)                      |
| SHA3-256, SHAKE128, SHAKE256 and Keccak-256 — the byte hashes over one sponge, and that sponge (why it is not `sponge.py`) | [`keccak/byte_hashes.py`](../hash_frx/keccak/byte_hashes.py), [`keccak/sponge.py`](../hash_frx/keccak/sponge.py) |

## Fusion machinery

| Question                                                                                   | Where                                               |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| Marking a region as one fused kernel, or routing it to a dedicated emitter                  | [`fusion.py`](../hash_frx/fusion.py)                 |
| The single place a `lax.composite` marker is emitted                                        | [`_composite.py`](../hash_frx/_composite.py)         |
| Writing a linear layer that does not split the kernel — the column-scaled normal form        | [`linear.py`](../hash_frx/linear.py)                 |
| Rotating, rolling, or packing bytes to words without emitting a call or a gather              | [`word.py`](../hash_frx/word.py)                     |

## Test helpers

| Question                                                                                   | Where                                                       |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| Proving a body still lowers to one kernel                                                   | [`testing/fusion_ready.py`](../hash_frx/testing/fusion_ready.py) |
| Proving a module-level jit zone is not re-traced per instance                                | [`testing/jit_cache.py`](../hash_frx/testing/jit_cache.py)   |
| Drawing field elements in a test, in the production Montgomery encoding                     | [`testing/random_field.py`](../hash_frx/testing/random_field.py) |

## The fusion contract

The unit is one **marked region**: a `frx.lax.composite` that a Fractalyze XLA
rewriter turns into a single custom-fusion kernel. A permutation call, a digest
call, and a whole sponge hash are each one such region — by construction, never
by a compiler pattern-match on the hash. Markers come in two flavors.

**Generic** — `zorch.fused_region`. The generic rewriter accepts a straight-line
element-wise body only: no loop, reduction, gather, dynamic index, or call. That
constraint is what shapes the code — the authoring rules it forces are in
[`reference/conventions.md`](reference/conventions.md#a-marked-body-is-authored-to-lower-not-to-read).

**Name-routed** — `zorch.poseidon`, `zorch.sparse_poseidon`, `zorch.poseidon2`,
`zorch.sha256`, `zorch.sponge_hash`. Each goes to a dedicated emitter that owns
its own operand ABI and, unlike the generic path, tolerates reductions and calls.
A permutation advertises which path it is on through `has_dedicated_fusion`, and
hands out its operand layout through `fused_region_spec` — that pair is what lets
a consumer wrap a whole computation as one region without naming the hash
underneath it.

**Marker names are a wire ABI.** The rewriters match by name, and an unrecognized
name does not error: the composite inlines and fusion is silently lost. A
contract change therefore rides `composite.version` rather than a rename. The
operand layout is equally part of that ABI, which is what keeps a region's
constants from being written as closed-over values.

**Losing fusion is silent**, and nothing about it is caught by comparing bytes: a
marker that stops being recognized inlines and still computes the right answer,
and where a marker is absent there is a working fallback rather than a failure —
a `Sponge` over a permutation without dedicated fusion runs its absorb as a
`while_loop` over `permute`. So the gate is the lowering itself, asserted in the
suite and re-checked on the GPU CI leg; the assertions that carry it, and the
rule that each must be shown to bite, are in
[`reference/conventions.md`](reference/conventions.md#testing).

Findings, measurements, and open fusion decisions live on the epic
[fractalyze/hash-frx#1](https://github.com/fractalyze/hash-frx/issues/1).

## Growing this hub

A new primitive family adds a row under Implementations and lands in its own
`hash_frx/<family>/` package, not a buried flat file. A new construction over an
existing seam adds a row under Constructions. A design note that outgrows its
module docstring lands in `blocks/`, and a convention that a reviewer has to
point at more than once lands in `reference/`.

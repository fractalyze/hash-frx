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
| Why the layer is shaped this way — two seams, what may live here at all, the SHA-256 pair, which hashes get a host row, fusion as a design property | [`blocks/hash.md`](blocks/hash.md)                 |
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
| The sponge schedules themselves — the byte one over static counts, the field one over a runtime count, and why they are two | [`extension/sponge.py`](../hash_frx/extension/sponge.py) |
| How a message becomes whole blocks — `PadRule` for Merkle–Damgård, `SpongePad` for the sponges | [`extension/pad.py`](../hash_frx/extension/pad.py) |
| The tree schedule — the unit ceiling, the intra-chunk chain, and why bottom-up pairing is the spec's own tree | [`extension/tree.py`](../hash_frx/extension/tree.py) |
| Interleaved absorb/squeeze with add-mode absorb, for a classic Fiat-Shamir prover            | [`duplex_sponge.py`](../hash_frx/duplex_sponge.py)   |
| n-to-1 truncated-permutation compression, for a hash tree                                   | [`compression.py`](../hash_frx/compression.py)       |

## Constructions over a `ByteHash`

| Question                                                                                   | Where                                               |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| HMAC — the FIPS 198-1 keyed MAC, one class parameterized by hash and block size             | [`hmac.py`](../hash_frx/hmac.py)                     |
| HKDF — RFC 5869 extract-then-expand, the KDF the composition standards name                  | [`hkdf.py`](../hash_frx/hkdf.py)                     |
| PBKDF2 — the RFC 8018 iterated KDF as a traced c-loop, with the ipad/opad midstate fast path (BIP-39's seed derivation) | [`pbkdf2.py`](../hash_frx/pbkdf2.py)                 |

## Implementations

| Question                                                                                   | Where                                               |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| Poseidon2 — permutation, parameter surface, and its two linear layers                       | [`poseidon2/`](../hash_frx/poseidon2)                |
| Classic Poseidon — the naive Hades schedule and the optimized-sparse refactor of it          | [`poseidon/`](../hash_frx/poseidon)                  |
| Vision — the binary-field Marvellous permutation (Vision Mark-32 over the GF(2^32) tower)    | [`vision/`](../hash_frx/vision)                      |
| SHA-256 — batched digest, incremental midstate, and the device / host `ByteHash` pair        | [`sha256.py`](../hash_frx/sha256.py)                 |
| SHA-512 — the 64-bit SHA-2 sibling over uint32 half pairs: batched digest, incremental midstate, the device / host `ByteHash` pair, and the truncated variants SHA-384 and SHA-512/256 as IV rows on the same marker | [`sha512.py`](../hash_frx/sha512.py)                 |
| Keccak-f[1600] — the permutation under SHA-3, SHAKE and Keccak-256, over uint32 lane halves  | [`keccak/`](../hash_frx/keccak)                      |
| BLAKE3 — the chunk tree, and hash / keyed / derive-key as `ByteHash` rows at any output length over one compression function | [`blake3/`](../hash_frx/blake3), [`blake3/modes.py`](../hash_frx/blake3/modes.py), [`blake3/rows.py`](../hash_frx/blake3/rows.py) |
| BLAKE2b — the HAIFA `ByteHash` pair: the device row over 64-bit half pairs behind its digest marker, and the host row over `hashlib` (why the host row came first) | [`blake2b/`](../hash_frx/blake2b) |
| BLAKE2s — the 32-bit RFC 7693 sibling at native uint32: device and host rows on its own digest marker | [`blake2s.py`](../hash_frx/blake2s.py) |
| SM3 — the GB/T 32905 ShangMi hash, SHA-256's structural cousin: device and host rows on its own digest marker | [`sm3.py`](../hash_frx/sm3.py) |
| Grøstl-256 — the AES-round Merkle–Damgård `ByteHash` over GF(2^8), with a bitsliced S-box and a testonly host partner | [`grostl/`](../hash_frx/grostl) |
| Ascon-Hash256 and Ascon-XOF128 — the NIST SP 800-232 lightweight-standard sponge `ByteHash` rows over uint32 word halves, plus Ascon-p[12] as a `Permutation`, with a testonly host partner | [`ascon/`](../hash_frx/ascon) |
| RIPEMD-160 — the little-endian Merkle–Damgård `ByteHash` (Bitcoin HASH160's second half), with a testonly host partner | [`ripemd160.py`](../hash_frx/ripemd160.py) |
| SHA3-256, SHA3-512, SHAKE128, SHAKE256 and Keccak-256 — the byte hashes over one sponge, and that sponge (why it is not `sponge.py`) | [`keccak/byte_hashes.py`](../hash_frx/keccak/byte_hashes.py), [`keccak/sponge.py`](../hash_frx/keccak/sponge.py) |

## Fusion machinery

| Question                                                                                   | Where                                               |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| Marking a region as one fused kernel, or routing it to a dedicated emitter                  | [`fusion.py`](../hash_frx/fusion.py)                 |
| The single place a `lax.composite` marker is emitted                                        | [`_composite.py`](../hash_frx/_composite.py)         |
| Writing a linear layer that does not split the kernel — the column-scaled normal form        | [`linear.py`](../hash_frx/linear.py)                 |
| Rotating, rolling, or packing bytes to words without emitting a call or a gather              | [`word.py`](../hash_frx/word.py)                     |
| 64-bit rotate / shift / carry-add / XOR over `(lo, hi)` uint32 half pairs, shared by the 64-bit-word hashes | [`word64.py`](../hash_frx/word64.py)                 |

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

**Name-routed** — spelled `hash_frx.<kind>.<name>` by fusible-unit kind:
`hash_frx.perm.{poseidon, poseidon_sparse, poseidon2, keccak_f, vision, ascon_p}`,
`hash_frx.compress.{blake3, blake3_parent}`, and
`hash_frx.digest.{sha256, sha512, field_sponge, keccak_sponge, blake3, grostl256, ascon_hash256, ascon_xof128, ripemd160, blake2b, blake2s, sm3}`
(`hash_frx/markers.py` is the registry). Each goes to a
dedicated emitter that owns its own operand ABI and, unlike the generic path,
tolerates reductions and calls; the recognizers also still accept the
pre-namespace spellings until they retire.

Two of them wrap a whole hash rather than one primitive — `hash_frx.digest.field_sponge`
over the field sponge, `hash_frx.digest.keccak_sponge` over the byte one — and both are
assembled by `fused_region_over`, which rebuilds the permute from the operand
layout `fused_region_spec` hands out. That is what keeps a construction from
naming the permutation's constants, and what makes a whole absorb one region
instead of a marked permute per block with the glue between them left outside.

The markers wait for the toolchain in two different ways, because the cost of
being early is not the same for both. `hash_frx.digest.blake3`,
`hash_frx.digest.sha256`, `hash_frx.digest.sha512`, `hash_frx.digest.grostl256`,
`hash_frx.digest.ascon_hash256`, `hash_frx.digest.ripemd160`,
`hash_frx.digest.blake2b`, `hash_frx.digest.blake2s` and
`hash_frx.digest.sm3` are emitted
whether or not the pinned plugin recognizes the name: an
unrecognized *name* only inlines, so being early costs the fusion and nothing
else, and the hash reports `fusion_path = GENERIC` while carrying its marker.
`hash_frx.perm.keccak_f` is emitted only where the plugin ships its emitter — off
the pin *and* the backend, the Keccak arms being GPU-only — and the `frx>=`
floor in `pyproject.toml` is what holds the pin half true. That switch has to
track the pin rather than be left optimistic, because a permutation's marker
also decides whether a `Sponge` over it wraps its whole hash as
`hash_frx.digest.field_sponge` — and that marker carries a `permutation` discriminator
a plugin without the arm rejects outright, which is a failed compile rather
than a lost kernel.

`hash_frx.digest.sha256_bytes_len` is gated the same way and for a third
reason. It takes the message length as an operand rather than as part of the
message shape, so one compiled kernel serves every length its buffer can hold —
but its decomposition has to derive a data-dependent block count from that
operand, which in plain HLO means speculating every block the buffer could need.
Inlining it is therefore *slower* than the static-length marker it replaces,
where an unrecognized name normally costs only the fusion. Being early costs
performance rather than nothing, so the switch tracks the pin and the backend.

A permutation advertises which path it is on through `fusion_path`
(`hash_frx.fusion.FusionPath`: `DEDICATED` / `GENERIC` / `HOST`, the last only
on the byte seam), and hands out its operand layout through
`fused_region_spec` — that pair is
what lets a consumer wrap a whole computation as one region without naming the
hash underneath it.

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

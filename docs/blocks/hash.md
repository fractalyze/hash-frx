# hash — the symmetric layer

The *why* behind `hash_frx/`. The *what* lives in the code and its tests, which
run on every commit. Open decisions live on the epic issue
[fractalyze/hash-frx#1](https://github.com/fractalyze/hash-frx/issues/1).

## Why two seams

A hash is reached through one of two Protocols, and which one depends on what it
consumes rather than on how it is built.

**`Permutation`** ([`permutation.py`](../../hash_frx/permutation.py)) is a
fixed-width permutation over a single dtype — `width`, `dtype`, `permute`.
The sponge, the duplex sponge, and the compression function all read
`width`/`dtype` to size and allocate state and then call `permute`; none of them
names a concrete hash. Poseidon2 is one implementation, classic Poseidon a
second, and any other fixed-width permutation drops in unchanged.

That dtype is deliberately not required to be a field, which is what lets the
bit-oriented hashes share the seam rather than fork it. `Sponge` and
`Compression` do no arithmetic on state — they allocate it, index it, and
overwrite lanes — so a machine-word permutation satisfies the Protocol as
written. `DuplexSponge` is the exception and owns its own constraint: its absorb
merges with `+`, which is the intended group operation for a field and the wrong
one for machine words, where a Keccak-style sponge merges by XOR. That is a
property of the construction, so it is stated there rather than narrowed into the
seam. A bit-oriented duplex is its own construction for the same reason
`DuplexTranscript` is separate from `DuplexSponge`: the conventions diverge on
several independent axes, and one config object holding two of them is not reuse.

**`ByteHash`** ([`byte_hash.py`](../../hash_frx/byte_hash.py)) is the byte
sibling: `digest(uint8[B, L]) -> uint8[B, digest_size]`, byte-identical to the
hash's standard. A consumer reads `digest_size` and calls `digest`.

The split is load-bearing because the two have no common surface below `digest`.
A byte hash's internal construction differs per family — Merkle–Damgård chains a
midstate, a tree hash combines subtree roots, a sponge absorbs into a state — so
`digest` is the *only* operation that generalizes across them. That is also why a
shared **streaming** surface is deliberately absent from the seam: the
incremental state has a different shape in each construction, so a common
`absorb`/`squeeze` pair would be a fiction that fits one family.

Width is not a free parameter anywhere: it is whatever the permutation provides.
The free parameters are `rate`/`out` (`SpongeParams`) and `arity`/`chunk`
(`CompressionParams`), and validation splits along the same line — the parameter
object checks what it can see on its own (`rate >= 1`, `out >= 1`) and the
operator, which knows the width, checks the rest (`rate < width`,
`out <= width`).

## What may live here at all

The seams keep a *consumer* from naming a hash. The reverse obligation — keeping
this repo from naming a consumer — needs its own test, because the tempting
additions are the ones a caller would find useful:

> A shape belongs in `hash-frx` only if its name and ABI survive deleting all
> knowledge of the caller.

`absorb(framing) -> counter-squeeze -> re-absorb` fails it: "counter-squeeze" is a
Fiat-Shamir step, and naming it here would put proving-scheme knowledge in a
library whose defining property is that it holds none.
`absorb_then_squeeze(state, bytes_in, nbytes) -> (state, bytes_out)` passes it —
it describes a Merkle–Damgård operation, and a transcript hop and a
rejection-sampling loop can both ride it without this layer knowing which is
calling.

The test is what decides where a fused streaming surface lives, and the current
split follows from it rather than from where the code happened to sit: the
streaming primitives are here, and the transcript's fused squeeze hop — the
`zorch.sha256_squeeze` marker — stays in the consumer that knows what a
transcript is.

The cost of that split is real and worth stating so nobody re-derives it. Because
the padding choice is data-dependent, both `sha256_stream_absorb` and
`sha256_stream_finalize` compress every candidate block count and select, so an
absorb followed by a finalize emits **3** composites and roughly 818 StableHLO
ops of glue. A consumer streaming at volume pays that until a shared fused
surface exists — which needs a second independent consumer to shape it, since a
surface designed against one real caller and one predicted caller fits the first
and bends for the second.

## Two implementations of one standard

`Sha256` and `HostSha256` ([`sha256.py`](../../hash_frx/sha256.py)) produce the
same FIPS 180-4 bytes and differ only in substrate: one lowers to the device
marker, the other loops `hashlib` on the host. `has_dedicated_fusion` is what a
consumer branches on, exactly as on the permutation side — the substrate is a
value the hash carries, never a class name a caller has to test.

**That holds for SHA-256 and does not yet hold in general.** The flag says a
digest lowers to a hash-*dedicated* marker, which for SHA-256 coincides with
"runs on the device". Keccak's `ByteHash`es
([`keccak/byte_hashes.py`](../../hash_frx/keccak/byte_hashes.py)) are the case where the
two come apart: they run on the device and accept a tracer, but carry only the
generic region marker until a Keccak emitter exists, so every one of them —
device and host alike — reports `False`. A consumer branching on the flag alone will size a nonce window
for the host path against a device hash. The seam anticipated this and named the
remedy — a second field for "the digest returns an `Array`" — which is a decision
owed to the next consumer that has to make the choice, not to the hash that
exposed it.

Both stay, and the reason is not only speed. A byte transcript is host-shaped by
construction — a `bytes` buffer with host framing — so a device hash forces a
device-to-host sync on every squeeze; and a proof-of-work grind sizes its nonce
window off `has_dedicated_fusion`, testing a wide window on a device hash and one
nonce at a time on a host hash. The host implementation is what makes that branch
reachable at all.

The choice between them is **by batch shape, not by machine** — measured on the
CPU backend, `[B, 64]` messages:

| B | `Sha256` | `HostSha256` | winner |
|---:|---:|---:|---|
| 1 | 32.5 us | 1.5 us | host, 22x |
| 32 | 35.4 us | 24.4 us | host |
| 64 | 37.9 us | 47.7 us | device |
| 1024 | 138 us | 754 us | device, 5.5x |

Crossover is around B=48. Device latency is flat out to B=64 because it is
dispatch-bound rather than compute-bound, and these are CPU-backend numbers — so
a sequential caller pays the gap on every machine, not just a machine without a
GPU.

## Names say the construction, not the shape

An identifier names the construction it implements —
`sha256_merkle_damgard`, `SpongeType.MERKLE_DAMGARD` — and shape words like
"chain" stay in prose. A chain is what a Merkle–Damgård fold looks like, not what
it is, and a name that describes the picture stops matching the moment a second
construction has the same picture.

`zorch` carries its own copy of this layer alongside its dependency on this one,
and one symbol differs across the two: `sha256_merkle_damgard` here is
`sha256_chain` there. The divergence is deliberate rather than upstream drift —
the name is corrected in the repo that owns the primitive, not in a copy — so a
reader comparing the two is seeing this repo's correction.

## Fusion by construction

The **permutation is the fusion unit**: `permute` wraps all rounds in one marker
that Fractalyze XLA turns into a single kernel, and a `vmap` over a Merkle layer
or a sponge's block loop batches into that one region rather than multiplying it.
Above it, a whole `Sponge.hash` is itself one marked region when the permutation
carries dedicated fusion, assembled here over the permutation's own operand ABI.

What the layer owes that path — the unrolled rounds, the normal-form linear
layers, the operand rules — is in
[`../reference/conventions.md`](../reference/conventions.md#a-marked-body-is-authored-to-lower-not-to-read),
and what the unit *is* is in the hub's
[fusion contract](../README.md#the-fusion-contract). The design point here is
narrower: the primitives are written *for* a marker, so the marker is not an
optimization applied to them afterwards.

## Out of scope

Domain separation, the choice between a sponge and a compression function for a
given protocol, and any parameter set tied to a protocol belong to the consumer.
This layer supplies the primitive and its fusion contract, and nothing that would
require it to know what the digest is for.

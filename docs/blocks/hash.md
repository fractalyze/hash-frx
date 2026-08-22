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

The byte-side sponge schedule is shared, and it was deliberately not
generalized until it could be shaped against two implementations — a surface
generalized from a single one encodes that implementation's accidents as the
family's. Ascon ([`ascon/`](../../hash_frx/ascon)) was the second, so
[`extension/sponge.py`](../../hash_frx/extension/sponge.py) carries the schedule
both run (absorb a block and permute; then read the rate and permute between
reads, with none after the last) and
[`extension/pad.py`](../../hash_frx/extension/pad.py) carries `SpongePad`. What
stays family-specific is what actually differs: the state and its packing, the
pad parameters, and the initial state.

The **field** sponge ([`sponge.py`](../../hash_frx/sponge.py)) shares that
vocabulary and not that schedule, and the reason is the loop form rather than
the merge. Its block count is read at runtime through a `lax.while_loop` so a
concrete and a symbolic `n` lower alike, where both byte sponges are
Python-unrolled over static counts; it also has no batch axis, no padding, and a
single truncating read instead of an iterated squeeze. A body parameterized on
which loop it is would be two bodies behind one name, so the two schedules are
siblings in `extension/sponge.py`. The same split holds one layer down: field
trip counts are runtime operands where byte ones are static, so the XLA
envelopes keep both forms too.

The split is load-bearing because the two have no common surface below `digest`.
A byte hash's internal construction differs per family — Merkle–Damgård chains a
midstate, a tree hash combines subtree roots, a sponge absorbs into a state — so
`digest` is the *only* operation that generalizes across them. That is also why a
shared **streaming** surface is deliberately absent from the seam: the
incremental state has a different shape in each construction, so a common
`absorb`/`squeeze` pair would be a fiction that fits one family.

The same intersection rule decides where HMAC's block size lives. HMAC
([`hmac.py`](../../hash_frx/hmac.py)) and HKDF over it
([`hkdf.py`](../../hash_frx/hkdf.py)) are constructions *over* `ByteHash` the
way `Sponge`/`Compression` are over `Permutation`, and FIPS 198-1's `B` is a
parameter only a block-keyed construction can interpret — BLAKE3's keyed mode
is native and has no `B` for HMAC to read. So `Hmac(hash, block_size)` carries
it, and the seam stays `digest` alone; the parallel is `DuplexSponge` owning
its `+`-merge rather than narrowing it into `Permutation`.

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
`squeeze(state, nbytes) -> (state, bytes)` passes it — it describes a sponge
operation, and a transcript hop and a rejection-sampling loop both ride it
without this layer knowing which is calling.

Passing the naming test is necessary and not sufficient. A fused surface is also
drawn from the **intersection** of independent callers rather than their union,
because one shaped against a single real caller and a predicted second fits the
first and bends for the second. The two streaming callers differ in exactly the
place that decides this shape: a Fiat-Shamir transcript hops
`absorb(framing) -> squeeze -> re-absorb`, while a lattice scheme's rejection
sampler absorbs its seed once and then squeezes until enough candidates survive —
FIPS 204's `RejNTTPoly` and `RejBoundedPoly` never re-absorb mid-stream. An
`absorb_then_squeeze(state, bytes_in, nbytes)` would be handed an empty
`bytes_in` on every iteration of the shape that actually repeats, so the half
that makes it a *hop* is dead weight in the loop. What both callers contain is
the squeeze, so the squeeze is what this layer may fuse; the transcript's
re-absorb tail has one caller and stays with it, in the `zorch.sha256_squeeze`
marker owned by the consumer that knows what a transcript is.

The cost of an unfused streaming call is real and worth stating so nobody
re-derives it. Because the padding choice is data-dependent, both
`sha256_stream_absorb` and `sha256_stream_finalize` compress every candidate
block count and select, so an absorb followed by a finalize emits **3**
composites and roughly 818 StableHLO ops of glue at a message inside one block —
a longer absorb adds a candidate and its composite (a 70-byte one is 4).

Two independent costs hide in that count, and only one of them is a marker's to
remove. A marker removes the launch overhead *around* the work; it does not
remove work the schedule speculates: a kernel handed a traced offset still
computes both candidates and selects. On the sponge side, that speculation is
what a whole-block squeezer removes — it holds the offset statically at zero, so
the output is the rate prefix with no `dynamic_slice` and no chain-select over
the carried state.

What it does *not* remove is a permutation, and the earlier claim that it did
was an off-by-one rather than a property of the traced offset. `squeeze(n)`
sized its permutation chain by the block count its *byte* stream needs,
`ceil((rate - 1 + n) / rate)`, but the carried state can only reach
`floor((rate - 1 + n) / rate)` — the trailing permute was live only at
`n ≡ 1 (mod rate)`. Corrected, a whole-block squeeze is one permutation with the
offset still traced. So the structural half is smaller than it looked, and the
lesson stands in a different form: measure the schedule before crediting the
marker, and check that the schedule is the one you think it is.

## Two implementations of one standard

`Sha256` and `HostSha256` ([`sha256.py`](../../hash_frx/sha256.py)) produce the
same FIPS 180-4 bytes and differ only in substrate: one lowers to the device
marker, the other loops `hashlib` on the host. `fusion_path` is what a consumer
branches on, exactly as on the permutation side — the substrate is a value the
hash carries, never a class name a caller has to test.

**The three states are all live, and which hash sits where moves with the
pin.** `FusionPath` separates "lowers to one dedicated kernel" (`DEDICATED`)
from "device and traceable, marker un-routed" (`GENERIC`) from "host loop over
a native library" (`HOST`), because a consumer that cannot tell the middle
state from the last will size a nonce window for the host path against a
device hash. Keccak's byte rows are the standing `GENERIC` case on the CPU leg
(the Keccak arms are GPU-only); BLAKE3's rows were `GENERIC` everywhere until
the emitter shipped and now read `DEDICATED` on cpu and gpu, with Metal keeping
`GENERIC` alive. The gap is a property of the pin and the backend, not of the
design — so the seam names it as a value, and the return type of `digest`
(`Array` against `np.ndarray`) remains the authority `is_traceable` answers
to.

Both stay, and the reason is not only speed. A byte transcript is host-shaped by
construction — a `bytes` buffer with host framing — so a device hash forces a
device-to-host sync on every squeeze; and a proof-of-work grind sizes its nonce
window off `fusion_path`, testing a wide window on a device hash and one nonce
at a time on a host hash. The host implementation is what makes that branch
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

**Every family has host rows, and one body runs them all.** `host_digest`
([`byte_hash.py`](../../hash_frx/byte_hash.py)) is that body — `hash_one` per
message, over the batch row's own buffer rather than a `bytes` copy of it — so a
row is that callable plus its `digest_size`, and a new one is a two-line
`_hash_one` instead of a fourth transcription of the same loop.

**A host row may cost a dependency, and BLAKE3's does.** SHA-256 and the FIPS 202
rows ride `hashlib`; the standard library has no BLAKE3 and will not grow one, so
`HostBlake3` and its keyed and derive-key siblings are built on `blake3`, the
BLAKE3 team's own Rust binding, and it is a declared runtime dependency of this
package. What clears that bar is the property the row exists for: native speed is
the whole reason to choose a host row, and a plain-Python implementation is
slower than the device dispatch it would be standing in for — which is why
`HostKeccak256` is `testonly` and these are not.

The dependency buys a second thing the published vectors cannot: a differential
partner that is not this tree. The binding wraps the reference implementation the
vectors are generated from, so device-against-host agreement at lengths nobody
published is evidence about the implementation rather than about one reading of
the spec applied twice — which is what
[the byte-exactness rule](../reference/conventions.md) asks for when it refuses a
pin against another implementation in this tree.

It also narrows one row. The binding takes a derive-key context as a `str`, so
`HostBlake3DeriveKey` refuses a context that is not valid UTF-8 where
`Blake3DeriveKey` hashes it. The standard names that context UTF-8, so a caller
following it never meets the narrowing.

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

A marker's **cost** is the emitter's business, not this layer's. When one
lowered form compiles or runs worse than another on some backend, the reflex is
a routing constant here — a size threshold, a list of backends — and
[#197](https://github.com/fractalyze/hash-frx/issues/197) is the worked example
of why that is a symptom rather than a fix. The raw-bytes SHA-256 marker cost
2.14x the blocks marker to compile downstream, because the CPU emitter
assembled its padded words once per SIMD lane; fixing that where the cost was
decided ([fractalyze/xla#572](https://github.com/fractalyze/xla/pull/572)) cut
it to 1.33x and retired the proposed threshold entirely. A constant encoding
what some backend charges dates instantly, is invisible to the emitter that
could remove it, and makes this layer carry hardware knowledge it otherwise
does not need.

## Out of scope

Domain separation, the choice between a sponge and a compression function for a
given protocol, and any parameter set tied to a protocol belong to the consumer.
This layer supplies the primitive and its fusion contract, and nothing that would
require it to know what the digest is for.

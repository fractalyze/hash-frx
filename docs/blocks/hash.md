# hash — the symmetric layer

The *why* behind `hash_frx/`. The *what* lives in the code and its tests, which
run on every commit. Open decisions live on the epic issue
[fractalyze/hash-frx#1](https://github.com/fractalyze/hash-frx/issues/1).

## Three layers

The tree is **primitive / extension / adapter**, and the layer a thing belongs to
is decided by what it reads rather than by what it computes.

- A **primitive** is a seam: a fixed-width permutation, or a finished byte hash.
  It reads its own parameters and nothing else.
- An **extension** is a schedule that turns a primitive into a hash —
  Merkle–Damgård, a sponge, a tree. It reads a primitive through its seam, and it
  is written **once per construction**, never once per family.
- An **adapter** is a construction over a *finished* hash. It reads `digest` and
  nothing below it, so it works over any `ByteHash` without naming one.

The layering is the epic's whole claim: a new family should cost a round function
and its constants, not a vertical. Before Phase 1 that claim was false in a way
the tree could show — the Merkle–Damgård padding rule was transcribed **nine
times**, once per family plus the two sponges, and six of the seven MD docstrings
cited SHA-256's copy as "the arrangement" they reproduce.

Two of the three layers are also directories (`extension/`, `adapter/`). The
permutation-side extensions — [`sponge.py`](../../hash_frx/sponge.py),
[`duplex_sponge.py`](../../hash_frx/duplex_sponge.py),
[`compression.py`](../../hash_frx/compression.py) — are still top-level modules.
That is bookkeeping left over from the extraction, not a fourth category.

## Primitives — the two seams

A hash is reached through one of two Protocols, and which one depends on what it
consumes rather than on how it is built.

**`Permutation`** ([`permutation.py`](../../hash_frx/permutation.py)) is a
fixed-width permutation over a single dtype — `width`, `dtype`, `permute`.
The extensions it can enter are [`sponge.py`](../../hash_frx/sponge.py),
[`duplex_sponge.py`](../../hash_frx/duplex_sponge.py) and
[`compression.py`](../../hash_frx/compression.py) — all three read `width`/`dtype`
to size and allocate state and then call `permute`; none of them names a concrete
hash. Poseidon2 is one implementation, classic Poseidon a
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
hash's standard. A consumer reads `digest_size` and calls `digest`. It is a
*finished* hash, so nothing extends it — the layer above it is the
[adapters](#adapters--constructions-over-a-finished-hash). The extensions that
*produce* one run under it, over a compression or round function rather than over
this seam.

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

## Extensions — one schedule per construction

An extension is written **once per construction and shaped against every family
that will enter it**, and the ordering matters: a surface generalized from a
single implementation encodes that implementation's accidents as the family's.
That discipline is what the sponge extension paid for by waiting for a second
byte sponge, and it is what the Merkle–Damgård one paid for by designing against
all seven before writing a line.

### Merkle–Damgård

[`extension/pad.py`](../../hash_frx/extension/pad.py)'s `PadRule` is the padding
rule the seven MD families share, and designing it against all seven first
surfaced four axes a rule derived from SHA-256 alone would have encoded wrong:

- RIPEMD-160 writes its length field **little**-endian, alone among the four
  that write one;
- SHA-512 reserves **16** bytes for a 128-bit field it fills to 64, which moves
  where a message spills only in the 112..119 band and is invisible everywhere
  else;
- **Grøstl's trailer is the block count, not the bit length**, so 0, 1 and 55
  bytes all encode one block;
- BLAKE2b and BLAKE2s have no trailer and no `0x80` at all — HAIFA carries the
  length into the compression instead (`pad.haifa_counter`, RFC 7693 §3.2/§3.3),
  and the empty message is the one case a bare modulo gets wrong.

Each axis is pinned by a vector that fails when the axis is set wrong. The rule
went from nine definitions to one, and `PadRule.tail` is memoized and keyed by
value — SHA-256's rule and SM3's are equal and share an entry, so the array it
returns is read-only.

[`extension/md.py`](../../hash_frx/extension/md.py) holds the schedule around it:
`chain` is the block loop, and `MdStream` is the streaming midstate as an opaque
pytree with leading batch dims. `MdStream.absorb` and `.finalize` were each
written twice — SHA-256's and SHA-512's copies had **zero differing code lines**
after rename normalization — and are now written once.

**The absorb compresses each block once.** Its live block count depends on
`pending_len` by at most one, so both static candidates are computed and selected
between; the two share a *prefix*, so continuing the high one from the low one
costs `min + 1` compressions instead of `min + max`:

| absorb (bytes) | before | after |
|---:|---:|---:|
| 64 | 1 | 1 |
| 65 | 3 | 2 |
| 200 | 7 | 4 |
| 1000 | 31 | 16 |
| 4096 | 64 | 64 |
| 4097 | 129 | 65 |

The unchanged rows are the half that is easy to break — the optimization must not
add work where it does not help — and
[`testing/absorb_cost_test.py`](../../hash_frx/testing/absorb_cost_test.py)
asserts the count for both families. `keccak.streaming.ShakeAbsorb.absorb` had this form
first, with the comment explaining it; the MD absorb re-derived it rather than
reading it off the sponge in the same package, so both files now cross-reference
each other.

**The seven families are two wire ABIs, and both now share the block walk.**
Words-in (`*_merkle_damgard`, padding applied outside the marked region) and
bytes-in (`*_bytes`, padding applied inside) is the split
[`markers.py`](../../hash_frx/markers.py) already draws. `chain` covers the three
words-in families; `masked_chain` underneath it is the walk itself, and the
bytes-in families call it directly.

The earlier measurement here — that a shared bytes-in helper needs five to seven
callbacks to share **three lines out of forty-one to fifty-six** — was about a
whole *decomposition* helper: padding, packing, loop and serialize behind one
call. It stands for that. It does not reach the **walk alone**, which is one
callback for five lines and was extracted once a second family
(`grostl256_bytes`) needed the same runtime-length masking SHA-256 already had.

What made the walk shareable where the decomposition was not is that
`masked_chain` takes the block's **index**, so it emits nothing but the select —
the shape [`tree.chain`](../../hash_frx/extension/tree.py) and
[`sponge.absorb_squeeze`](../../hash_frx/extension/sponge.py) already take, and
for the reason `sponge.py` writes down: a helper handed a *block* must emit that
block first, which fixes one emission order for every caller and silently
excludes whichever builds theirs in the other order. Grøstl slices a flat byte
region where the packed families index a word one; both ride an index callback
unchanged, and neither could ride a block one.

`haifa_counter` is the other residue that was genuinely shared. BLAKE2b/2s read
it off the same `i`, so the callback admits them; they have not been routed
through yet, which is a rollout step rather than a design question.

### Sponge

The byte-side sponge schedule is shared, and it was deliberately not generalized
until it could be shaped against two implementations. Ascon
([`ascon/`](../../hash_frx/ascon)) was the second, so
[`extension/sponge.py`](../../hash_frx/extension/sponge.py) carries the schedule
both run (absorb a block and permute; then read the rate and permute between
reads, with none after the last) and `extension/pad.py` carries `SpongePad`. What
stays family-specific is what actually differs: the state and its packing, the
pad parameters, and the initial state.

The **field** sponge ([`sponge.py`](../../hash_frx/sponge.py)) shares that
vocabulary and not that schedule, and the reason is the loop form rather than the
merge. Its block count is read at runtime through a `lax.while_loop` so a
concrete and a symbolic `n` lower alike, where both byte sponges are
Python-unrolled over static counts; it also has no batch axis, no padding, and a
single truncating read instead of an iterated squeeze. A body parameterized on
which loop it is would be two bodies behind one name, so the two schedules are
siblings in `extension/sponge.py`. The same split holds one layer down: field
trip counts are runtime operands where byte ones are static, so the XLA envelopes
keep both forms too.

### Tree

[`extension/tree.py`](../../hash_frx/extension/tree.py) is BLAKE3's schedule — the
unit ceiling, the intra-chunk chain, and bottom-up pairing. Like the MD chain it
is host arithmetic and control flow: it answers *how many* and *which pairs with
which*, and the caller answers *with what ops*. That is what makes it testable
without a device, which is the layer these rules actually live on.

### Why there is no seam under the compression functions

The extensions above take a plain `compress_block` callback rather than a
`CompressionFunction` Protocol, mirroring what `Permutation` does for the n→n
kind. That seam was proposed, written, and removed again — it had no implementors
and nothing consumed it, and reading the candidates afterwards showed the shape
had been chosen before the families were read.

The honest candidate set is the three that take a per-block counter and flag at
all, and they disagree on every axis a seam would have to fix:

| | counter / final flag | IV arrival | partial block | result width |
|---|---|---|---|---|
| BLAKE3 | `counter [B,2]`, `flags [B]` — **device operands** | one `iv [8]` | `block_len [B]` operand | `[B,16]`, **double** its `[B,8]` state |
| BLAKE2b | `t: int`, `f: bool` — **host scalars** folded to literal XORs | pre-split `iv_lo [8]`, `iv_hi [8]` | none | `[B,16]` = state width |
| BLAKE2s | same as 2b | pre-split `iv_a [4]`, `iv_b [4]` | none | `[B,8]` = state width |

BLAKE3's counter varies per row because chunks batch, so it must be traced;
BLAKE2's message length is static, so its counter folds into the working vector
as a scalar literal. That is one concept on opposite sides of the host/device
boundary, and a seam spanning it has to move one across — pushing BLAKE2's
scalars onto the device **adds operands to a marked region**, which is a wire-ABI
change, and pulling BLAKE3's onto the host is not possible at all. `block_len`
and the 8→16 widening have no BLAKE2 counterpart.

Separately, [`compression.py`](../../hash_frx/compression.py) already spells
`Compression`/`CompressionParams` for the unrelated truncated-permutation n→1
construction, so a second "compression" seam would sit badly beside it.

The decline is recorded in `extension/tree.py`'s own docstring, next to the
callback that took the seam's place, rather than only in review history.

## Adapters — constructions over a finished hash

An adapter reads `digest` and nothing below it. [`adapter/`](../../hash_frx/adapter)
holds them: HMAC and HKDF, MGF1, PBKDF2, the `Xof` and `Dual` holders, the
per-hash `block_size` table, and the named duplex conventions.

The intersection rule that shapes the seams decides where HMAC's block size
lives. HMAC ([`adapter/hmac.py`](../../hash_frx/adapter/hmac.py)) and HKDF over it
([`adapter/hkdf.py`](../../hash_frx/adapter/hkdf.py)) are constructions *over*
`ByteHash` the way `Sponge`/`Compression` are over `Permutation`, and FIPS
198-1's `B` is a parameter only a block-keyed construction can interpret —
BLAKE3's keyed mode is native and has no `B` for HMAC to read. So
`Hmac(hash, block_size)` carries it, and the seam stays `digest` alone; the
parallel is `DuplexSponge` owning its `+`-merge rather than narrowing it into
`Permutation`.

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
`sha256_stream_absorb` and `sha256_stream_finalize` emit every candidate block
count and select between them, so an absorb followed by a finalize emits **3**
composites and roughly 818 StableHLO ops of glue at a message inside one block —
a longer absorb adds a candidate and its composite (a 70-byte one is 4).

The op figure predates the prefix-chaining absorb and is an upper bound now.
*Emitting* both candidates is not the same as *compressing* both: the absorb's
two candidates share a prefix, so it still emits two `chain` regions but folds
`min + 1` blocks rather than `min + max`
([above](#extensions--one-schedule-per-construction)). The finalize is the half
that still speculates in full — its one-block and two-block layouts are separate
shapes, chosen on a `pending_len` only known at runtime.

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

`Sha256` and `HostSha256` ([`sha256/sha256.py`](../../hash_frx/sha256/sha256.py)) produce the
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
(the Keccak arms are GPU-only) and Grøstl-256 is that case inverted, its
emitter being CPU-only; BLAKE3's rows were `GENERIC` everywhere until the
emitter shipped and now read `DEDICATED` on cpu and gpu, with Metal keeping
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
the whole reason to choose a host row, and a plain-Python implementation is two
orders of magnitude off what a `Host*` name promises.

Keccak-256 is where that bar bites, because its plain-Python sponge is not
simply slow: at `B = 1` on a leg without the emitters it beats the device row
several times over, and by two further orders of magnitude the first time a
length is met. Both halves of that win are the substrate rather than the sponge:
the first is the un-routed marker, which an emitter for this leg would close,
and the second is the compilation term, which a marker ABI would. Neither is a
reason to ship a plain-Python row — which is why Keccak-256 has no shipped host
row, and why paying a dependency to make one genuinely native was judged not
worth it either.

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

## Which hashes get a host row

A `Host*` row ships when two conditions hold together:

1. the host path is faster than the device path **at the batch shape the
   consumer lives at**, and
2. a **host-shaped consumer** exists — one that is not tracing, and needs the
   result bytes for ordinary Python control flow.

Both are load-bearing, and the set they carve out is the whole reason the
`Host*` names look the way they do: the `hashlib` families and BLAKE3 ship
because both hold, and Keccak-256, RIPEMD-160, Grøstl-256 and Ascon-Hash256
have no shipped host row because the first fails.

Condition 1 reads as a claim about kernel speed, and that is its smaller half
wherever the message length is part of the *input shape*. `digest` takes
`uint8[B, L]`, so for most families it is, and the block count and the pad are
static by construction —
[`keccak/sponge.py`](../../hash_frx/keccak/sponge.py) states the trade in one
line ("every loop bound here is static … that is what lets `digest` take a
tracer"), and `extension/pad.py`'s `PadRule` builds the pad as a host constant
*from* the static length. So an eager caller whose messages are not all one
length pays a **compilation per length**, not a kernel per call — and on the CPU
backend at `B = 1` that compilation is four to five orders of magnitude above the
warm call it enables, which is itself an order above the `hashlib` row. The host
row's real saving is the compile, and it dwarfs the dispatch saving that the
batch-shape table above measures.

**A dedicated emitter does not shrink that term; it grows it.** A routed marker
compiles a fresh length roughly an order of magnitude slower than an un-routed
one, because the emitter's own MLIR and LLVM codegen is the cost — and it buys
back roughly two orders on the warm call. That is the trade a dedicated kernel
makes, and it is a good one for a traced caller and a bad one for a caller that
meets each length once. So "ship the emitter and the host row stops paying for
itself" is wrong twice: the emitter does not touch this term, and it raises it.

The figures move with the pin — a wheel that gains or loses an emitter moves
them by an order of magnitude, which is why they are stated as ratios here and
measured fresh when they matter.

Naming the causes is what keeps the condition honest, because it also says what
retires it: a marker ABI taking the length as an operand, with the block loop
inside the emitter, compiles once per buffer width rather than once per length.
SHA-256 has one on **both** cpu and gpu — its runtime-length digest ABI
([`sha256/sha256.py`](../../hash_frx/sha256/sha256.py), `_BYTES_EMITTER_BACKENDS`) — so a
message widened to a shared buffer width is hashed at its own length by a kernel
every other length reuses. That ABI is now the only whole-message form: the
static-tail arm it once shared the marker name with was retired once the operand
form reached every backend that has an emitter at all.

Grøstl-256 is the second family on it, cpu only
([`grostl/grostl.py`](../../hash_frx/grostl/grostl.py), `_EMITTER_BACKENDS`), and it
carries the form under its EXISTING marker name rather than a sibling: the two
operand forms are told apart by operand 4 being disjoint in element type and
rank, so no second name and no `composite.version` bump were needed.
Where that holds, condition 1's larger half is gone and the host row rests on
dispatch alone, which is a far weaker justification than the paragraphs above
describe.

Everywhere else the compile-per-length term stands as stated: the five
remaining tail-operand families, and either of the two above on a backend whose
plugin carries no emitter for that ABI. So condition 1 is a question about a (hash, backend) pair rather than about
the package, and a host row retired on the strength of one cell would be a
regression in the others.

`Permutation` has **no host category at all**, and that is a different
statement from a byte hash having no shipped host row. A permutation fails both
conditions rather than one: there is no native standard to wrap, and it runs
inside a circuit, where a tracer's bytes cannot be read. `HostPoseidon` is not
missing, it is categorically absent — so the absence needs no per-hash
justification the way Keccak-256's does.

**Why not emit a host-side composite from XLA instead.** A `ByteHash` is a
finished product, so the question recurs: why keep `hashlib` at all rather than
mark a host-side region and delete the device/host split? It is a category
error. A composite exists only inside HLO, and reaching HLO means tracing —
while the moment a host row is wanted is exactly the moment there is no HLO,
because the caller is not tracing and needs concrete bytes (a KAT comparison, a
sequential byte transcript, assembling a Merkle tree). Handing `np.ndarray` to
a marked region leaves nothing to wrap. Running the *device* row on the CPU
backend is possible and gives the right bytes; it is condition 1 restated, at
the two costs measured above.

**Dropping host rows wholesale is a seam change, not a third option.** It is
worth naming because it keeps being re-proposed as though it were cheap. The
measurements above are what rule it out — every sequential caller pays the two
costs, and the host-shaped consumers of condition 2 exist today. What it would
*entail* is recorded separately, because a consequence is not an argument:
`FusionPath.HOST` goes dead, `is_traceable` becomes trivially true for every
value so [`adapter/pbkdf2.py`](../../hash_frx/adapter/pbkdf2.py)'s guard becomes unreachable,
and `DEDICATED`/`GENERIC` is a boolean — the shape `FusionPath` exists to
replace. That collapse is correct bookkeeping *if* host rows genuinely go; it
is not itself the reason they stay.

## Names say the construction they implement

An identifier names the construction it implements — `sha256_merkle_damgard` —
and shape words like "chain" stay in prose. A chain is what a Merkle–Damgård
fold looks like, not what it is, and a name that describes the picture stops
matching the moment a second construction has the same picture.

The rule cuts the other way too: a name may not claim a construction it does not
implement, however well the picture fits. `SpongeChaining.DIGEST_IN_CAPACITY` was
spelled `SpongeType.MERKLE_DAMGARD`, and what it selects is a discipline *inside*
the sponge — whether the prior digest is written into capacity — not the
Merkle–Damgård domain extension. The borrowed name read as though a sponge could
be Merkle–Damgård, and it was cited on the epic
([#1](https://github.com/fractalyze/hash-frx/issues/1)) as this rule's own
precedent while being its clearest violation.

A corrected name over a frozen value is the normal end state, not an
inconsistency to tidy away: that member's *value* is a marker attribute, so it
renames on the wire's schedule rather than the code's
([`markers.py`](../../hash_frx/markers.py)).

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

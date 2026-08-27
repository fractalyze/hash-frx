# hash-frx docs

Reference indexed by what you're trying to do. For the project overview, install,
and the dev quick-start, see [`../README.md`](../README.md).

The authoritative description of a primitive is its module docstring — every
seam, construction, and implementation states its own contract and the reasoning
behind its shape. This hub is the index into them, plus the one contract that
spans all of them: [fusion](#the-fusion-contract).

The tree mirrors the layering, which is **primitive / extension / adapter**:

- a **primitive** is one of the two seams — a fixed-width permutation
  ([`permutation.py`](../hash_frx/permutation.py)) or a finished byte hash
  ([`byte_hash.py`](../hash_frx/byte_hash.py));
- an **extension** is a schedule that turns a primitive into a hash —
  Merkle-Damgard, a sponge, a tree — and it is written once, not once per family;
- an **adapter** is a construction over a *finished* hash, which reads `digest`
  and nothing below it ([`adapter/`](../hash_frx/adapter)).

`hash_frx/<family>/` (or a flat `hash_frx/<family>.py` for the ones that need no
package) is one implementation family: its parameter surface, its round function,
its constants. Every package carries a `testing/` subdir with its tests and the
reusable helpers those tests share.

The directory layout states two of the three layers and not the third. The
byte-side extensions live in [`extension/`](../hash_frx/extension) and the
adapters in [`adapter/`](../hash_frx/adapter), but the permutation-side
constructions — [`sponge.py`](../hash_frx/sponge.py),
[`duplex_sponge.py`](../hash_frx/duplex_sponge.py) and
[`compression.py`](../hash_frx/compression.py) — are still top-level modules. They
are extensions by role; the tables below group them by role rather than by
directory, and that mismatch is bookkeeping rather than a design boundary.

## Working here

| Question                                                                                   | Where                                             |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| The rules a hash primitive is written to — fusion authoring, the seam and pin requirements, byte-exactness, and the assertions that must bite | [`reference/conventions.md`](reference/conventions.md) |
| Why the layer is shaped this way — the three layers, why each schedule is written once, what may live here at all, why every row is a device row and what that costs, fusion as a design property | [`blocks/hash.md`](blocks/hash.md)                 |
| Getting a dev loop — backend selection, the two test legs, the lowering gate, the CUDA version trap, an unreleased XLA, the compile cache | [`reference/development.md`](reference/development.md) |
| Depending on this package from another repo — the import form, the Bazel dep that goes with it, and what to do about a name the root does not export | [`reference/consuming.md`](reference/consuming.md) |

## Primitives — the seams a consumer codes against

| Question                                                                                   | Where                                             |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| Hashing with a fixed-width permutation, algebraic or bit-oriented, without naming one       | [`permutation.py`](../hash_frx/permutation.py)     |
| Hashing raw bytes byte-identically to a published standard, without naming one              | [`byte_hash.py`](../hash_frx/byte_hash.py)         |

## Extensions — the schedules that turn a primitive into a hash

A schedule is written once here, not once per family. Each row names the
primitive it sits over; `extension/pad.py` sits under the others rather than over
a primitive, since padding is host arithmetic on the message.

| Question                                                                                   | Where                                               |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| Merkle–Damgård over a compression function — the block chain the seven MD families share, the padded-region helper, and the streaming midstate over it | [`extension/md.py`](../hash_frx/extension/md.py) |
| How a message becomes whole blocks — `PadRule` for the Merkle–Damgård families, `SpongePad` for the sponges | [`extension/pad.py`](../hash_frx/extension/pad.py) |
| The sponge schedules themselves — a static count unrolled, a runtime count in a `while`, a large static count in a `scan`, and why the block contract is what separates them | [`extension/sponge.py`](../hash_frx/extension/sponge.py) |
| The tree schedule over a compression function — the unit ceiling, the intra-chunk chain, and why bottom-up pairing is the spec's own tree | [`extension/tree.py`](../hash_frx/extension/tree.py) |
| One-shot sponge hash over a `Permutation` — either chaining rule, selected per call | [`sponge.py`](../hash_frx/sponge.py)                 |
| Interleaved absorb/squeeze with add-mode absorb over a `Permutation`, for a classic Fiat-Shamir prover | [`duplex_sponge.py`](../hash_frx/duplex_sponge.py)   |
| n-to-1 truncated-permutation compression over a `Permutation`, for a hash tree | [`compression.py`](../hash_frx/compression.py)       |

There is no seam under the compression functions the Merkle–Damgård and tree
rows run over — `compress_block` is a plain callback. Why that is a decision rather than a gap is
in [`extension/tree.py`](../hash_frx/extension/tree.py)'s docstring and in
[`blocks/hash.md`](blocks/hash.md#extensions--one-schedule-per-construction).

## Adapters — constructions over a finished hash

An adapter reads `digest` and nothing below it, so it works over any `ByteHash`
without naming one.

| Question                                                                                   | Where                                               |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| HMAC — the FIPS 198-1 keyed MAC, one class parameterized by hash and block size             | [`adapter/hmac.py`](../hash_frx/adapter/hmac.py)                     |
| Which hashes have an HMAC block size, and why BLAKE3, keyed BLAKE2 and Ascon deliberately do not | [`adapter/block_size.py`](../hash_frx/adapter/block_size.py) |
| Holding a variable-output family before an output length is chosen, and what fills that slot | [`adapter/xof.py`](../hash_frx/adapter/xof.py) |
| HKDF — RFC 5869 extract-then-expand, the KDF the composition standards name                  | [`adapter/hkdf.py`](../hash_frx/adapter/hkdf.py)                     |
| MGF1 — RFC 8017's mask generation, a hash stretched to any length by a counter suffix (RSA-OAEP / PSS) | [`adapter/mgf1.py`](../hash_frx/adapter/mgf1.py) |
| PBKDF2 — the RFC 8018 iterated KDF as a traced c-loop, with the ipad/opad midstate fast path (BIP-39's seed derivation) | [`adapter/pbkdf2.py`](../hash_frx/adapter/pbkdf2.py)                 |
| Which released duplex a sponge reproduces — `ARK_0_3` (including its spill-permute bug) and `ARK_0_5` | [`adapter/duplex.py`](../hash_frx/adapter/duplex.py) |

## Implementations

| Question                                                                                   | Where                                               |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| Poseidon2 — permutation, parameter surface, and its two linear layers                       | [`poseidon2/`](../hash_frx/poseidon2)                |
| Classic Poseidon — the naive Hades schedule and the optimized-sparse refactor of it          | [`poseidon/`](../hash_frx/poseidon)                  |
| Vision — the binary-field Marvellous permutation (Vision Mark-32 over the GF(2^32) tower)    | [`vision/`](../hash_frx/vision)                      |
| SHA-256 — batched digest, incremental midstate, and the `ByteHash` row        | [`sha256/sha256.py`](../hash_frx/sha256/sha256.py)                 |
| SHA-512 — the 64-bit SHA-2 sibling over uint32 half pairs: batched digest, incremental midstate, the `ByteHash` row, and the truncated variants SHA-384 and SHA-512/256 as IV rows on the same marker | [`sha512/sha512.py`](../hash_frx/sha512/sha512.py)                 |
| Keccak-f[1600] — the permutation under SHA-3, SHAKE and Keccak-256, over uint32 lane halves  | [`keccak/`](../hash_frx/keccak)                      |
| BLAKE3 — the chunk tree, and hash / keyed / derive-key as `ByteHash` rows at any output length over one compression function | [`blake3/`](../hash_frx/blake3), [`blake3/modes.py`](../hash_frx/blake3/modes.py), [`blake3/rows.py`](../hash_frx/blake3/rows.py) |
| BLAKE2b — the HAIFA `ByteHash` over 64-bit half pairs, behind its digest marker | [`blake2b/`](../hash_frx/blake2b) |
| BLAKE2 keyed, salted and personalized hashing — RFC 7693 §2.8's parameter block, and why keying rides the existing marker rather than a new one | [`blake2_params.py`](../hash_frx/blake2_params.py), [`blake2b/blake2b.py`](../hash_frx/blake2b/blake2b.py) |
| BLAKE2s — the 32-bit RFC 7693 sibling at native uint32, on its own digest marker | [`blake2s/blake2s.py`](../hash_frx/blake2s/blake2s.py) |
| SM3 — the GB/T 32905 ShangMi hash, SHA-256's structural cousin, on its own digest marker | [`sm3/sm3.py`](../hash_frx/sm3/sm3.py) |
| Grøstl-256 — the AES-round Merkle–Damgård `ByteHash` over GF(2^8), with a bitsliced S-box | [`grostl/`](../hash_frx/grostl) |
| Ascon-Hash256, Ascon-XOF128 and Ascon-CXOF128 — the NIST SP 800-232 lightweight-standard sponge `ByteHash` rows over uint32 word halves, plus Ascon-p[12] as a `Permutation` | [`ascon/`](../hash_frx/ascon) |
| RIPEMD-160 — the little-endian Merkle–Damgård `ByteHash` (Bitcoin HASH160's second half) | [`ripemd160/ripemd160.py`](../hash_frx/ripemd160/ripemd160.py) |
| The four SHA-3 rows, SHAKE128, SHAKE256 and Keccak-256 — the byte hashes over one sponge, and that sponge (why it is not `sponge.py`) | [`keccak/byte_hashes.py`](../hash_frx/keccak/byte_hashes.py), [`keccak/sponge.py`](../hash_frx/keccak/sponge.py) |
| cSHAKE128 / cSHAKE256 — SHAKE customized by a function name and a customization string, and why empty `N` and `S` is a branch back to plain SHAKE | [`keccak/cshake.py`](../hash_frx/keccak/cshake.py) |
| KMAC128 / KMAC256 and their XOF forms — the NIST Keccak MAC over cSHAKE, why it is not HMAC, and what `right_encode(L)` separates | [`keccak/kmac.py`](../hash_frx/keccak/kmac.py) |
| TupleHash128 / 256 and their XOF forms — unambiguous hashing of a *sequence* of strings, and why it is deliberately not a `ByteHash` | [`keccak/tuple_hash.py`](../hash_frx/keccak/tuple_hash.py) |
| SP 800-185's length encodings (`left_encode`, `bytepad`, …) — the prefixes every SHA-3 derived function is built from, and which of them counts bits | [`keccak/encodings.py`](../hash_frx/keccak/encodings.py) |

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
pre-namespace spellings until they retire. Whether one tolerates **control
flow** is likewise its ABI's to say rather than the contract's:
`hash_frx.digest.field_sponge` carries a `stablehlo.while` for its runtime absorb
length ([`sponge.py`](../hash_frx/sponge.py)), where `keccak_sponge` does not. A
loop *around* a marked region is always fine — only a loop inside a
*generically* marked body is the bug `CLAUDE.md` names.

**Operation-named** — `hash_frx.permute`, a single segment with no kind prefix.
It names the OPERATION and carries WHICH primitive runs as the `permutation`
composite attribute, so a new permutation routes by registering a primitive
rather than by minting a marker name on both sides of the seam. All six
`hash_frx.perm.*` spellings above are retiring into it: the recognizer accepts
both (fractalyze/xla#616), and hash-frx keeps emitting the old ones until the
`frx>=` floor carries that recognizer — emitting a name the pinned plugin does
not know costs the fusion silently rather than failing, so the flip follows the
pin instead of leading it. The other operation names the relayering adds
(`digest`, `digest_bytes`, `stream_*`, `duplex`) arrive with the envelopes that
consume them.

The Merkle–Damgård operation names are one per **wire-ABI schema**, not one per
family. There are three, and a fourth schema that stays primitive-named:

| name | operands | the message is |
| --- | --- | --- |
| `hash_frx.digest` | `[h0, consts…, blocks W[…, nblocks, w]]` | already padded and packed by the producer |
| `hash_frx.digest_bytes` | `[h0, consts…, msg u8[…, L], tail u8[P]]` | unpadded: the padding runs inside the region, and `L` is a shape |
| `hash_frx.stream_finalize` | `[h, consts…, pending u8[block], counts s32[2], extras u8[…, E]]` | a stream POSITION: `pending[:counts[0]] ‖ extras` |

The fourth is the **runtime-length** raw-bytes form, which carries the length as
a scalar operand and synthesizes its padding from it — so its block count is a
runtime value rather than a shape property. `hash_frx.digest.sha256_bytes` and
`hash_frx.digest.grostl256` are both in it and both keep their per-family names,
the SHA-256 one because its recognizer dispatches on operands rather than on the
name, which is what lets two ABIs ride under it.

Note `digest_bytes` and that fourth schema are both "raw bytes" — what separates
them is **static versus runtime length**, not padded versus unpadded. Wiring a
runtime-length family behind `digest_bytes` would hand it an envelope with no
length operand.

What makes `stream_finalize` its own schema is that `counts` —
`[pending_len, total_len]` — is a **runtime operand**. The strengthening field
claims `total_len + E`, the whole stream, where the fold covers only
`pending_len + E`: Merkle–Damgård's resume property, and the reason a resumed
digest equals a one-shot one over the same bytes. `markers.py` carries why the
position rides as an operand at all, and why the name is flat.

`E` is bounded above at `block_size - reserve` — `pending_len` is runtime data,
so only a tail that fits the two-block layout at *every* reachable offset is
representable. `E = 0` is legal here and is exercised; the pinned recognizer
declines it rather than fusing it, so such a call inlines. Right bytes, no
kernel — the ordinary cost of a marker the plugin will not take.

Two of them wrap a whole hash rather than one primitive — `hash_frx.digest.field_sponge`
over the field sponge, `hash_frx.digest.keccak_sponge` over the byte one — and both are
assembled by `fused_region_over`, which rebuilds the permute from the operand
layout `fused_region_spec` hands out. That is what keeps a construction from
naming the permutation's constants, and what makes a whole absorb one region
instead of a marked permute per block with the glue between them left outside.

The markers wait for the toolchain in two different ways, because the cost of
being early is not the same for both. `hash_frx.digest.blake3`,
`hash_frx.digest.sha256`, `hash_frx.digest.grostl256` and
`hash_frx.digest.ascon_hash256` are emitted
whether or not the pinned plugin recognizes the name: an
unrecognized *name* only inlines, so being early costs the fusion and nothing
else, and the hash reports `fusion_path = GENERIC` while carrying its marker.
`hash_frx.perm.keccak_f` is emitted only where the plugin ships its emitter — off
the pin *and* the backend — and the `frx>=` floor in `pyproject.toml` is what
holds the pin half true. Both legs carry the Keccak arms from the wheel that
first shipped the CPU sponge emitter; `hash_frx.perm.poseidon_sparse` is the
family still on one leg. That switch has to
track the pin rather than be left optimistic, because a permutation's marker
also decides whether a `Sponge` over it wraps its whole hash as
`hash_frx.digest.field_sponge` — and that marker carries a `permutation` discriminator
a plugin without the arm rejects outright, which is a failed compile rather
than a lost kernel.

The five Merkle–Damgård families have left the first group entirely. SHA-512 and
SM3 ride `hash_frx.digest`, RIPEMD-160, BLAKE2b and BLAKE2s ride
`hash_frx.digest_bytes`, and all five are recognized on both legs, so they report
`fusion_path = DEDICATED` and no longer name themselves on the wire. What is
still waiting is Ascon and Vision, and neither is an MD family — a sponge pads
differently, so it needs an envelope of its own rather than an entry in this one.

`hash_frx.digest.sha256_bytes` is gated the same way and for a third reason, and
`hash_frx.digest.grostl256` now carries the same gate for the same reason. Each
takes the message length as an operand rather than as part of the message shape,
so one compiled kernel serves every length its buffer can hold — but its
decomposition has to derive a data-dependent block count from that operand,
which in plain HLO means speculating every block the buffer could need.
Inlining it is therefore *slower* than the static-tail form it replaced, where
an unrecognized name normally costs only the fusion. Being early costs
performance rather than nothing, so the switch tracks the pin and the backend —
and for Grøstl, which kept no fallback arm, it is the `frx>=` floor rather than a
routing switch that refuses the wheels below it.

The pin and the backend are not the only things that decide routing. A third
axis is the **parameterization's own values**: an emitter's marker carries its
constants as attributes, and a recognizer declines a set whose numbers it cannot
serve — so a permutation can be on a pin and a backend that both ship the
emitter, and still belong on the generic marker. All three field permutations answer it, each in a
`_select_fused_region_name` alongside the pin question: `Poseidon2` asks whether
its external matrix is M4-block-structured, `SparsePoseidon` whether its four
matrices fit the int64 attributes it bit-casts them into, and classic `Poseidon`
whether its MDS sits in the range its emitter's add-chain can apply. The three
arrived separately, which is the tell that an emitter whose attributes carry
values rather than only shape raises this question by construction — and getting
it wrong there is a failed compile rather than a lost kernel.

A permutation advertises which path it is on through `fusion_path`
(`hash_frx.fusion.FusionPath`: `DEDICATED` or `GENERIC`), and hands out its
operand layout through
`fused_region_spec` — that pair is
what lets a consumer wrap a whole computation as one region without naming the
hash underneath it.

**Marker names are a wire ABI.** The rewriters match by name, and an unrecognized
name does not error: the composite inlines and fusion is silently lost. A
contract change therefore rides `composite.version` rather than a rename. The
operand layout is equally part of that ABI, which is what keeps a region's
constants from being written as closed-over values — and, where two operand
forms are disjoint in element type *and* rank, is what lets them share one name
instead of needing a second: the pinned recognizer claims two under
`hash_frx.digest.sha256_bytes`.

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

Route by layer, and the row follows.

- **A new hash family** implements a seam and adds a row under Implementations.
  It needs a round function, its constants and its parameter surface — and *not*
  a schedule, a padding rule or a routing step, which is what the extensions
  above already hold. Every family is a `hash_frx/<family>/` package, whatever
  it currently spans, and its tests live in `hash_frx/<family>/testing/` — the
  shared `hash_frx/testing/` holds only the suites that cross families. A rule
  keyed on module count instead put the same standard in two shapes at two
  widths, and made a family's second module a rename rather than a new file.
- **A new schedule** — a construction that turns a primitive into a hash — adds a
  row under Extensions. Shape it against every family that would enter it before
  writing it, never against the first one; the reasoning is in
  [`blocks/hash.md`](blocks/hash.md#extensions--one-schedule-per-construction).
- **A new construction over a finished hash** adds a row under Adapters and lands
  in `hash_frx/adapter/`.

A design note that outgrows its module docstring lands in `blocks/`, and a
convention that a reviewer has to point at more than once lands in `reference/`.

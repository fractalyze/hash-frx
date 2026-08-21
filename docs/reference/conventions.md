# Coding conventions

> Code, symbols, and file paths are English.

This page carries only what is specific to writing hash primitives. The rules
every FRX consumer shares — `@jit` placement, `for` vs `lax.scan` vs `vmap`,
pytree registration mechanics, type annotations, the `testing/` layout, the
comment rules — are not repeated here. They follow from FRX and XLA semantics
rather than from what a repo computes, so a copy per repo is how they drift
apart; read them in
[`zorch`'s page](https://github.com/fractalyze/zorch/blob/main/docs/reference/conventions.md),
which states them in full.

## A marked body is authored to lower, not to read

The [fusion contract](../README.md#the-fusion-contract) says what the unit is.
These are the rules that keep a body inside it, and each one trades the idiom a
Python reviewer would prefer for a lowering that stays one kernel.

- **A round loop is an unrolled Python `for`.** Round counts are static and
  small, and `lax.scan` lowers to a `stablehlo.while` — a control-flow boundary
  the region cannot contain.
- **A linear layer is an unrolled sum of column-scaled lanes**
  ([`linear.py`](../../hash_frx/linear.py)), never `fnp.dot` or `fnp.sum` (a
  reduction, which is the `kInput` fusion boundary) and never a dynamic index (a
  gather).
- **`lax` primitives rather than the `fnp` wrapper** where the wrapper carries an
  internal `jit` — `lax.select`, not `fnp.where`. The wrapper lowers to a call
  inside the body, which the single-kernel rewriter rejects. A nested `@jit` is
  the same failure, so a marked body sits under exactly one `@jit` boundary.
- **A constant a name-routed emitter reads is an explicit operand.** The
  emitter's operand ABI is positional and fixed: it reads the round-constant
  table at the index its recognizer declares, so a constant left inside the
  decomposition is one the emitter cannot find, and a constant threaded in the
  wrong order is one it misreads. Two remedies, and every marked body needs one
  of them per constant: **thread it as an explicit operand**
  (`sha256.compress`'s round-constant table, `blake3.Mode.iv`), or **compute it
  on device** — `fnp.arange`/`iota` is an index the kernel counts rather than a
  value the program carries (`blake3._counters`).

  This rule used to carry a second, stronger reason that no longer holds:
  `lax.composite` once lifted *every* host-materialised array into an operand
  ahead of the declared ones, one per call site, so the ABI became a function of
  the input shape whether the author threaded the constant or not — BLAKE3
  measured 3 operands at a 64-byte message and 23 at a 2049-byte one. frx#218
  stopped the lift; a closed-over array now stays inline in the decomposition.
  The threading rule survives on the emitter-ABI reason alone, and
  `poseidon/linear.py` and `poseidon/sparse.py` are where the new behavior is
  relied on.

None of this is enforced by the type system, and none of it changes an output
byte. [Testing](#testing) is where the enforcement is.

## A type per named member, a parameter per choice within one

**Where a standard names its members, each name is a type.** FIPS 202 names
SHA3-256, SHA3-512, SHAKE128, SHAKE256; the BLAKE3 spec names `hash`,
`keyed_hash`, `derive_key`. Those become `Sha3_256` / `Sha3_512` / `Shake128` /
`Shake256` and `Blake3` / `Blake3Keyed` / `Blake3DeriveKey`, sharing a body and
differing by the constants the standard attaches to the name — a rate, a
domain-separation byte, a mode flag. What a caller then chooses *within* one named member rides as a
constructor parameter: an output length, a key, a context string.

A member can carry both, and BLAKE3's keyed row does. The standard fixes its
mode flag and fixes that its tree opens from a caller-supplied key rather than
from the IV; the caller supplies the key's 32 bytes. **The test is whether a
caller could reach a different member by changing it**, not whether a value
appears in it. No key turns `Blake3Keyed` into `Blake3`, so the flag splits the
type and the key rides inside it. Folding that flag into a parameter would give
one class a knob that turns it into a different standard, which is how a
consumer ends up choosing at runtime a hash it should have named.

**Where a standard specifies a generator over members instead of naming them,
the parameterization is one value-compared object.** Poseidon2 is a construction
over a field, a width, an α and a round-constant schedule, with no enumerated
list to make types from — so `Poseidon2(params)` takes the whole
parameterization as a parameter and `Params` carries the value equality. Writing
a type per parameterization there would be a class per instantiation of an
infinite family.

Either way the split decides only what is a *type*. Everything left over is a
parameter, and **two instances differing in any parameter are two hashes** — the
next section is why that has to reach `__eq__`.

**Aux-safety is not the reason.** A rate folded into `__eq__` would be as
re-trace-safe as a rate baked into a type, so [pytree
registration](#pytree-registration) does not decide this and cannot: it says every
parameter that survives the split must reach `__eq__`/`__hash__`, which is what
stops a consumer holding one key from being served the trace built for another.

**A default output length is permitted where the standard names one, and refused
otherwise.** BLAKE3 names 32 bytes for each of its three modes, so every row
takes it; an XOF with no such size has none, and picking one for the caller is
how a scheme ends up silently truncating.

## Pytree registration

A permutation or a hash is not itself a pytree; it rides in one as **aux**
(`meta_fields`), because a consumer threads it through `jit` inside a transcript
or a Merkle committer. Aux compares by value, so every implementation defines
`__eq__`/`__hash__` over its **full parameter surface** — including anything that
changes what `permute` lowers to, which is why
[`sparse.py`](../../hash_frx/poseidon/sparse.py) folds its marker name into the
key.

The `Permutation` and `ByteHash` Protocols require this and cannot enforce it: a
Protocol carries no implementation. Getting it wrong does not error — identity
equality makes every freshly built instance a new jit cache key and re-traces the
enclosing zone per call, so it surfaces as a slow caller and never as a failure.
`assert_single_trace` ([`testing/jit_cache.py`](../../hash_frx/testing/jit_cache.py))
is what pins it.

The dataclass-derived `__eq__` is not usable for a parameter record here: `==` on
its `Array` fields is element-wise.

## Seam conformance pins

Every implementation module ends with a one-line pin, so signature drift fails
pre-commit at the implementation rather than at a consumer call site — or never,
which is the normal case in a library whose implementations have no in-tree
consumer:

```python
if TYPE_CHECKING:
    _: type[Permutation] = Poseidon2
```

A module shipping two implementations of one seam names each pin, since mypy
rejects re-annotating `_` — [`sha256.py`](../../hash_frx/sha256.py) pins `Sha256`
and `HostSha256` separately.

The pin bites only because the full-annotation rule is enforced
(`disallow_untyped_defs` in [`pyproject.toml`](../../pyproject.toml)): against an
unannotated implementation it compiles and proves nothing.

Write the pin by drifting the implementation until mypy fails **at the pin**.
There is less margin here than it looks — mypy cannot see through FRX's stubs, so
`Array` collapses to `Any` and the pin checks little more than names and arity. A
`@runtime_checkable` `assertIsInstance` is complementary rather than a substitute:
it checks member presence on a live object, never signatures.

## Byte-exactness is the gate

A byte hash reproduces its standard exactly; that is the seam's whole promise. So
it is pinned against the published vectors of that standard, never against
another implementation in this tree — two implementations of one standard agree
with each other perfectly while both being wrong.

A magic constant, an initial state, or a padding rule carries the document and
section it comes from (`# FIPS 180-4 §5.3.3`, not `# round constants`). Hash code
is easy to write plausibly and wrongly, and the citation is what lets a reviewer
check it rather than agree with it.

## Testing

A seam test uses a shape-correct double, not a real hash: it tests the Protocol's
shape and must keep running before any implementation of that seam exists. The
differential cases that pin an implementation against its standard live with that
implementation. A double that computes real digests duplicates an implementation
and implies a fidelity the seam test does not check.

Three assertions here exist because the failure they catch is silent — right
bytes, wrong lowering. **Each is written so that it bites**, and the proof that
it bites is a test:

- **Fusion shape.** `assert_fusion_ready`
  ([`testing/fusion_ready.py`](../../hash_frx/testing/fusion_ready.py)) whitelists
  the ops a lowered body may contain, and is paired with a negative case
  asserting it *rejects* the plain `m @ v` the normal form exists to avoid
  ([`poseidon2/testing/linear_test.py`](../../hash_frx/poseidon2/testing/linear_test.py)).
  Rewriting an unrolled fold to `fnp.sum` keeps every value test green while
  splitting the kernel.
- **Marker survival.** A marker is checked in the lowered MLIR text — its name,
  its attributes, and its operand order — and by counting composites rather than
  finding one. Matching output values proves nothing about a marker: an
  unrecognized one inlines and still computes the right answer.
- **Trace count.** `assert_single_trace` drives a sequence of calls and asserts
  the module-level jit zone gains no cache entries after the first.

Tests draw field elements as the Montgomery-form `zk_dtypes`
([`testing/random_field.py`](../../hash_frx/testing/random_field.py)): Montgomery
is the production encoding the GPU kernels compute in, so a test in the canonical
dtypes exercises an arithmetic path nothing ships. The `mont-test-dtypes`
pre-commit hook rejects a bare canonical dtype in a `*_test.py`; a test genuinely
*about* the canonical encoding opts out by marking that line
`# canonical-encoding test`.

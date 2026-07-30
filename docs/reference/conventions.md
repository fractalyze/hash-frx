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
- **Round constants ride as explicit operands.** A constant merely closed over by
  the decomposition can be lifted out of the composite, which breaks the
  emitter's operand ABI.

None of this is enforced by the type system, and none of it changes an output
byte. [Testing](#testing) is where the enforcement is.

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

# Consuming this package

> For working *on* hash-frx, see [`conventions.md`](conventions.md). This page is
> for a repo that depends on it.

## Take the names off the package root

```python
from hash_frx import Sha256, Shake128, SHAKE128_RATE
```

not `from hash_frx.sha256 import Sha256`, and not
`from hash_frx.keccak.byte_hashes import Shake128`.

[`__init__.py`](../../hash_frx/__init__.py) re-exports every public name from the
module that defines it, and that indirection is the whole point: the modules
underneath move as the package is re-layered into primitive / extension /
adapter, and an import that spells the layout breaks on every move. The adapters
sit under [`adapter/`](../../hash_frx/adapter) for that reason, and the
rearrangement cost each consumer that had spelled the layout a broken build.

The re-exports are lazy (PEP 562), so reaching a name imports exactly the module
that defines it — the same module the layout import would have named, at the same
moment. The root spelling buys insulation, not import time.

## Under bzlmod, the dep is the whole package

A consumer that pins hash-frx with `git_override` deps
`@hash_frx//hash_frx`, never a narrow label like `@hash_frx//hash_frx:sha256`.

The import form and the dep are one decision rather than two.
`hash_frx/__init__.py` ships only in the `//hash_frx:hash_frx` target — the
`//hash_frx:package_init` target that also carries it is
`//hash_frx:__subpackages__`-visible and cannot be reached from outside. With a
narrow dep and `--incompatible_default_to_explicit_init_py`, `hash_frx` resolves
as a PEP 420 namespace package with no `__getattr__`, so `from hash_frx import X`
fails at **runtime**, with analysis already green.

The whole-package dep is wider than a narrow label — every family, and blake3's
extension module with them. That is the price of the insulation, and it is
bounded: the runfiles delta measures under a percent on a tree that already
carries frx.

A consumer that takes hash-frx as a **wheel** — `requirement("hash_frx")` — has
no such constraint, because a wheel always ships `__init__.py`. Only the import
rule above applies there.

## A missing name is an upstream fix

When a name a consumer needs is absent from
[`_EXPORTS`](../../hash_frx/__init__.py), the fix is to export it here rather
than to keep a carve-out downstream. The SHAKE and SHA-3 rates are the worked
example: a consumer sizing a sponge budget in whole blocks has no other source
for them, so they are exported, and a scheme repo needs no layout import to
reach one.

Not everything in a module is a candidate. A rate is a parameter a caller must
know to size a budget and the seam does not carry it, so it is exported; a
padding suffix is applied by the row and never read by a caller, and a digest
size is already on the seam as `digest_size`. What earns an export is a name a
consumer cannot otherwise obtain and would otherwise copy.

Until an export lands, convert a module wholesale or not at all. Splitting one
module's names across both spellings leaves the same coupling and two imports to
read.

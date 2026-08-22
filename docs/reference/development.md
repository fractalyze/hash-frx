# Development environment

`hash-frx` is pure Python on FRX, lowered through the Fractalyze XLA PJRT plugin.
The Bazel build is hermetic — it fetches the pinned wheels itself, so
`bazel test //...` needs no virtualenv. Everything below is about the loop
*around* that: the interactive venv, choosing a backend, and running against an
unreleased XLA. For a first install see the quick start in
[`../../README.md`](../../README.md).

## `FRX_*` is an alias for `JAX_*`

FRX is a fork of JAX and keeps JAX's configuration names internally, so its
config options are still `jax_*` and read `JAX_*` from the environment. On import
`frx` mirrors every `FRX_`-prefixed variable onto its `JAX_` twin with
`setdefault` — the mapping is by prefix rather than an enumerated list, so **any
`JAX_*` knob documented upstream has a working `FRX_*` spelling**, and an
explicitly set `JAX_*` wins over its alias.

That is why `.bazelrc` selects a backend with `FRX_PLATFORMS` while an error
message from inside the library still talks about `JAX_PLATFORMS`. Both are real;
they are the same variable.

## The GPU path needs CUDA 12 specifically

The pinned plugin is `frx-cuda12-*`, and it version-checks the CUDA runtime at
import. A machine carrying **CUDA 13** has no `libcudart.so.12`, so the check
fails and the backend never registers — `FRX_PLATFORMS=cuda` then dies with:

```text
RuntimeError: Unable to load CUDA. Is it installed?
RuntimeError: Backend 'cuda' is not in the list of known backends: ['cpu', 'tpu'].
```

Both messages point away from the actual cause. CUDA *is* installed; the major
version is wrong. The preceding `Could not find cuda drivers on your machine` is
equally misleading, and `ldd` on the plugin `.so` shows nothing missing because
the CUDA libraries are `dlopen`ed lazily rather than linked. One line settles it:

```sh
ldconfig -p | grep libcudart
```

`libcudart.so.13` and no `.so.12` means this box cannot run the GPU leg no matter
what the card is — an idle RTX 5090 behind CUDA 13 is still a CPU-only box for
these wheels. Run the GPU leg on a CUDA 12 machine, or install a CUDA 12 runtime
alongside.

## Run the GPU leg serialized on a card you are sharing

```sh
bazel test --test_env=FRX_PLATFORMS=cuda --local_test_jobs=1 -- //...
```

Each test process opens its own CUDA context and pre-allocates against the
*free* memory it sees, so several running at once starve each other. On a card
another process is already using, that reads as `CUDA_ERROR_OUT_OF_MEMORY`
across the tree — 20 of 30 targets in one run, including targets nothing in the
change under test touches. The same tree passed 30 of 30 serialized.

The tell that this is the environment rather than the code: the failed target's
`test.log` ends in `Ran N tests ... OK` while Bazel still reports FAILED. The
suite finished green and the process died afterwards, so the two verdicts
disagree. When they do, serialize before reading the test body.

CI's GPU leg runs on a dedicated box and needs none of this — it runs `//...`
unserialized.

## Running against a local Fractalyze XLA build

Plugin resolution has **no env-var indirection**: `frx_plugins/xla_cuda12` loads
the `.so` in the venv's site-packages, and there is no `*_GPU_PLUGIN_PATH`
override. To run against unmerged XLA changes, build the plugin and PJRT wheels
from that checkout (the XLA repo's `docs/build_from_source.md`) and force them
over the pinned ones:

```sh
pip install --force-reinstall --no-deps \
    <dist>/frx_cuda12_plugin-*.whl <dist>/frx_cuda12_pjrt-*.whl
```

This reaches the **venv loop only**. Bazel resolves its wheels from
`requirements_lock_3_11.txt`, so `bazel test` keeps running the pinned plugin no
matter what the venv holds. Once the XLA change is published, bump
`requirements.in` and regenerate the lock (`bazel run //:requirements.update`)
rather than keeping the force-install.

A stale plugin against newer Python surfaces as
`custom op 'stablehlo.composite' is unknown` on the first fresh compile — the
marker seam every primitive here lowers through is exactly what a mismatched
plugin fails to recognize.

## Which XLA a pinned wheel carries

A wheel's `devYYYYMMDDHHMMSS` suffix is the timestamp of the Dev Release run that
built it. It orders wheels; it says nothing about what is in one. When a change
on the XLA side is what you are waiting for — a new emitter, a fixed lowering —
resolve the wheel back to its source and check:

```sh
# The release tag matching the wheel's dev suffix names the jax commit built.
gh api repos/fractalyze/jax/git/ref/tags/dev-<YYYYMMDDHHMMSS> --jq .object.sha
# That commit pins the XLA the wheel carries.
gh api "repos/fractalyze/jax/contents/third_party/xla/revision.bzl?ref=<sha>" \
    --jq .content | base64 -d | grep XLA_COMMIT
# Finally, in an XLA checkout: is the commit you need an ancestor of that one?
git merge-base --is-ancestor <commit-you-need> <XLA_COMMIT> && echo present
```

The ancestor check is the part worth not skipping: the pinned XLA moves on, so
the answer is rarely the exact commit you are looking for. Two wheels published
47 minutes apart once straddled an emitter merge, and the later-numbered of the
two did not contain it — a bump to it would have looked like progress and
changed nothing.

## The compile cache is per toolchain

A persistent `FRX_COMPILATION_CACHE_DIR` skips recompiles across venv runs, which
is worth having because the compile dominates the run for a fusion-marked body.
Keep **one cache directory per frxlib build**, and treat a rebuilt wheel as a new
toolchain: self-built wheels reuse their version string, so a shared directory
serves the *other* build's executables back and the results are neither current
nor obviously stale.

Bazel passes only the variables named by `--test_env`, so a cache directory set
in the shell reaches the venv loop and never the Bazel one.

## The pytest loop

`bazel test //...` is the source of truth for "all tests pass"
([`conventions.md`](conventions.md)). For an interactive loop over one file,
pytest is configured in [`pyproject.toml`](../../pyproject.toml) but is
deliberately not a runtime pin, so install it into the venv:

```sh
pip install pytest
pytest hash_frx/testing/sponge_test.py
```

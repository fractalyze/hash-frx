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

## `bazel test` is one leg, not both

`.bazelrc` pins `test --test_env=FRX_PLATFORMS=cpu`, so a plain `bazel test
//...` runs the **CPU leg only**. The GPU leg is a second command:

```sh
bazel test //...                                            # CPU leg (the default)
bazel test --test_env=FRX_PLATFORMS=cuda \
           --local_test_jobs=1 //...                        # GPU leg
```

The two are different *programs*, not the same tests run twice: a family whose
emitter is GPU-only routes `DEDICATED` on one leg and `GENERIC` on the other, so
the marked region a step is meant to preserve exists on one leg and not the
other ([`../README.md#the-fusion-contract`](../README.md#the-fusion-contract)).
A change validated on one leg has been validated for half the wire surface.

`bazel run` behaves differently from `bazel test` here: it inherits the client
environment and lands on the real default backend, so forcing the other leg for
a `run` target is a plain `FRX_PLATFORMS=cpu bazel run ...` prefix.

**`--local_test_jobs=1` on the GPU leg is required, not tuning.** Concurrent
test jobs each reserve a large fraction of free VRAM, and the ones that lose
fail during device init. The message names the wrong thing:

```text
INTERNAL: RET_CHECK failure (gpu_compiler.cc:3064) dnn_support != nullptr
RuntimeError: Bad StatusOr access: RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY
```

The first line reads as a missing or mismatched cuDNN and sends you to
`ldconfig`; the cause is the second. Measured on a 32 GB card: 42 of 60 targets
failed in parallel, all 60 passed serially. Tells that it is contention rather
than a real break — targets fail that the change never touched, and `nvidia-smi`
shows the card empty afterwards because the hogs have exited. The
`Could not get kernel mode driver version` warning printed on every GPU run is
unrelated and benign.

## The lowering gate, for a change that must not move the wire

A refactor that is supposed to move code and nothing else needs a gate the
byte-exactness suites cannot provide: a renamed marker, a moved operand, a
dropped attribute or a lost composite all still compute the right digest,
because an unrecognized or absent marker inlines its decomposition. The digest
tests pass and the kernel is gone.
[`hash_frx/testing/lowering_golden.py`](../../hash_frx/testing/lowering_golden.py)
is the helper; the captures it takes are branch state, so the harness that
drives it is written per step and deleted before the PR.

```sh
# 1. A py_binary under hash_frx/testing/ that walks every row the step can move
#    x several message lengths x batch {1, 4}, json.dump()ing lowering_text()
#    per case. Include every family a step TOUCHES, not just the one it is
#    named for — a gate that stops at the family it was written for is not one.
# 2. Capture on the base revision, before any change, on both legs:
bazel run //hash_frx/testing:<capture> -- /tmp/golden-gpu.json
FRX_PLATFORMS=cpu bazel run //hash_frx/testing:<capture> -- /tmp/golden-cpu.json
# 3. Re-run with --compare after every commit, against both files.
# 4. Delete the harness and its BUILD target before the PR.
```

**A failing gate is answered by reverting, not by re-capturing.** Emission order
is part of what it pins, and re-baselining mid-refactor stops it protecting the
steps that follow. One concrete constraint this exposes: a shared helper fixes
emission order for all its callers, because Python evaluates a call's arguments
before the body — so a helper taking a `block` argument always emits that
block's ops before any slice it does internally. Two call sites that build the
same ops in opposite order cannot both go through it, and the one to keep inline
is whichever the gate says moved.

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
([`conventions.md`](conventions.md)) — on the leg it runs; see above for the
second one. For an interactive loop over one file,
pytest is configured in [`pyproject.toml`](../../pyproject.toml) but is
deliberately not a runtime pin, so install it into the venv:

```sh
pip install pytest
pytest hash_frx/testing/sponge_test.py
```

**A green target does not mean the cases you just wrote ran.** Test classes
appended below a module's trailing `if __name__ == "__main__": absltest.main()`
are never defined — `main()` exits first — so they are silently not collected
and the target still passes. Appending to an existing test file is the usual
way in; the runtime is the tell (a new suite that costs nothing). After editing
one:

```sh
grep -n '^class \|^if __name__' hash_frx/.../foo_test.py | tail -20
bazel test //hash_frx/...:foo_test --test_output=all 2>&1 | grep -E "^Ran [0-9]+ test"
```

Nothing should sit below the `__main__` block, and `Ran N tests` should match
the `def test_` count.

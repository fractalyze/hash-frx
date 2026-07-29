"""Shared Starlark helpers for hash-frx BUILD files."""

load("@hash_frx_pip//:requirements.bzl", "requirement")

# GPU runtime plugins (frx-cuda12 PJRT + plugin). Carried by every frx-using
# py_test — the GPU leg (FRX_PLATFORMS=cuda) initializes the device; the CPU leg
# never initializes the plugin, so they are inert there.
#
# Not free, though: `rules_python` fetches whl repos on demand, so these two are
# in the graph ONLY because this list puts them there. Measured on the CPU path —
# +130 MB fetched, 559 MB extracted into the output base, and 430 MB of the
# runfiles of every frx-using test (`xla_cuda_plugin.so` alone is 415 MB). Local
# runfiles are symlinks, so it mostly bites under `--config=remote`, where those
# blobs must reach CAS before any test action runs.
#
# Gating this behind a `bool_flag` + `select()` would keep them off the CPU leg
# at the cost of a build flag CI has to pass. Attach the list per-target rather
# than reflexively: a test that never touches a device does not need it.
GPU_PLUGIN_DEPS = [
    requirement("frx_cuda12_plugin"),
    requirement("frx_cuda12_pjrt"),
]

"""Shared Starlark helpers for hash-frx BUILD files."""

load("@hash_frx_pip//:requirements.bzl", "requirement")

# GPU runtime plugins (frx-cuda12 PJRT + plugin). Carried by every
# frx-using py_test — CI's GPU leg (FRX_PLATFORMS=cuda) initializes the device;
# the CPU leg never initializes the plugin, so they are inert there.
# Ungated (no select): two self-contained wheels already in the build graph, so
# the CPU leg pays no extra download. Revisit if a heavier CUDA wheel set ever
# lands in the lock.
GPU_PLUGIN_DEPS = [
    requirement("frx_cuda12_plugin"),
    requirement("frx_cuda12_pjrt"),
]

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0

# The single version literal. `pyproject.toml` packages it, release.yml requires
# a tag to equal it, dev-release.yml appends `.dev<timestamp>` to it.
#
# It names the release being worked toward, not the last one shipped — see
# .github/workflows/release.yml for why the bump follows a tag rather than
# preceding it.
__version__ = "0.1.0"

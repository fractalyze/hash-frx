# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Package-integrity guards for accidents tooling keeps reintroducing."""

from absl.testing import absltest

import hash_frx


class PackageTest(absltest.TestCase):
    def test_version_attr_survives(self) -> None:
        # dev-release stamps hash_frx.__version__ at release time, so an emptied
        # hash_frx/__init__.py breaks the wheel build and nothing else — weeks
        # later. Fail here instead.
        self.assertTrue(getattr(hash_frx, "__version__", ""))

    def test_version_is_a_final_release_version(self) -> None:
        # release.yml refuses a tag that disagrees with this, so a pre-release
        # or dev suffix reaching main makes every tag unreleasable — and makes
        # dev-release stamp `X.Y.ZrcN.devTS`. The suffixes belong on the build.
        self.assertRegex(hash_frx.__version__, r"^\d+(\.\d+)*$")


if __name__ == "__main__":
    absltest.main()

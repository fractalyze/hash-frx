# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The constructions carry a non-field permutation.

`Permutation.dtype` is any dtype a consumer can allocate and index, not a field
dtype, and this pins the part that makes that claim true: `Sponge` and
`Compression` do no arithmetic on state, so a machine-word permutation drives
them unchanged. Bit-oriented hashes (Keccak-f[1600] over uint32 lane halves,
BLAKE3's compression) therefore share the seam rather than forking it.

The permutation below is a machine-word bijection built from the operations
those hashes actually use — a lane rotation, a per-lane bit rotation, and an XOR
— so a construction that reached for field arithmetic would fail here rather
than in a hash nobody has written yet.

`DuplexSponge` is the exception and is checked here too, from the other side: its
absorb merges with `+`, which is the intended group operation for a field and the
wrong one for machine words, so it must *reject* this permutation rather than
silently wrap. The sponge to pair with a bit-oriented permutation is a separate
construction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array

from hash_frx.compression import Compression, CompressionParams
from hash_frx.duplex_sponge import DuplexSponge
from hash_frx.fusion import FUSED_REGION_MARKER, FusionPath
from hash_frx.permutation import Permutation
from hash_frx.poseidon2.testing.koalabear16 import koalabear16_perm
from hash_frx.sponge import Sponge, SpongeParams, SpongeType

_WIDTH = 8
_RATE = 5
_OUT = 3  # rate + out == width, so MERKLE_DAMGARD is legal too
_K = 0x9E3779B9
_ROT = 7


def _rotl(x: Array, n: int) -> Array:
    """Rotate each uint32 lane left by n — a per-lane bijection, as in Keccak's rho."""
    return (x << fnp.uint32(n)) | (x >> fnp.uint32(32 - n))


class _WordPerm:
    """A uint32-lane bijection: rotate lanes, rotate bits, XOR a constant.

    Each step is invertible, so the composition is a genuine permutation rather
    than merely a fixed-width map.
    """

    width = _WIDTH
    dtype = fnp.uint32
    fusion_path = FusionPath.GENERIC
    fused_region_marker = (FUSED_REGION_MARKER, 0)

    def permute(self, state: Array) -> Array:
        return _rotl(fnp.roll(state, 1), _ROT) ^ fnp.uint32(_K)

    # Inert fused-region ABI: non-fused, never called.
    def fused_region_spec(
        self, leading: Array
    ) -> tuple[tuple[Array, ...], Callable[..., Array], dict[str, object]]:
        return (leading,), (lambda state, *ops: self.permute(state)), {}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _WordPerm)

    def __hash__(self) -> int:
        return hash(("_WordPerm", self.width))


def _ref_permute(state: np.ndarray) -> np.ndarray:
    """The permutation again, in numpy — an oracle independent of frx's lowering."""
    rolled = np.roll(state, 1)
    rot = ((rolled << np.uint32(_ROT)) | (rolled >> np.uint32(32 - _ROT))).astype(
        np.uint32
    )
    return rot ^ np.uint32(_K)


class NonFieldPermutationTest(absltest.TestCase):
    def test_word_permutation_satisfies_the_protocol(self) -> None:
        self.assertIsInstance(_WordPerm(), Permutation)

    def test_compression_carries_word_lanes(self) -> None:
        # compress = zero pre-image, place arity*chunk lanes, permute, take chunk.
        c = Compression(_WordPerm(), CompressionParams(arity=2, chunk=3))
        inputs = np.arange(6, dtype=np.uint32).reshape(2, 3)
        got = c.compress(fnp.asarray(inputs))

        pre = np.zeros(_WIDTH, dtype=np.uint32)
        pre[:6] = inputs.reshape(-1)
        want = _ref_permute(pre)[:3]

        self.assertEqual(got.dtype, fnp.uint32)
        np.testing.assert_array_equal(np.asarray(got), want)

    def test_padding_free_sponge_carries_word_lanes(self) -> None:
        # One exact rate block: overwrite the rate lanes, permute, squeeze `out`.
        s = Sponge(_WordPerm(), SpongeParams(rate=_RATE, out=_OUT))
        inp = np.arange(_RATE, dtype=np.uint32)
        got = s.hash(fnp.asarray(inp))

        state = np.zeros(_WIDTH, dtype=np.uint32)
        state[:_RATE] = inp
        want = _ref_permute(state)[:_OUT]

        self.assertEqual(got.dtype, fnp.uint32)
        np.testing.assert_array_equal(np.asarray(got), want)

    def test_merkle_damgard_sponge_carries_word_lanes(self) -> None:
        # Two blocks, so the construction's zero-fill and capacity carry both run.
        s = Sponge(_WordPerm(), SpongeParams(rate=_RATE, out=_OUT))
        inp = np.arange(2 * _RATE, dtype=np.uint32)
        got = s.hash(fnp.asarray(inp), sponge_type=SpongeType.MERKLE_DAMGARD)

        state = np.zeros(_WIDTH, dtype=np.uint32)
        for blk in range(2):
            cap = state[:_OUT].copy()  # prior digest, snapshot before overwrite
            state[:_RATE] = inp[blk * _RATE : (blk + 1) * _RATE]
            state[_RATE : _RATE + _OUT] = cap
            state = _ref_permute(state)
        want = state[:_OUT]

        self.assertEqual(got.dtype, fnp.uint32)
        np.testing.assert_array_equal(np.asarray(got), want)

    def test_duplex_sponge_rejects_word_lanes(self) -> None:
        # The add-absorb is a carrying `+` on machine words, so the wrong result
        # would be silent bytes rather than an error. Rejected at construction.
        with self.assertRaises(TypeError):
            DuplexSponge(_WordPerm(), rate=_RATE)

    def test_duplex_sponge_accepts_a_field(self) -> None:
        # The rejection above must be about the dtype, not about DuplexSponge
        # being unusable — a field permutation of the same shape is accepted.
        self.assertIsInstance(DuplexSponge(koalabear16_perm(), rate=8), DuplexSponge)

    def test_sponge_lowers_under_jit(self) -> None:
        # The seam has to survive tracing, not just eager execution.
        s = Sponge(_WordPerm(), SpongeParams(rate=_RATE, out=_OUT))
        inp = fnp.asarray(np.arange(_RATE, dtype=np.uint32))
        np.testing.assert_array_equal(
            np.asarray(frx.jit(s.hash)(inp)), np.asarray(s.hash(inp))
        )


if TYPE_CHECKING:
    _perm: type[Permutation] = _WordPerm


if __name__ == "__main__":
    absltest.main()

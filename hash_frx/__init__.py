# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The package's public API — the names a consumer codes against.

Everything here is re-exported from the module that defines it, so a consumer
writes `from hash_frx import Sha256` rather than binding itself to the file
layout. That indirection is the point: the modules underneath move as the
package is re-layered into primitive / extension / adapter, and a consumer that
imported the layout would break on every move.

**The re-exports are lazy** (PEP 562), and that is load-bearing rather than an
optimization. Importing a hash puts its constant tables on the default backend,
which *initializes* that backend — measured at this commit, `import hash_frx` is
0.1 ms and starts no backend, while `import hash_frx.sha256.sha256` starts one. Binding
the names eagerly here would move that cost onto `import hash_frx` itself, which
would in turn defeat `markers.py`'s stated property that the wire-surface
registry can be read "free of every hash's dependencies" — reading the marker
list would boot a device. So `_EXPORTS` maps a name to the module that defines
it and `__getattr__` imports on first access, caching the binding in `globals()`
so the second access is a plain dict hit. (The underlying import-time device
materialization is #167; this file is careful not to widen it.)

`_EXPORTS` and the `TYPE_CHECKING` block below say the same thing twice — once
for the interpreter, once for mypy, which cannot see through a runtime
`__getattr__`. `package_test` holds the two equal, which is where a name added
to one and not the other is caught.

A name here may not collide with a submodule name: `from hash_frx import X`
prefers the attribute, but importing `hash_frx.X` anywhere binds the *module*
onto the package, so a collision resolves differently depending on what else has
been imported. `package_test` pins the absence of collisions, and
`public_api_test` pins that no export shares a name with a submodule.

`pbkdf2` was the one name that rule cost: the function could not be re-exported
while `hash_frx/pbkdf2.py` held the name. Moving the adapters under
`hash_frx/adapter/` freed it, so `from hash_frx import pbkdf2` is now the
function.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# The single version literal. `pyproject.toml` packages it, release.yml requires
# a tag to equal it, dev-release.yml appends `.dev<timestamp>` to it.
#
# It names the release being worked toward, not the last one shipped — see
# .github/workflows/release.yml for why the bump follows a tag rather than
# preceding it.
__version__ = "0.2.0"

# name -> the module that defines it. Grouped by layer, which is the taxonomy
# the package is being re-layered onto: a primitive is a fixed-width function, an
# extension is the schedule that feeds it a message, an adapter is a mode over an
# extension, and a row is one standard's named member.
_EXPORTS: dict[str, str] = {
    # -- seams -------------------------------------------------------------
    "ByteHash": "hash_frx.byte_hash",
    "Permutation": "hash_frx.permutation",
    # -- the fusion contract's vocabulary ----------------------------------
    "FUSED_REGION_MARKER": "hash_frx.fusion",
    "FusionPath": "hash_frx.fusion",
    "fused_region": "hash_frx.fusion",
    "fused_region_over": "hash_frx.fusion",
    "inert_region_spec": "hash_frx.fusion",
    # -- the wire surface (data only; importing this starts no backend) ----
    "MARKERS": "hash_frx.markers",
    "Marker": "hash_frx.markers",
    "MarkerKind": "hash_frx.markers",
    # -- primitives: permutations ------------------------------------------
    "AsconP": "hash_frx.ascon.permutation",
    "BabyBear16": "hash_frx.poseidon2.standard",
    "KeccakF1600": "hash_frx.keccak.permutation",
    "KoalaBear16": "hash_frx.poseidon2.standard",
    "Poseidon": "hash_frx.poseidon.poseidon",
    "Poseidon2": "hash_frx.poseidon2.poseidon2",
    "SparsePoseidon": "hash_frx.poseidon.sparse",
    "Vision": "hash_frx.vision.vision",
    # -- primitive parameters ----------------------------------------------
    "BABYBEAR16_PARAMS": "hash_frx.poseidon2.standard",
    "KECCAK256_RATE": "hash_frx.keccak.byte_hashes",
    "KOALABEAR16_PARAMS": "hash_frx.poseidon2.standard",
    "Poseidon2Params": "hash_frx.poseidon2.params",
    "PoseidonParams": "hash_frx.poseidon.params",
    "SHA3_224_RATE": "hash_frx.keccak.byte_hashes",
    "SHA3_256_RATE": "hash_frx.keccak.byte_hashes",
    "SHA3_384_RATE": "hash_frx.keccak.byte_hashes",
    "SHA3_512_RATE": "hash_frx.keccak.byte_hashes",
    "SHAKE128_RATE": "hash_frx.keccak.byte_hashes",
    "SHAKE256_RATE": "hash_frx.keccak.byte_hashes",
    "SparsePoseidonParams": "hash_frx.poseidon.params",
    "VisionParams": "hash_frx.vision.params",
    "default_external_matrix": "hash_frx.poseidon2.params",
    "vision_mark32_params": "hash_frx.vision.params",
    # -- extensions and the constructions over them ------------------------
    "Compression": "hash_frx.compression",
    "CompressionParams": "hash_frx.compression",
    "DEFAULT_CONVENTION": "hash_frx.duplex_sponge",
    "DuplexConvention": "hash_frx.duplex_sponge",
    "DuplexSponge": "hash_frx.duplex_sponge",
    "RateLanes": "hash_frx.duplex_sponge",
    "SpillPermute": "hash_frx.duplex_sponge",
    "scanned_absorb": "hash_frx.extension.sponge",
    "squeeze": "hash_frx.extension.sponge",
    "squeeze_blocks": "hash_frx.extension.sponge",
    "KeccakSponge": "hash_frx.keccak.sponge",
    "Sponge": "hash_frx.sponge",
    "SpongeChaining": "hash_frx.sponge",
    "SpongeParams": "hash_frx.sponge",
    # -- rows --------------------------
    "AsconHash256": "hash_frx.ascon.ascon",
    "AsconXof128": "hash_frx.ascon.ascon",
    "AsconCxof128": "hash_frx.ascon.ascon",
    "Blake2b": "hash_frx.blake2b.blake2b",
    "Blake2bKeyed": "hash_frx.blake2b.blake2b",
    "Blake2s": "hash_frx.blake2s.blake2s",
    "Blake2sKeyed": "hash_frx.blake2s.blake2s",
    "Blake3": "hash_frx.blake3.rows",
    "Blake3DeriveKey": "hash_frx.blake3.rows",
    "Blake3Keyed": "hash_frx.blake3.rows",
    "CShake128": "hash_frx.keccak.cshake",
    "CShake256": "hash_frx.keccak.cshake",
    "Grostl256": "hash_frx.grostl.grostl",
    "Keccak256": "hash_frx.keccak.byte_hashes",
    "Kmac128": "hash_frx.keccak.kmac",
    "Kmac256": "hash_frx.keccak.kmac",
    "KmacXof128": "hash_frx.keccak.kmac",
    "KmacXof256": "hash_frx.keccak.kmac",
    "kmac128": "hash_frx.keccak.kmac",
    "kmac256": "hash_frx.keccak.kmac",
    "kmac_xof128": "hash_frx.keccak.kmac",
    "kmac_xof256": "hash_frx.keccak.kmac",
    "TupleHash128": "hash_frx.keccak.tuple_hash",
    "TupleHash256": "hash_frx.keccak.tuple_hash",
    "TupleHashXof128": "hash_frx.keccak.tuple_hash",
    "TupleHashXof256": "hash_frx.keccak.tuple_hash",
    "Ripemd160": "hash_frx.ripemd160.ripemd160",
    "Sha224": "hash_frx.sha256.sha256",
    "Sha256": "hash_frx.sha256.sha256",
    "Sha384": "hash_frx.sha512.sha512",
    "Sha3_224": "hash_frx.keccak.byte_hashes",
    "Sha3_256": "hash_frx.keccak.byte_hashes",
    "Sha3_384": "hash_frx.keccak.byte_hashes",
    "Sha3_512": "hash_frx.keccak.byte_hashes",
    "Sha512": "hash_frx.sha512.sha512",
    "Sha512_224": "hash_frx.sha512.sha512",
    "Sha512_256": "hash_frx.sha512.sha512",
    "Shake128": "hash_frx.keccak.byte_hashes",
    "Shake256": "hash_frx.keccak.byte_hashes",
    "Sm3": "hash_frx.sm3.sm3",
    # -- streaming state (the midstate is per-construction, so are these) --
    "Blake3Stream": "hash_frx.blake3.streaming",
    "Sha256State": "hash_frx.sha256.sha256",
    "Sha512State": "hash_frx.sha512.sha512",
    "ShakeAbsorb": "hash_frx.keccak.streaming",
    "ShakeBlockSqueeze": "hash_frx.keccak.streaming",
    "ShakeSqueeze": "hash_frx.keccak.streaming",
    "blake3_stream_init": "hash_frx.blake3.streaming",
    "sha256_stream_absorb": "hash_frx.sha256.sha256",
    "sha256_stream_finalize": "hash_frx.sha256.sha256",
    "sha256_stream_init": "hash_frx.sha256.sha256",
    "sha512_stream_absorb": "hash_frx.sha512.sha512",
    "sha512_stream_finalize": "hash_frx.sha512.sha512",
    "sha512_stream_init": "hash_frx.sha512.sha512",
    "shake128_init": "hash_frx.keccak.streaming",
    "shake256_init": "hash_frx.keccak.streaming",
    "shake_init": "hash_frx.keccak.streaming",
    # -- adapters ----------------------------------------------------------
    "ARK_0_3": "hash_frx.adapter.duplex",
    "ARK_0_5": "hash_frx.adapter.duplex",
    "Hmac": "hash_frx.adapter.hmac",
    "Mgf1": "hash_frx.adapter.mgf1",
    "Xof": "hash_frx.adapter.xof",
    "block_size": "hash_frx.adapter.block_size",
    "hkdf_expand": "hash_frx.adapter.hkdf",
    "hkdf_extract": "hash_frx.adapter.hkdf",
    "pbkdf2": "hash_frx.adapter.pbkdf2",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Import the module that defines `name` on first access (PEP 562).

    The binding is cached in `globals()`, so this runs once per name and every
    later access is an ordinary attribute lookup. A name outside `_EXPORTS`
    raises `AttributeError` rather than being invented, which is also what lets
    `from hash_frx import sha256` keep resolving to the *submodule*: the import
    system falls back to importing it once this declines.
    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """The public names, so `dir(hash_frx)` and tab-completion see the exports
    that `__getattr__` has not been asked for yet."""
    return sorted(__all__)


# mypy cannot see through the runtime `__getattr__`, so the same surface is
# spelled statically here. It costs nothing at run time and `package_test` holds
# it equal to `_EXPORTS`.
if TYPE_CHECKING:
    from hash_frx.adapter.block_size import block_size as block_size
    from hash_frx.adapter.duplex import ARK_0_3 as ARK_0_3
    from hash_frx.adapter.duplex import ARK_0_5 as ARK_0_5
    from hash_frx.adapter.hkdf import hkdf_expand as hkdf_expand
    from hash_frx.adapter.hkdf import hkdf_extract as hkdf_extract
    from hash_frx.adapter.hmac import Hmac as Hmac
    from hash_frx.adapter.mgf1 import Mgf1 as Mgf1
    from hash_frx.adapter.pbkdf2 import pbkdf2 as pbkdf2
    from hash_frx.adapter.xof import Xof as Xof
    from hash_frx.ascon.ascon import AsconCxof128 as AsconCxof128
    from hash_frx.ascon.ascon import AsconHash256 as AsconHash256
    from hash_frx.ascon.ascon import AsconXof128 as AsconXof128
    from hash_frx.ascon.permutation import AsconP as AsconP
    from hash_frx.blake2b.blake2b import Blake2b as Blake2b
    from hash_frx.blake2b.blake2b import Blake2bKeyed as Blake2bKeyed
    from hash_frx.blake2s.blake2s import Blake2s as Blake2s
    from hash_frx.blake2s.blake2s import Blake2sKeyed as Blake2sKeyed
    from hash_frx.blake3.rows import Blake3 as Blake3
    from hash_frx.blake3.rows import Blake3DeriveKey as Blake3DeriveKey
    from hash_frx.blake3.rows import Blake3Keyed as Blake3Keyed
    from hash_frx.blake3.streaming import Blake3Stream as Blake3Stream
    from hash_frx.blake3.streaming import blake3_stream_init as blake3_stream_init
    from hash_frx.byte_hash import ByteHash as ByteHash
    from hash_frx.compression import Compression as Compression
    from hash_frx.compression import CompressionParams as CompressionParams
    from hash_frx.duplex_sponge import DEFAULT_CONVENTION as DEFAULT_CONVENTION
    from hash_frx.duplex_sponge import DuplexConvention as DuplexConvention
    from hash_frx.duplex_sponge import DuplexSponge as DuplexSponge
    from hash_frx.duplex_sponge import RateLanes as RateLanes
    from hash_frx.duplex_sponge import SpillPermute as SpillPermute
    from hash_frx.extension.sponge import scanned_absorb as scanned_absorb
    from hash_frx.extension.sponge import squeeze as squeeze
    from hash_frx.extension.sponge import squeeze_blocks as squeeze_blocks
    from hash_frx.fusion import FUSED_REGION_MARKER as FUSED_REGION_MARKER
    from hash_frx.fusion import FusionPath as FusionPath
    from hash_frx.fusion import fused_region as fused_region
    from hash_frx.fusion import fused_region_over as fused_region_over
    from hash_frx.fusion import inert_region_spec as inert_region_spec
    from hash_frx.grostl.grostl import Grostl256 as Grostl256
    from hash_frx.keccak.byte_hashes import KECCAK256_RATE as KECCAK256_RATE
    from hash_frx.keccak.byte_hashes import SHA3_224_RATE as SHA3_224_RATE
    from hash_frx.keccak.byte_hashes import SHA3_256_RATE as SHA3_256_RATE
    from hash_frx.keccak.byte_hashes import SHA3_384_RATE as SHA3_384_RATE
    from hash_frx.keccak.byte_hashes import SHA3_512_RATE as SHA3_512_RATE
    from hash_frx.keccak.byte_hashes import SHAKE128_RATE as SHAKE128_RATE
    from hash_frx.keccak.byte_hashes import SHAKE256_RATE as SHAKE256_RATE
    from hash_frx.keccak.byte_hashes import Keccak256 as Keccak256
    from hash_frx.keccak.byte_hashes import Sha3_224 as Sha3_224
    from hash_frx.keccak.byte_hashes import Sha3_256 as Sha3_256
    from hash_frx.keccak.byte_hashes import Sha3_384 as Sha3_384
    from hash_frx.keccak.byte_hashes import Sha3_512 as Sha3_512
    from hash_frx.keccak.byte_hashes import Shake128 as Shake128
    from hash_frx.keccak.byte_hashes import Shake256 as Shake256
    from hash_frx.keccak.cshake import CShake128 as CShake128
    from hash_frx.keccak.cshake import CShake256 as CShake256
    from hash_frx.keccak.kmac import Kmac128 as Kmac128
    from hash_frx.keccak.kmac import Kmac256 as Kmac256
    from hash_frx.keccak.kmac import KmacXof128 as KmacXof128
    from hash_frx.keccak.kmac import KmacXof256 as KmacXof256
    from hash_frx.keccak.kmac import kmac128 as kmac128
    from hash_frx.keccak.kmac import kmac256 as kmac256
    from hash_frx.keccak.kmac import kmac_xof128 as kmac_xof128
    from hash_frx.keccak.kmac import kmac_xof256 as kmac_xof256
    from hash_frx.keccak.permutation import KeccakF1600 as KeccakF1600
    from hash_frx.keccak.sponge import KeccakSponge as KeccakSponge
    from hash_frx.keccak.streaming import ShakeAbsorb as ShakeAbsorb
    from hash_frx.keccak.streaming import (
        ShakeBlockSqueeze as ShakeBlockSqueeze,
    )
    from hash_frx.keccak.streaming import ShakeSqueeze as ShakeSqueeze
    from hash_frx.keccak.streaming import shake128_init as shake128_init
    from hash_frx.keccak.streaming import shake256_init as shake256_init
    from hash_frx.keccak.streaming import shake_init as shake_init
    from hash_frx.keccak.tuple_hash import TupleHash128 as TupleHash128
    from hash_frx.keccak.tuple_hash import TupleHash256 as TupleHash256
    from hash_frx.keccak.tuple_hash import TupleHashXof128 as TupleHashXof128
    from hash_frx.keccak.tuple_hash import TupleHashXof256 as TupleHashXof256
    from hash_frx.markers import MARKERS as MARKERS
    from hash_frx.markers import Marker as Marker
    from hash_frx.markers import MarkerKind as MarkerKind
    from hash_frx.permutation import Permutation as Permutation
    from hash_frx.poseidon.params import PoseidonParams as PoseidonParams
    from hash_frx.poseidon.params import SparsePoseidonParams as SparsePoseidonParams
    from hash_frx.poseidon.poseidon import Poseidon as Poseidon
    from hash_frx.poseidon.sparse import SparsePoseidon as SparsePoseidon
    from hash_frx.poseidon2.params import Poseidon2Params as Poseidon2Params
    from hash_frx.poseidon2.params import (
        default_external_matrix as default_external_matrix,
    )
    from hash_frx.poseidon2.poseidon2 import Poseidon2 as Poseidon2
    from hash_frx.poseidon2.standard import (
        BABYBEAR16_PARAMS as BABYBEAR16_PARAMS,
    )
    from hash_frx.poseidon2.standard import (
        KOALABEAR16_PARAMS as KOALABEAR16_PARAMS,
    )
    from hash_frx.poseidon2.standard import BabyBear16 as BabyBear16
    from hash_frx.poseidon2.standard import KoalaBear16 as KoalaBear16
    from hash_frx.ripemd160.ripemd160 import Ripemd160 as Ripemd160
    from hash_frx.sha256.sha256 import Sha224 as Sha224
    from hash_frx.sha256.sha256 import Sha256 as Sha256
    from hash_frx.sha256.sha256 import Sha256State as Sha256State
    from hash_frx.sha256.sha256 import sha256_stream_absorb as sha256_stream_absorb
    from hash_frx.sha256.sha256 import sha256_stream_finalize as sha256_stream_finalize
    from hash_frx.sha256.sha256 import sha256_stream_init as sha256_stream_init
    from hash_frx.sha512.sha512 import Sha384 as Sha384
    from hash_frx.sha512.sha512 import Sha512 as Sha512
    from hash_frx.sha512.sha512 import Sha512_224 as Sha512_224
    from hash_frx.sha512.sha512 import Sha512_256 as Sha512_256
    from hash_frx.sha512.sha512 import Sha512State as Sha512State
    from hash_frx.sha512.sha512 import sha512_stream_absorb as sha512_stream_absorb
    from hash_frx.sha512.sha512 import sha512_stream_finalize as sha512_stream_finalize
    from hash_frx.sha512.sha512 import sha512_stream_init as sha512_stream_init
    from hash_frx.sm3.sm3 import Sm3 as Sm3
    from hash_frx.sponge import Sponge as Sponge
    from hash_frx.sponge import SpongeChaining as SpongeChaining
    from hash_frx.sponge import SpongeParams as SpongeParams
    from hash_frx.vision.params import VisionParams as VisionParams
    from hash_frx.vision.params import vision_mark32_params as vision_mark32_params
    from hash_frx.vision.vision import Vision as Vision

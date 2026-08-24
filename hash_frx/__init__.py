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
0.1 ms and starts no backend, while `import hash_frx.sha256` starts one. Binding
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
    "KeccakF1600": "hash_frx.keccak.permutation",
    "KoalaBear16": "hash_frx.poseidon2.standard",
    "Poseidon": "hash_frx.poseidon.poseidon",
    "Poseidon2": "hash_frx.poseidon2.poseidon2",
    "SparsePoseidon": "hash_frx.poseidon.sparse",
    "Vision": "hash_frx.vision.vision",
    # -- primitive parameters ----------------------------------------------
    "KOALABEAR16_PARAMS": "hash_frx.poseidon2.standard",
    "Poseidon2Params": "hash_frx.poseidon2.params",
    "PoseidonParams": "hash_frx.poseidon.params",
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
    "KeccakSponge": "hash_frx.keccak.sponge",
    "Sponge": "hash_frx.sponge",
    "SpongeParams": "hash_frx.sponge",
    "SpongeType": "hash_frx.sponge",
    # -- rows: device (traceable, return `Array`) --------------------------
    "AsconHash256": "hash_frx.ascon.ascon",
    "AsconXof128": "hash_frx.ascon.ascon",
    "Blake2b": "hash_frx.blake2b.blake2b",
    "Blake2s": "hash_frx.blake2s",
    "Blake3": "hash_frx.blake3.rows",
    "Blake3DeriveKey": "hash_frx.blake3.rows",
    "Blake3Keyed": "hash_frx.blake3.rows",
    "Grostl256": "hash_frx.grostl.grostl",
    "Keccak256": "hash_frx.keccak.byte_hashes",
    "Ripemd160": "hash_frx.ripemd160",
    "Sha256": "hash_frx.sha256",
    "Sha384": "hash_frx.sha512",
    "Sha3_256": "hash_frx.keccak.byte_hashes",
    "Sha3_512": "hash_frx.keccak.byte_hashes",
    "Sha512": "hash_frx.sha512",
    "Sha512_256": "hash_frx.sha512",
    "Shake128": "hash_frx.keccak.byte_hashes",
    "Shake256": "hash_frx.keccak.byte_hashes",
    "Sm3": "hash_frx.sm3",
    # -- rows: host (never traceable, return `np.ndarray`) -----------------
    "HostBlake2b": "hash_frx.blake2b.byte_hashes",
    "HostBlake2s": "hash_frx.blake2s",
    "HostBlake3": "hash_frx.blake3.rows",
    "HostBlake3DeriveKey": "hash_frx.blake3.rows",
    "HostBlake3Keyed": "hash_frx.blake3.rows",
    "HostSha256": "hash_frx.sha256",
    "HostSha384": "hash_frx.sha512",
    "HostSha3_256": "hash_frx.keccak.byte_hashes",
    "HostSha3_512": "hash_frx.keccak.byte_hashes",
    "HostSha512": "hash_frx.sha512",
    "HostSha512_256": "hash_frx.sha512",
    "HostShake128": "hash_frx.keccak.byte_hashes",
    "HostShake256": "hash_frx.keccak.byte_hashes",
    "HostSm3": "hash_frx.sm3",
    # -- streaming state (the midstate is per-construction, so are these) --
    "Blake3Stream": "hash_frx.blake3.streaming",
    "Sha256State": "hash_frx.sha256",
    "Sha512State": "hash_frx.sha512",
    "ShakeAbsorb": "hash_frx.keccak.streaming",
    "ShakeSqueeze": "hash_frx.keccak.streaming",
    "blake3_stream_init": "hash_frx.blake3.streaming",
    "sha256_stream_absorb": "hash_frx.sha256",
    "sha256_stream_finalize": "hash_frx.sha256",
    "sha256_stream_init": "hash_frx.sha256",
    "sha512_stream_absorb": "hash_frx.sha512",
    "sha512_stream_finalize": "hash_frx.sha512",
    "sha512_stream_init": "hash_frx.sha512",
    "shake128_init": "hash_frx.keccak.streaming",
    "shake256_init": "hash_frx.keccak.streaming",
    "shake_init": "hash_frx.keccak.streaming",
    # -- adapters ----------------------------------------------------------
    "ARK_0_3": "hash_frx.adapter.duplex",
    "ARK_0_5": "hash_frx.adapter.duplex",
    # `Dual` names every paired row type, so importing it imports eight row
    # modules and starts a backend: measured cold, 570 ms and 886 modules
    # against 5.5 ms and 65 for `Xof` and `block_size` beside it. That buys a
    # rename failing at import rather than at a consumer's call site
    # (`adapter/dual.py`); it is called out here because nothing about this
    # group hints at the spread.
    "Dual": "hash_frx.adapter.dual",
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
    from hash_frx.adapter.dual import Dual as Dual
    from hash_frx.adapter.duplex import ARK_0_3 as ARK_0_3
    from hash_frx.adapter.duplex import ARK_0_5 as ARK_0_5
    from hash_frx.adapter.hkdf import hkdf_expand as hkdf_expand
    from hash_frx.adapter.hkdf import hkdf_extract as hkdf_extract
    from hash_frx.adapter.hmac import Hmac as Hmac
    from hash_frx.adapter.mgf1 import Mgf1 as Mgf1
    from hash_frx.adapter.pbkdf2 import pbkdf2 as pbkdf2
    from hash_frx.adapter.xof import Xof as Xof
    from hash_frx.ascon.ascon import AsconHash256 as AsconHash256
    from hash_frx.ascon.ascon import AsconXof128 as AsconXof128
    from hash_frx.ascon.permutation import AsconP as AsconP
    from hash_frx.blake2b.blake2b import Blake2b as Blake2b
    from hash_frx.blake2b.byte_hashes import HostBlake2b as HostBlake2b
    from hash_frx.blake2s import Blake2s as Blake2s
    from hash_frx.blake2s import HostBlake2s as HostBlake2s
    from hash_frx.blake3.rows import Blake3 as Blake3
    from hash_frx.blake3.rows import Blake3DeriveKey as Blake3DeriveKey
    from hash_frx.blake3.rows import Blake3Keyed as Blake3Keyed
    from hash_frx.blake3.rows import HostBlake3 as HostBlake3
    from hash_frx.blake3.rows import HostBlake3DeriveKey as HostBlake3DeriveKey
    from hash_frx.blake3.rows import HostBlake3Keyed as HostBlake3Keyed
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
    from hash_frx.fusion import FUSED_REGION_MARKER as FUSED_REGION_MARKER
    from hash_frx.fusion import FusionPath as FusionPath
    from hash_frx.fusion import fused_region as fused_region
    from hash_frx.fusion import fused_region_over as fused_region_over
    from hash_frx.fusion import inert_region_spec as inert_region_spec
    from hash_frx.grostl.grostl import Grostl256 as Grostl256
    from hash_frx.keccak.byte_hashes import HostSha3_256 as HostSha3_256
    from hash_frx.keccak.byte_hashes import HostSha3_512 as HostSha3_512
    from hash_frx.keccak.byte_hashes import HostShake128 as HostShake128
    from hash_frx.keccak.byte_hashes import HostShake256 as HostShake256
    from hash_frx.keccak.byte_hashes import Keccak256 as Keccak256
    from hash_frx.keccak.byte_hashes import Sha3_256 as Sha3_256
    from hash_frx.keccak.byte_hashes import Sha3_512 as Sha3_512
    from hash_frx.keccak.byte_hashes import Shake128 as Shake128
    from hash_frx.keccak.byte_hashes import Shake256 as Shake256
    from hash_frx.keccak.permutation import KeccakF1600 as KeccakF1600
    from hash_frx.keccak.sponge import KeccakSponge as KeccakSponge
    from hash_frx.keccak.streaming import ShakeAbsorb as ShakeAbsorb
    from hash_frx.keccak.streaming import ShakeSqueeze as ShakeSqueeze
    from hash_frx.keccak.streaming import shake128_init as shake128_init
    from hash_frx.keccak.streaming import shake256_init as shake256_init
    from hash_frx.keccak.streaming import shake_init as shake_init
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
        KOALABEAR16_PARAMS as KOALABEAR16_PARAMS,
    )
    from hash_frx.poseidon2.standard import KoalaBear16 as KoalaBear16
    from hash_frx.ripemd160 import Ripemd160 as Ripemd160
    from hash_frx.sha256 import HostSha256 as HostSha256
    from hash_frx.sha256 import Sha256 as Sha256
    from hash_frx.sha256 import Sha256State as Sha256State
    from hash_frx.sha256 import sha256_stream_absorb as sha256_stream_absorb
    from hash_frx.sha256 import sha256_stream_finalize as sha256_stream_finalize
    from hash_frx.sha256 import sha256_stream_init as sha256_stream_init
    from hash_frx.sha512 import HostSha384 as HostSha384
    from hash_frx.sha512 import HostSha512 as HostSha512
    from hash_frx.sha512 import HostSha512_256 as HostSha512_256
    from hash_frx.sha512 import Sha384 as Sha384
    from hash_frx.sha512 import Sha512 as Sha512
    from hash_frx.sha512 import Sha512_256 as Sha512_256
    from hash_frx.sha512 import Sha512State as Sha512State
    from hash_frx.sha512 import sha512_stream_absorb as sha512_stream_absorb
    from hash_frx.sha512 import sha512_stream_finalize as sha512_stream_finalize
    from hash_frx.sha512 import sha512_stream_init as sha512_stream_init
    from hash_frx.sm3 import HostSm3 as HostSm3
    from hash_frx.sm3 import Sm3 as Sm3
    from hash_frx.sponge import Sponge as Sponge
    from hash_frx.sponge import SpongeParams as SpongeParams
    from hash_frx.sponge import SpongeType as SpongeType
    from hash_frx.vision.params import VisionParams as VisionParams
    from hash_frx.vision.params import vision_mark32_params as vision_mark32_params
    from hash_frx.vision.vision import Vision as Vision

# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The BLAKE3 team's published test vectors, split by what they exercise.

**Why these are the anchor, stated once for the package.** BLAKE3 has no
standard-library implementation to act as the third party `hashlib` is for
SHA-3, so a second reading of the spec is all this repo could otherwise check
itself against — and two implementations written here would agree with each
other just as readily while both being wrong.

The vectors pin whole hashes rather than compression intermediates, which looks
at first like they cannot reach anything below a whole hash. They can: an input
of at most one 1024-byte chunk hashes as the compression function chained over
that chunk's blocks with no tree above it, and at most 64 bytes is a single
compression call. So `SINGLE_BLOCK` pins the compression function directly and
`MULTI_BLOCK` adds the chaining, both without any tree machinery in between.

`EXTENDED` is the same table read to its full width: the published `hash` field
is **131 bytes**, of which the 32 above are only the first. So one column pins
the digest and the other pins the extendable output, at every length, with no
second source — and the first 32 bytes of each `EXTENDED` row are its
`ALL_LENGTHS` row, which is the standard's own statement that a digest is the
head of the stream rather than a different computation.

131 bytes is three 64-byte output blocks and a 3-byte remainder, so the table
reaches the block counter running 0, 1, 2 and a partial trailing block at every
input length it covers.

The digest groups are grouped by the layer each range first reaches — a single
compression, a chunk chain, a parent-node tree — so a group names what its
lengths would catch. A test that wants one layer selects its group;
`ALL_LENGTHS` is the whole published table, which is what a whole-hash test
wants.

Provenance: `test_vectors/test_vectors.json` in BLAKE3-team/BLAKE3.
"""

from __future__ import annotations

# One compression call each — these pin the compression function itself.
SINGLE_BLOCK = (
    (0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
    (1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"),
    (2, "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63"),
    (3, "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f"),
    (4, "f30f5ab28fe047904037f77b6da4fea1e27241c5d132638d8bedce9d40494f32"),
    (5, "b40b44dfd97e7a84a996a91af8b85188c66c126940ba7aad2e7ae6b385402aa2"),
    (6, "06c4e8ffb6872fad96f9aaca5eee1553eb62aed0ad7198cef42e87f6a616c844"),
    (7, "3f8770f387faad08faa9d8414e9f449ac68e6ff0417f673f602a646a891419fe"),
    (8, "2351207d04fc16ade43ccab08600939c7c1fa70a5c0aaca76063d04c3228eaeb"),
    (63, "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"),
    (64, "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"),
)

# Several blocks chained inside one chunk: the CHUNK_START / CHUNK_END flag
# placement and the chaining-value hand-off, which a single block cannot reach.
MULTI_BLOCK = (
    (65, "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"),
    (127, "d81293fda863f008c09e92fc382a81f5a0b4a1251cba1634016a0f86a6bd640d"),
    (128, "f17e570564b26578c33bb7f44643f539624b05df1a76c81f30acd548c44b45ef"),
    (129, "683aaae9f3c5ba37eaaf072aed0f9e30bac0865137bae68b1fde4ca2aebdcb12"),
    (1023, "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"),
    (1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"),
)


# More than one chunk, so a parent-node tree stands above them. The chunk counts
# are what these are chosen for rather than the byte counts: 3, 5, 6, 7, 9, 31
# and 100 are not powers of two, and a tree built by halving instead of by the
# standard's largest-power-of-two split agrees with one on every power of two and
# on nothing else.
MULTI_CHUNK = (
    (1025, "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"),
    (2048, "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"),
    (2049, "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030"),
    (3072, "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2"),
    (3073, "7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd3"),
    (4096, "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e969"),
    (4097, "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995"),
    (5120, "9cadc15fed8b5d854562b26a9536d9707cadeda9b143978f319ab34230535833"),
    (5121, "628bd2cb2004694adaab7bbd778a25df25c47b9d4155a55f8fbd79f2fe154cff"),
    (6144, "3e2e5b74e048f3add6d21faab3f83aa44d3b2278afb83b80b3c35164ebeca205"),
    (6145, "f1323a8631446cc50536a9f705ee5cb619424d46887f3c376c695b70e0f0507f"),
    (7168, "61da957ec2499a95d6b8023e2b0e604ec7f6b50e80a9678b89d2628e99ada77a"),
    (7169, "a003fc7a51754a9b3c7fae0367ab3d782dccf28855a03d435f8cfe74605e7817"),
    (8192, "aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a63"),
    (8193, "bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b"),
    (16384, "f875d6646de28985646f34ee13be9a576fd515f76b5b0a26bb324735041ddde4"),
    (31744, "62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47"),
    (102400, "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085"),
)

# Every length the standard publishes.
ALL_LENGTHS = SINGLE_BLOCK + MULTI_BLOCK + MULTI_CHUNK

# The same lengths, with the published output read to its full 131 bytes.
EXTENDED = (
    (
        0,
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262e00f03e7b69af26b7faaf09fcd333050"
        "338ddfe085b8cc869ca98b206c08243a26f5487789e8f660afe6c99ef9e0c52b92e7393024a80459cf91f476f9ffdbda"
        "7001c22e159b402631f277ca96f2defdf1078282314e763699a31c5363165421cce14d",
    ),
    (
        1,
        "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213c3a6cb8bf623e20cdb535f8d1a5ffb86"
        "342d9c0b64aca3bce1d31f60adfa137b358ad4d79f97b47c3d5e79f179df87a3b9776ef8325f8329886ba42f07fb138b"
        "b502f4081cbcec3195c5871e6c23e2cc97d3c69a613eba131e5f1351f3f1da786545e5",
    ),
    (
        2,
        "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63d8386b22e2ddc05836b7c1bb693d92af"
        "006deb5ffbc4c70fb44d0195d0c6f252faac61659ef86523aa16517f87cb5f1340e723756ab65efb2f91964e14391de2"
        "a432263a6faf1d146937b35a33621c12d00be8223a7f1919cec0acd12097ff3ab00ab1",
    ),
    (
        3,
        "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f5b49b82f805a538c68915c1ae8035c90"
        "0fd1d4b13902920fd05e1450822f36de9454b7e9996de4900c8e723512883f93f4345f8a58bfe64ee38d3ad71ab02776"
        "5d25cdd0e448328a8e7a683b9a6af8b0af94fa09010d9186890b096a08471e4230a134",
    ),
    (
        4,
        "f30f5ab28fe047904037f77b6da4fea1e27241c5d132638d8bedce9d40494f328f603ba4564453e06cdcee6cbe728a45"
        "19bbe6f0d41e8a14b5b225174a566dbfa61b56afb1e452dc08c804f8c3143c9e2cc4a31bb738bf8c1917b55830c6e657"
        "97211701dc0b98daa1faeaa6ee9e56ab606ce03a1a881e8f14e87a4acf4646272cfd12",
    ),
    (
        5,
        "b40b44dfd97e7a84a996a91af8b85188c66c126940ba7aad2e7ae6b385402aa2ebcfdac6c5d32c31209e1f81a4547512"
        "80db64942ce395104e1e4eaca62607de1c2ca748251754ea5bbe8c20150e7f47efd57012c63b3c6a6632dc1c7cd15f3e"
        "1c999904037d60fac2eb9397f2adbe458d7f264e64f1e73aa927b30988e2aed2f03620",
    ),
    (
        6,
        "06c4e8ffb6872fad96f9aaca5eee1553eb62aed0ad7198cef42e87f6a616c844611a30c4e4f37fe2fe23c0883cde5cf7"
        "059d88b657c7ed2087e3d210925ede716435d6d5d82597a1e52b9553919e804f5656278bd739880692c94bff2824d8e0"
        "b48cac1d24682699e4883389dc4f2faa2eb3b4db6e39debd5061ff3609916f3e07529a",
    ),
    (
        7,
        "3f8770f387faad08faa9d8414e9f449ac68e6ff0417f673f602a646a891419fe66036ef6e6d1a8f54baa9fed1fc11c77"
        "cfb9cff65bae915045027046ebe0c01bf5a941f3bb0f73791d3fc0b84370f9f30af0cd5b0fc334dd61f70feb60dad785"
        "f070fef1f343ed933b49a5ca0d16a503f599a365a4296739248b28d1a20b0e2cc8975c",
    ),
    (
        8,
        "2351207d04fc16ade43ccab08600939c7c1fa70a5c0aaca76063d04c3228eaeb725d6d46ceed8f785ab9f2f9b06acfe3"
        "98c6699c6129da084cb531177445a682894f9685eaf836999221d17c9a64a3a057000524cd2823986db378b074290a1a"
        "9b93a22e135ed2c14c7e20c6d045cd00b903400374126676ea78874d79f2dd7883cf5c",
    ),
    (
        63,
        "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b1197012b1e7d9af4d7cb7bdd1f3bb49a"
        "90a9b5dec3ea2bbc6eaebce77f4e470cbf4687093b5352f04e4a4570fba233164e6acc36900e35d185886a827f7ea9bd"
        "c1e5c3ce88b095a200e62c10c043b3e9bc6cb9b6ac4dfa51794b02ace9f98779040755",
    ),
    (
        64,
        "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98fc9cc56cb831ffe33ea8e7e1d1df09b2"
        "6efd2767670066aa82d023b1dfe8ab1b2b7fbb5b97592d46ffe3e05a6a9b592e2949c74160e4674301bc3f97e04903f8"
        "c6cf95b863174c33228924cdef7ae47559b10b294acd660666c4538833582b43f82d74",
    ),
    (
        65,
        "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee0e16e0a4749d6811dd1d6d1265c29729"
        "b1b75a9ac346cf93f0e1d7296dfcfd4313b3a227faaaaf7757cc95b4e87a49be3b8a270a12020233509b1c3632b3485e"
        "ef309d0abc4a4a696c9decc6e90454b53b000f456a3f10079072baaf7a981653221f2c",
    ),
    (
        127,
        "d81293fda863f008c09e92fc382a81f5a0b4a1251cba1634016a0f86a6bd640de3137d477156d1fde56b0cf36f8ef18b"
        "44b2d79897bece12227539ac9ae0a5119da47644d934d26e74dc316145dcb8bb69ac3f2e05c242dd6ee06484fcb0e956"
        "dc44355b452c5e2bbb5e2b66e99f5dd443d0cbcaaafd4beebaed24ae2f8bb672bcef78",
    ),
    (
        128,
        "f17e570564b26578c33bb7f44643f539624b05df1a76c81f30acd548c44b45efa69faba091427f9c5c4caa873aa07828"
        "651f19c55bad85c47d1368b11c6fd99e47ecba5820a0325984d74fe3e4058494ca12e3f1d3293d0010a9722f7dee64f7"
        "1246f75e9361f44cc8e214a100650db1313ff76a9f93ec6e84edb7add1cb4a95019b0c",
    ),
    (
        129,
        "683aaae9f3c5ba37eaaf072aed0f9e30bac0865137bae68b1fde4ca2aebdcb12f96ffa7b36dd78ba321be7e842d364a6"
        "2a42e3746681c8bace18a4a8a79649285c7127bf8febf125be9de39586d251f0d41da20980b70d35e3dac0eee59e468a"
        "894fa7e6a07129aaad09855f6ad4801512a116ba2b7841e6cfc99ad77594a8f2d181a7",
    ),
    (
        1023,
        "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11a182d27a591b05592b15607500e1e8dd"
        "56bc6c7fc063715b7a1d737df5bad3339c56778957d870eb9717b57ea3d9fb68d1b55127bba6a906a4a24bbd5acb2d12"
        "3a37b28f9e9a81bbaae360d58f85e5fc9d75f7c370a0cc09b6522d9c8d822f2f28f485",
    ),
    (
        1024,
        "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af71cf8107265ecdaf8505b95d8fcec83a9"
        "8a6a96ea5109d2c179c47a387ffbb404756f6eeae7883b446b70ebb144527c2075ab8ab204c0086bb22b7c93d465efc5"
        "7f8d917f0b385c6df265e77003b85102967486ed57db5c5ca170ba441427ed9afa684e",
    ),
    (
        1025,
        "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444f4c4a22b4b399155358a994e52bf255d"
        "e60035742ec71bd08ac275a1b51cc6bfe332b0ef84b409108cda080e6269ed4b3e2c3f7d722aa4cdc98d16deb554e562"
        "7be8f955c98e1d5f9565a9194cad0c4285f93700062d9595adb992ae68ff12800ab67a",
    ),
    (
        2048,
        "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a9a60bf80001410ec9eea6698cd537939"
        "fad4749edd484cb541aced55cd9bf54764d063f23f6f1e32e12958ba5cfeb1bf618ad094266d4fc3c968c2088f677454"
        "c288c67ba0dba337b9d91c7e1ba586dc9a5bc2d5e90c14f53a8863ac75655461cea8f9",
    ),
    (
        2049,
        "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b687952256303096de31d71d74103403822a2e0bc1eb19"
        "3e7aecc9643a76b7bbc0c9f9c52e8783aae98764ca468962b5c2ec92f0c74eb5448d519713e09413719431c802f948dd"
        "5d90425a4ecdadece9eb178d80f26efccae630734dff63340285adec2aed3b51073ad3",
    ),
    (
        3072,
        "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd29a3f6b0b978d6608335c09dc94ccf682"
        "f9951cdfc501bfe47b9c9189a6fc7b404d120258506341a6d802857322fbd20d3e5dae05b95c88793fa83db1cb08e7d8"
        "008d1599b6209d78336e24839724c191b2a52a80448306e0daa84a3fdb566661a37e11",
    ),
    (
        3073,
        "7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd39a27ae3b79d68d89da9bf25bc27139ae"
        "65a324918a5f9b7828181e52cf373c84f35b639b7fccbb985b6f2fa56aea0c18f531203497b8bbd3a07ceb5926f1cab7"
        "4d14bd66486d9a91eba99059a98bd1cd25876b2af5a76c3e9eed554ed72ea952b603bf",
    ),
    (
        4096,
        "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e9690289e9409ddb1b99768eafe1623da896"
        "faf7e1114bebeadc1be30829b6f8af707d85c298f4f0ff4d9438aef948335612ae921e76d411c3a9111df62d27eaf871"
        "959ae0062b5492a0feb98ef3ed4af277f5395172dbe5c311918ea0074ce0036454f620",
    ),
    (
        4097,
        "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb99505f91b0b5600a11251652eacfa9497b3"
        "1cd3c409ce2e45cfe6c0a016967316c426bd26f619eab5d70af9a418b845c608840390f361630bd497b1ab4401931635"
        "7c61dbe091ce72fc16dc340ac3d6e009e050b3adac4b5b2c92e722cffdc46501531956",
    ),
    (
        5120,
        "9cadc15fed8b5d854562b26a9536d9707cadeda9b143978f319ab34230535833acc61c8fdc114a2010ce8038c853e121"
        "e1544985133fccdd0a2d507e8e615e611e9a0ba4f47915f49e53d721816a9198e8b30f12d20ec3689989175f1bf7a300"
        "eee0d9321fad8da232ece6efb8e9fd81b42ad161f6b9550a069e66b11b40487a5f5059",
    ),
    (
        5121,
        "628bd2cb2004694adaab7bbd778a25df25c47b9d4155a55f8fbd79f2fe154cff96adaab0613a6146cdaabe498c3a94e5"
        "29d3fc1da2bd08edf54ed64d40dcd6777647eac51d8277d70219a9694334a68bc8f0f23e20b0ff70ada6f844542dfa32"
        "cd4204ca1846ef76d811cdb296f65e260227f477aa7aa008bac878f72257484f2b6c95",
    ),
    (
        6144,
        "3e2e5b74e048f3add6d21faab3f83aa44d3b2278afb83b80b3c35164ebeca2054d742022da6fdda444ebc384b04a54c3"
        "ac5839b49da7d39f6d8a9db03deab32aade156c1c0311e9b3435cde0ddba0dce7b26a376cad121294b689193508dd631"
        "51603c6ddb866ad16c2ee41585d1633a2cea093bea714f4c5d6b903522045b20395c83",
    ),
    (
        6145,
        "f1323a8631446cc50536a9f705ee5cb619424d46887f3c376c695b70e0f0507f18a2cfdd73c6e39dd75ce7c1c6e3ef23"
        "8fd54465f053b25d21044ccb2093beb015015532b108313b5829c3621ce324b8e14229091b7c93f32db2e4e63126a377"
        "d2a63a3597997d4f1cba59309cb4af240ba70cebff9a23d5e3ff0cdae2cfd54e070022",
    ),
    (
        7168,
        "61da957ec2499a95d6b8023e2b0e604ec7f6b50e80a9678b89d2628e99ada77a5707c321c83361793b9af62a40f43b52"
        "3df1c8633cecb4cd14d00bdc79c78fca5165b863893f6d38b02ff7236c5a9a8ad2dba87d24c547cab046c29fc5bc1ed1"
        "42e1de4763613bb162a5a538e6ef05ed05199d751f9eb58d332791b8d73fb74e4fce95",
    ),
    (
        7169,
        "a003fc7a51754a9b3c7fae0367ab3d782dccf28855a03d435f8cfe74605e781798a8b20534be1ca9eb2ae2df3fae2ea6"
        "0e48c6fb0b850b1385b5de0fe460dbe9d9f9b0d8db4435da75c601156df9d047f4ede008732eb17adc05d96180f8a735"
        "48522840779e6062d643b79478a6e8dbce68927f36ebf676ffa7d72d5f68f050b119c8",
    ),
    (
        8192,
        "aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a635fe51a27db045a567c1ad51be5aa34c0"
        "1c6651c4d9b5b5ac5d0fd58cf18dd61a47778566b797a8c67df7b1d60b97b19288d2d877bb2df417ace009dcb0241ca1"
        "257d62712b6a4043b4ff33f690d849da91ea3bf711ed583cb7b7a7da2839ba71309bbf",
    ),
    (
        8193,
        "bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3bb2282aa69be089359ea1154b9a9286c4"
        "a56af4de975a9aa4a5c497654914d279bea60bb6d2cf7225a2fa0ff5ef56bbe4b149f3ed15860f78b4e2ad04e158e375"
        "c1e0c0b551cd7dfc82f1b155c11b6b3ed51ec9edb30d133653bb5709d1dbd55f4e1ff6",
    ),
    (
        16384,
        "f875d6646de28985646f34ee13be9a576fd515f76b5b0a26bb324735041ddde49d764c270176e53e97bdffa58d549073"
        "f2c660be0e81293767ed4e4929f9ad34bbb39a529334c57c4a381ffd2a6d4bfdbf1482651b172aa883cc13408fa67758"
        "a3e47503f93f87720a3177325f7823251b85275f64636a8f1d599c2e49722f42e93893",
    ),
    (
        31744,
        "62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47860cc51f2b0c28a7b77304bd55fe73af"
        "663c02d3f52ea053ba43431ca5bab7bfea2f5e9d7121770d88f70ae9649ea713087d1914f7f312147e247f87eb2d4ffe"
        "f0ac978bf7b6579d57d533355aa20b8b77b13fd09748728a5cc327a8ec470f4013226f",
    ),
    (
        102400,
        "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085e01c59dab908c04c3342b816941a26d6"
        "9c2605ebee5ec5291cc55e15b76146e6745f0601156c3596cb75065a9c57f35585a52e1ac70f69131c23d611ce11ee4a"
        "b1ec2c009012d236648e77be9295dd0426f29b764d65de58eb7d01dd42248204f45f8e",
    ),
)


def official_input(length: int) -> bytes:
    """The vectors' input rule: the repeating byte sequence `0, 1, ..., 250`."""
    return bytes(i % 251 for i in range(length))

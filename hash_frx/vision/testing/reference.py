# Copyright 2026 The hash-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A plain-Python Vision permutation, as the oracle the frx one is held to.

Transcribed from the designers' reference implementation,
IrreducibleOSS/binius-models (Apache-2.0) @ 7ac5ad72 — the tower arithmetic
from `binius_models/finite_fields/tower.py` (class FanPaarTowerField), the
permutation, S-box, and key-schedule recurrence from
`binius_models/hashes/vision/vision.py` (Vision.encrypt / Vision.sbox /
Vision.update_key_schedule); the same structure as the Marvellous authors'
KULeuven-COSIC/Marvellous `instance_generator.sage` (Vision.BlockCipher)
@ b265d9a9. Everything runs over arbitrary-precision Python ints, sharing
neither the dtype arithmetic nor the single-kernel authoring rules that shape
`vision.py` — an oracle that shared those would fail the same way the thing it
checks does.

The oracle is itself anchored: `reference_test` holds `vision_permutation`
over the shipped Mark-32 tables to the plaintext/ciphertext vectors published
in binius-models (`test_vision32b.py`, transcribed below) — data produced by a
third implementation (the repo's Sage generator), so a shared misreading here
and in `vision.py` cannot look like agreement. The generator inputs
(`INITIAL_CONSTANT`, `CONSTANTS_MATRIX`, `CONSTANTS_CONSTANT`) live only here:
the shipped parameter surface carries the EXPANDED schedule, and
`expand_round_keys` is what re-derives it from them.
"""

from __future__ import annotations

import operator
from functools import reduce


def multiply(a: int, b: int, bits: int = 32) -> int:
    """Fan-Paar tower multiply on `bits`-wide elements (2x2 Karatsuba per
    level; tower modulus X_k^2 + X_{k-1}*X_k + 1)."""
    if bits == 1:
        return a & b
    half = bits // 2
    mask = (1 << half) - 1
    a0, a1 = a & mask, a >> half
    b0, b1 = b & mask, b >> half
    z0 = multiply(a0, b0, half)
    z2 = multiply(a1, b1, half)
    z1 = multiply(a0 ^ a1, b0 ^ b1, half) ^ z0 ^ z2
    z2a = _multiply_alpha(z2, half)
    return (z0 ^ z2) | ((z1 ^ z2a) << half)


def _multiply_alpha(a: int, bits: int) -> int:
    """Multiply by the level generator X_{k-1} (binius `_multiply_alpha`)."""
    if bits == 1:
        return a
    half = bits // 2
    mask = (1 << half) - 1
    a0, a1 = a & mask, a >> half
    z1 = _multiply_alpha(a1, half)
    return a1 | ((a0 ^ z1) << half)


def square(a: int, bits: int = 32) -> int:
    return multiply(a, a, bits)


def inverse(a: int, bits: int = 32) -> int:
    """Fan-Paar inversion (binius `fanpaar_inverse_recursive`). Raises on 0 —
    the 0 |-> 0 S-box convention lives in `sbox`, as it does in the binius
    reference (`Vision.sbox`)."""
    if a == 0:
        raise ZeroDivisionError("inverting zero")
    if bits == 1:
        return a
    half = bits // 2
    mask = (1 << half) - 1
    if a >> half == 0:
        return inverse(a, half)
    a0, a1 = a & mask, a >> half
    intermediate = a0 ^ _multiply_alpha(a1, half)
    delta = multiply(a0, intermediate, half) ^ square(a1, half)
    delta_inv = inverse(delta, half)
    return multiply(delta_inv, intermediate, half) | (
        multiply(delta_inv, a1, half) << half
    )


def evaluate_affine(coeffs: list[int], x: int, bits: int = 32) -> int:
    """Evaluate the F2-linearized affine polynomial at `x`: `coeffs[0] +
    sum_j coeffs[1 + j] * x**(2**j)` — constant-FIRST, the `VisionParams`
    layout (binius-models stores the constant last), so at least a constant
    and one linear coefficient (the >= 2 `VisionParams` requires)."""
    result = coeffs[0] ^ multiply(coeffs[1], x, bits)
    power = x
    for c in coeffs[2:]:
        power = square(power, bits)
        result ^= multiply(c, power, bits)
    return result


def sbox(x: int, coeffs: list[int], bits: int = 32) -> int:
    """One Vision S-box: the affine polynomial after the 0 |-> 0 inverse
    (the paper's x^(2^n - 2))."""
    return evaluate_affine(coeffs, 0 if x == 0 else inverse(x, bits), bits)


def matrix_vector(
    matrix: list[list[int]], vector: list[int], bits: int = 32
) -> list[int]:
    return [
        reduce(operator.xor, (multiply(m, v, bits) for m, v in zip(row, vector)))
        for row in matrix
    ]


def _step(
    state: list[int],
    injected: list[int],
    r: int,
    b: list[int],
    b_inv: list[int],
    mds: list[list[int]],
    bits: int,
) -> list[int]:
    """One Vision step `r` on `state`: S-box (`B^{-1}` on even steps, `B` on
    odd) -> MDS -> add `injected`. The data path and the key schedule run the
    SAME step — sharing it is what keeps the even/odd polarity defined once."""
    out = [sbox(x, b_inv if r % 2 == 0 else b, bits) for x in state]
    out = matrix_vector(mds, out, bits)
    return [a ^ k for a, k in zip(out, injected)]


def expand_round_keys(
    initial_constant: list[int],
    constants_matrix: list[list[int]],
    constants_constant: list[int],
    b: list[int],
    b_inv: list[int],
    mds: list[list[int]],
    rounds: int,
    bits: int = 32,
) -> list[list[int]]:
    """The zero-master-key schedule (binius `update_key_schedule` with the
    all-zero key; the paper's key path with `Ks = Cs`): the key state runs the
    same two-step rounds as the data path, and the injection follows the
    affine recurrence `ki <- constants_matrix @ ki + constants_constant`.
    Returns the 2*rounds + 1 step keys K_0..K_{2N}."""
    injection = list(initial_constant)
    state = list(initial_constant)
    keys = [list(state)]
    for r in range(2 * rounds):
        injection = [
            a ^ c
            for a, c in zip(
                matrix_vector(constants_matrix, injection, bits), constants_constant
            )
        ]
        state = _step(state, injection, r, b, b_inv, mds, bits)
        keys.append(list(state))
    return keys


def vision_permutation(
    state: list[int],
    b: list[int],
    b_inv: list[int],
    mds: list[list[int]],
    round_keys: list[list[int]],
    bits: int = 32,
) -> list[int]:
    """The unkeyed Vision permutation (binius `encrypt` under the zero-key
    schedule; the paper's Algorithm 1): pre-whiten with K_0, then 2N steps of
    S-box (B^{-1} on even steps, B on odd) -> MDS -> key injection."""
    s = [x ^ k for x, k in zip(state, round_keys[0])]
    for r, keys in enumerate(round_keys[1:]):
        s = _step(s, keys, r, b, b_inv, mds, bits)
    return s


# --- Vision Mark-32 anchors, transcribed from binius-models @ 7ac5ad72 ---
#
# The generator inputs behind the shipped `_ROUND_KEYS`
# (binius_models/hashes/vision/vision.py, Vision32b.get_constants), sampled by
# the Marvellous SHAKE-256 procedure (the repo's generate_vision_instance.sage;
# https://eprint.iacr.org/2019/426, Section 4.1).
# fmt: off
INITIAL_CONSTANT = (
    0x545E66A7, 0x073FDD58, 0x84362677, 0x95FE8565,
    0x06269CD8, 0x9C17909E, 0xF1F0ADEE, 0x2694C698,
    0x94B2788F, 0x5EAC14AD, 0x21677A78, 0x5755730B,
    0x37CEF9CF, 0x2FB31FFE, 0xFC0082EC, 0x609C12F0,
    0x102769EE, 0x4732860D, 0xF97935E0, 0x36E77C02,
    0xBA9E70DF, 0x67B701D7, 0x829D77A4, 0xF6EC454D,
)

CONSTANTS_MATRIX = (
    (
        0x1108D307, 0xEC7A9199, 0x5DE33E41, 0xFF840AD4,
        0x3A924504, 0x6F8A208B, 0xDD362871, 0xC36A24C2,
        0xEDA3984E, 0x9E3751A2, 0xD0004B5E, 0x58E7A69E,
        0x11F7D42A, 0x62F17410, 0xCA8C7DBA, 0x6FD43901,
        0xE4F6985D, 0x6890B1DE, 0x65D8652F, 0x7900081C,
        0x6237EC32, 0x2D1B4D07, 0xDEB98A71, 0xE42B1A0A,
    ),
    (
        0xE0A55F9F, 0x1355DE11, 0x3D80BC0F, 0x485E159C,
        0xB7C1DA2A, 0x26A36B69, 0xA67B6D9C, 0xCF4D3BA4,
        0x91B6298D, 0x500FD329, 0x31EB526D, 0x850B601B,
        0xB07514BE, 0x7F242D97, 0x8787ED13, 0xA33E30E5,
        0x5EB3072D, 0xB6B253F0, 0x3C97E604, 0xBE7AA830,
        0xA37D06FA, 0xCC988002, 0x56BA30EF, 0x1041EE88,
    ),
    (
        0x6BC512A6, 0x075EBB93, 0xA4538DA6, 0xB71F2038,
        0xCEFFBA6B, 0xBBE96C4C, 0xA1E13CE3, 0x6C87EF2A,
        0xA3139F62, 0x2C4ACF0D, 0x6F4DE460, 0x1289208B,
        0x106EB3BB, 0xCA9B9171, 0x1B9D4248, 0x3DC38CC9,
        0x840C1893, 0x831048AC, 0x47667736, 0x7522713B,
        0x9DB55119, 0xDA6324B4, 0x720B04CF, 0x2CAED5BD,
    ),
    (
        0xCC8F36CC, 0x1FC0C9AE, 0x234BE4A7, 0x4565E621,
        0x3099EE6B, 0xF9ECFBAA, 0x14B28B3E, 0x5E293305,
        0x16122BC3, 0x87078441, 0x328C8192, 0xC75C86D1,
        0xCC141723, 0x141841AE, 0x0FEC8191, 0x783F477D,
        0xC5518D2A, 0xDFC6552B, 0x6D7EF44C, 0x1F83E675,
        0x1ABE750B, 0x1F1EBA62, 0x241F5967, 0x28FAA527,
    ),
    (
        0xB1CA576B, 0x09B1E1DC, 0xADE0D3AE, 0xC3CE6A43,
        0x6D4A4B6C, 0x38821CBE, 0x3B7060ED, 0xD645DBD2,
        0x8C0C9168, 0xEF6C6449, 0x4481F6CA, 0x896F624E,
        0x84747516, 0x75DDE272, 0x8D0AE03A, 0xE15A828C,
        0xF3AD2CCB, 0x98557F60, 0x33E0F702, 0xD333D762,
        0x04C66D30, 0xE4958B14, 0x7F8B569D, 0x670D9A9B,
    ),
    (
        0x90509D7F, 0x9ADCEFD0, 0x12741F97, 0xC91795AA,
        0x4ECDF02F, 0x8D9C91E3, 0xF3421547, 0xB963E5DE,
        0xA281EC4C, 0xB3524139, 0x6777299D, 0x23D1675D,
        0x9DB2D9BB, 0x261B5DC5, 0x0841E27C, 0xEAB73E4A,
        0x8D6EAD23, 0x607FCDA2, 0x72C8217E, 0x46F73848,
        0x868E11AA, 0x11EEF4A2, 0xF3AB54C4, 0x2CD05BA8,
    ),
    (
        0xB4E8B361, 0xB1B73CAE, 0x4AF26138, 0xA2706C9B,
        0x25B86E94, 0x6D3A3DD3, 0x24002C84, 0xA9C0C35E,
        0xDAF48349, 0x9AB33A67, 0x0CE87D3A, 0x66923746,
        0x89C2B0FC, 0x600B1E85, 0x974330BC, 0x3FF03A2E,
        0xDCE09562, 0xC2F1A8B4, 0x068B0B8F, 0x95F687EF,
        0x9C90441D, 0x17C68AF8, 0xBBE0AE8C, 0x0977220A,
    ),
    (
        0xA7F36139, 0x56887E90, 0x6EA48A38, 0xD0DC602B,
        0x09F0FD57, 0xC61523CF, 0x109B331E, 0xED9DBE39,
        0xCC0345C7, 0x5F57681A, 0x37670E8A, 0xFAB40121,
        0x298F80AF, 0x2E7F4E44, 0x2944D394, 0x6CF97518,
        0x3E9E61A9, 0x5F470D66, 0x3C213AA7, 0xF3B16F32,
        0x8575C1FD, 0x77164B13, 0xE2A7BB94, 0xC1DA2A1E,
    ),
    (
        0xFBD2EB5B, 0x94FDE9C9, 0xF94D4695, 0x63A8E16F,
        0x63C3B096, 0x4299383C, 0x8BEE7759, 0x64D23345,
        0x661204B6, 0x9B63D4A7, 0x61CA24C8, 0xC72FF155,
        0x43EAF867, 0xC26867BC, 0x13498541, 0x2E6031FD,
        0xC5A37CC5, 0x11C7BCBB, 0x5B5A2B1D, 0xCCEE17EF,
        0x586410E4, 0xF048EECB, 0x658AA18A, 0xC8DF0F21,
    ),
    (
        0xD5ECB636, 0x3D42499B, 0x565AD8A6, 0x9E5430BE,
        0xB0E6F553, 0x7DC33549, 0x6BC7E8DB, 0xB02AA685,
        0xB0094065, 0x57BC19A6, 0xAA55A612, 0x323C1958,
        0xAB994C0A, 0x26B9471A, 0xCD8DF824, 0x00A0CF18,
        0x483F95D8, 0xB2DD0490, 0x65347EC6, 0x3B078220,
        0x2FC83CFF, 0xFE238E9A, 0x0DF48724, 0x6E4B21EA,
    ),
    (
        0xDEF1960B, 0x3D3FDCE8, 0xD13A93A0, 0x4153E6C0,
        0x41FAAC96, 0xF861CA1E, 0x66D4B676, 0xC7BA3E6B,
        0xE5E86937, 0x12984A18, 0x4AB68320, 0xDC998665,
        0x865CE15E, 0xF40C2FCE, 0x5D92AE8C, 0x258DC776,
        0x37F6C58D, 0x2C5BFF1C, 0x3CC5D91A, 0xF8244009,
        0xC5AE7B36, 0x4A78DAA8, 0x3E62CEBD, 0x5FB85623,
    ),
    (
        0x22AA9A5C, 0x4AFAD2CC, 0x1392A38A, 0x3E5A0558,
        0xD6E884AA, 0x40D9E182, 0xD0E89E26, 0x210569D2,
        0x9395E023, 0xB429FEDC, 0xEE5C4A30, 0x41472CAA,
        0xB7525451, 0x021A33E7, 0xB1B6247C, 0x32C11C47,
        0x7F681C6B, 0xBE3500E6, 0x0F5AD41F, 0x1C83C8D4,
        0x5FA23114, 0xDF8229B1, 0x23B414BD, 0x4EC68613,
    ),
    (
        0xA29ABDDA, 0x3350412F, 0x3C48F63A, 0x583CC538,
        0x7012EFEF, 0x7DC91145, 0xF10AB4D4, 0xDDB56653,
        0xC8117349, 0xF0A92831, 0x6DDE832F, 0x0BE4EF8A,
        0xEE0C5BF9, 0x0AFB86D6, 0xC117DA33, 0xEBFF60AA,
        0x07413FA6, 0x420B6D83, 0x275160B7, 0x8FB66D03,
        0xC495D852, 0xDC1BE7D8, 0xAACB1388, 0xA8402D60,
    ),
    (
        0x6AF2D77F, 0x8FE4AED3, 0x948678D5, 0x76595E5B,
        0x45252576, 0xBEF2EA2D, 0xFB71E252, 0x0ED8FCF4,
        0x7B83686E, 0x18A3A132, 0xF745464C, 0x966597DB,
        0xFA4E2C6B, 0xA8458CC0, 0x6F084E4F, 0x41142CAF,
        0xE5697E4D, 0x98F80A8F, 0x7A743A6D, 0xA2D90DDA,
        0xD5131E90, 0x8B5E88E7, 0xF44CE732, 0xC74EB1C6,
    ),
    (
        0xBEF4A961, 0xC0AE671A, 0x296D9748, 0x9A2D4725,
        0x1E730BB3, 0xC0B0F235, 0x2797197D, 0x808091FC,
        0x006AC229, 0x2B3C8DB8, 0x3485F9D7, 0x521DE28D,
        0xDCFAFB34, 0x4785422C, 0x060E3619, 0x33102F36,
        0x544CA95F, 0x4D65F380, 0x84826CBF, 0xD7EF3944,
        0x7DFBB099, 0xB58AD793, 0x25D9A483, 0x08580A95,
    ),
    (
        0x5CB75041, 0x6D2DB475, 0xF41BA3E4, 0x838F8B59,
        0x60EA1AC5, 0x29182E40, 0x70B62618, 0x58C8D79C,
        0xA85B39E8, 0x6AB56677, 0x6634E86E, 0x5C041113,
        0x701EFF30, 0xE6DFF676, 0x235CBD33, 0x409599FA,
        0x81D2B0AA, 0x66BA001B, 0x8A5F2D86, 0x0F071B47,
        0x085F875F, 0x3C840120, 0x66B6140E, 0x9E9CB44C,
    ),
    (
        0xACF1793D, 0xDAA26B90, 0x19A8968B, 0x5C8E9AE0,
        0x81850159, 0xDC3F1E80, 0x13D5CE2E, 0x1942FB28,
        0xDA17C113, 0x597A6465, 0x0DA1919E, 0x43912545,
        0x01F84069, 0xC80F0280, 0xCACA8D04, 0x05CDEFC9,
        0x2898131C, 0xC58F9382, 0xD52038D3, 0x13E38635,
        0xE31FC907, 0xB1366729, 0xEABEA2AA, 0x6867ADF0,
    ),
    (
        0x3F503F05, 0x76B1170D, 0x6E8803AF, 0x841D8A7E,
        0x5125BE27, 0xBCD07A3C, 0xC17C40B0, 0x224B57EA,
        0x62A9B246, 0x9B3A73A7, 0xF033D5D1, 0x59206F4E,
        0xD8EAA375, 0xBEC2FF9F, 0xF2FD1A0B, 0xEAB1200F,
        0xE20E0711, 0x64DF8B56, 0xDC2A2973, 0x18CB0C1C,
        0xF4F82E5B, 0x732F1184, 0x42985BE7, 0x88F436CD,
    ),
    (
        0xB20F06C1, 0xF479F2BB, 0xE1252249, 0x61D80CBB,
        0x80FAB6A4, 0x503B8D9D, 0x4889D6BB, 0x61366C41,
        0x03B4467A, 0xD6114932, 0x6E6E88EF, 0xEAC3C262,
        0x8AA3CDD4, 0xE535972F, 0x142F485A, 0x7857D2ED,
        0x4D60F969, 0x3B0B5E1D, 0x0A9F23AC, 0x0A235AAC,
        0x36DE7127, 0x34532CEA, 0xCE70AAB4, 0x20D6F24F,
    ),
    (
        0x22543427, 0xCB071894, 0x3DC00D69, 0xC2B7BA33,
        0xB2A9A12C, 0xBED6A5A7, 0x06240A69, 0xA111AEFA,
        0x3ED567A0, 0x4F617033, 0x56E78466, 0x79027E36,
        0x74F32F09, 0x95804D70, 0xBEC55833, 0x063F1B9E,
        0x706AB0FF, 0x05329E54, 0x04E56D4D, 0x1A1D0BC4,
        0x699CFAC3, 0xA9D59C23, 0xB7BFAB35, 0x3EA262F9,
    ),
    (
        0x81F18F94, 0xF8489F72, 0x4DB46C59, 0x46906502,
        0x408A6192, 0x9F91B41B, 0xFCCC6032, 0x30F5E7B3,
        0x83F05261, 0xDCFA112D, 0xF5618E3D, 0xA7F171D6,
        0x272B9D8D, 0x34D2ACD6, 0xE4900F75, 0x8A6B0C59,
        0x627482E6, 0x65E9F3A3, 0x59AA4BD3, 0xC5807F4E,
        0x31754A05, 0xE06EFD2F, 0xE0B4F9B3, 0x7D24C239,
    ),
    (
        0xBE882322, 0xE9D84AE8, 0x716C9822, 0x5726F8C8,
        0x13286D00, 0x35DF225D, 0x29EF64AF, 0x8B3FD194,
        0xEB076443, 0x7F9B28C8, 0x5B9A0F70, 0xCBD606E3,
        0xBD198078, 0x07B00DE2, 0xCB3BFE5F, 0x39F12198,
        0xF2D82C9F, 0x34516B45, 0x9624FD4A, 0x3EA5AACE,
        0x8B2F0A0D, 0x2999DC3E, 0x6409FBAC, 0x3FC5D430,
    ),
    (
        0x70011997, 0x929A2C0F, 0x584CE413, 0xBFE8A9E8,
        0x23818E4E, 0xDAB3C865, 0x19F97044, 0x5CDDA0F1,
        0x14CE4DD0, 0x6E732DB0, 0x6D7A09AB, 0x5CD27095,
        0xB82B5FEF, 0x4E12AD28, 0x7F89E69F, 0xA18FA21C,
        0xD08270F2, 0xFDD7332D, 0x89019F23, 0xB0622A67,
        0x8DC3EA5C, 0xFFA3980A, 0x23169BFD, 0xB9F3541F,
    ),
    (
        0x4CB2D0DB, 0xE8CCB7E0, 0xC4BD5AD5, 0xBCA45BA8,
        0xF2847642, 0x44D2751C, 0x5F669322, 0x68AC2B6B,
        0xE8755FC7, 0x89C4435D, 0x8F7FF92C, 0xA867E3DD,
        0x26276ED6, 0x65DA94F5, 0xC091356D, 0x3C0D8CDD,
        0xC4A84214, 0x793D79F2, 0x6FBBF6ED, 0xCD20CA13,
        0x0FD8CFF0, 0x2853F0A1, 0x5EBDFCD1, 0xC7CCF32D,
    ),
)

CONSTANTS_CONSTANT = (
    0x8D36811E, 0x761F6307, 0x2E7061C4, 0x091C5558,
    0xE2E1A67C, 0xAC54F6A2, 0x60E1BD95, 0xF8307596,
    0x9120BC98, 0xA8F3FE14, 0xCE4BDC15, 0xB0BA0B6D,
    0xA901E2CC, 0x810A93B1, 0xEDCF6691, 0x04DA059C,
    0x58C77E1F, 0x4A66463B, 0x5D3A8F21, 0xED1A16D4,
    0xFB12937D, 0x1AC12409, 0xFB031873, 0x587C2F14,
)

# The published permutation vectors (binius_models/hashes/vision/
# test_vision32b.py, ids "zero" and "random"): zero-key encrypt == the
# permutation, so these pin every table end to end.
MARK32_ZERO_PLAINTEXT = (0x00000000,) * 24

MARK32_ZERO_CIPHERTEXT = (
    0xC12D2BD5, 0x7B69D9D2, 0x5D706D99, 0x24AC5453,
    0xFD9A0F9F, 0xFB7A3BE4, 0xD935A59F, 0x691667D7,
    0xB89EE8F9, 0x6FCA1714, 0x530CAEE0, 0x3F884B18,
    0x6D6C9A40, 0x940B294C, 0xC9976C5D, 0xB3029E46,
    0x8F5DD6AB, 0x8ADC785E, 0x15F0CBA2, 0x32CD00A1,
    0x5C4AD8B5, 0xD9CE2F5B, 0x8E9D1B5B, 0xE70A9E14,
)

MARK32_RANDOM_PLAINTEXT = (
    0xC45CAB27, 0x9B735F2E, 0x090D2A66, 0x140B8FF3,
    0xE7778EA8, 0xC19AE0A1, 0xC2FFA056, 0x03B13619,
    0xC7A95DA5, 0xE456A375, 0xD1D0A869, 0xE80CE171,
    0x12C89A69, 0xADA9072E, 0xC7A50CCF, 0xB9CE31BD,
    0xF7E16524, 0xB77C94B4, 0xD98D20AA, 0x8BAD80A7,
    0xE5F60FED, 0x9E082CFB, 0x1747575C, 0xBCACCAEB,
)

MARK32_RANDOM_CIPHERTEXT = (
    0xBBB328CC, 0x71D93B65, 0x473A5EF4, 0x5185446E,
    0xBD3F85C5, 0x1FABD2FB, 0xA62CD343, 0xFE07C31F,
    0x4475C510, 0xEE6B377D, 0x7BEB8619, 0x28A605BF,
    0x3D184345, 0x86B5F751, 0x1C02F436, 0x40562D71,
    0xEF782B79, 0x77BB5244, 0x1458083C, 0x7306F3AE,
    0x12A17146, 0x8202E4C3, 0x0381F579, 0x27628DC6,
)
# fmt: on

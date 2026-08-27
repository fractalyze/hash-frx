// Differential vector generator for hash-frx#261.
//
// Emits, for one shared Poseidon configuration, the outputs of a fixed set of
// absorb/squeeze scripts run through BOTH ark-sponge 0.3 and
// ark-crypto-primitives 0.5. The two differ on exactly two axes (rate position
// in the state; the squeeze spill-permute predicate), so any difference in the
// output is attributable to the sponge schedule.
//
// `alpha = 1` deliberately, and it is load-bearing. ark-sponge 0.3 applies the
// partial-round S-box to the LAST lane and ark-crypto-primitives 0.5 to lane 0
// — the one place the two disagree inside the permutation itself. With
// `alpha = 1` the S-box is the identity, so a partial round is a full round and
// that disagreement disappears: both releases compute the same permutation, and
// any difference in output is attributable to the sponge schedule alone.
//
// `partial_rounds` is 1 rather than 0 because hash-frx's dedicated Poseidon XLA
// emitter rejects a zero partial-round count ("unparsable composite.attributes").
//
// The permutation is still a real one for this purpose: every round is an ARC
// followed by an invertible non-symmetric MDS, so it is an affine map that is
// not the identity and does not commute with a rate read. A skipped or extra
// permute is fully visible. It is not a cryptographic configuration and is not
// meant to be one — the permutation's correctness is
// `poseidon/testing/poseidon_test.py`'s question.
//
// ark-sponge 0.3 hardcodes rate = 2, capacity = 1, width = 3, so the shared
// config follows it.

use ark_bn254_03::Fr as Fr03;
use ark_bn254_05::Fr as Fr05;
use ark_ff_03::{BigInteger as _, PrimeField as _};
use ark_ff_05::BigInteger as _;
use ark_ff_05::PrimeField as _;

use ark_cp_05::sponge::poseidon::{PoseidonConfig, PoseidonSponge as Sponge05};
use ark_cp_05::sponge::{
    CryptographicSponge as CS05, FieldBasedCryptographicSponge as FBCS05,
};
use ark_sponge_03::poseidon::{PoseidonParameters, PoseidonSponge as Sponge03};
use ark_sponge_03::{CryptographicSponge as CS03, FieldBasedCryptographicSponge as FBCS03};

const WIDTH: usize = 3;
const RATE: usize = 2;
const CAPACITY: usize = 1;
const ALPHA: u64 = 1;
const FULL_ROUNDS: u32 = 8;
const PARTIAL_ROUNDS: u32 = 1;

// Small canonical ints; the values are arbitrary but must be identical on both
// sides and reproducible in Python. A non-symmetric MDS is deliberate — a
// symmetric one can hide a lane-order mistake.
const MDS: [[u64; WIDTH]; WIDTH] = [[2, 3, 5], [7, 11, 13], [17, 19, 23]];

fn ark_row(round: usize) -> [u64; WIDTH] {
    // Deterministic, distinct per (round, lane), and small.
    let b = (round as u64 + 1) * 100;
    [b + 1, b + 2, b + 3]
}

/// One absorb/squeeze script. `Absorb(n)` absorbs n distinct elements;
/// `Squeeze(n)` squeezes n and records them.
#[derive(Clone, Copy, Debug)]
enum Step {
    Absorb(usize),
    Squeeze(usize),
}

fn scripts() -> Vec<(&'static str, Vec<Step>)> {
    use Step::*;
    vec![
        ("absorb1_squeeze1", vec![Absorb(1), Squeeze(1)]),
        ("absorb2_squeeze2", vec![Absorb(2), Squeeze(2)]),
        ("absorb3_squeeze2", vec![Absorb(3), Squeeze(2)]),
        ("absorb5_squeeze3", vec![Absorb(5), Squeeze(3)]),
        // THE QUIRK CASE. rate = 2: squeeze 1 (leaves next_squeeze_index = 1),
        // then squeeze exactly `rate`. 0.3 tests `len != rate` BEFORE advancing
        // the slice, so it skips the spill permute and re-reads state[0] — a
        // lane the first squeeze already returned. 0.5 advances first, finds
        // the remainder non-empty, and permutes.
        ("squeeze1_then_squeeze_rate", vec![Squeeze(1), Squeeze(2)]),
        (
            "absorb1_squeeze1_then_squeeze_rate",
            vec![Absorb(1), Squeeze(1), Squeeze(2)],
        ),
        // Same shape one block further out.
        ("squeeze1_then_squeeze4", vec![Squeeze(1), Squeeze(4)]),
        ("squeeze5", vec![Squeeze(5)]),
        ("squeeze1_x3", vec![Squeeze(1), Squeeze(1), Squeeze(1)]),
        (
            "interleaved",
            vec![Absorb(2), Squeeze(1), Absorb(1), Squeeze(3)],
        ),
    ]
}

fn run_03(steps: &[Step]) -> Vec<String> {
    let mds: Vec<Vec<Fr03>> = MDS
        .iter()
        .map(|r| r.iter().map(|&v| Fr03::from(v)).collect())
        .collect();
    let ark: Vec<Vec<Fr03>> = (0..(FULL_ROUNDS + PARTIAL_ROUNDS) as usize)
        .map(|i| ark_row(i).iter().map(|&v| Fr03::from(v)).collect())
        .collect();
    let params = PoseidonParameters::new(FULL_ROUNDS, PARTIAL_ROUNDS, ALPHA, mds, ark);
    let mut sponge = Sponge03::<Fr03>::new(&params);

    let mut out = Vec::new();
    let mut next = 1u64;
    for step in steps {
        match *step {
            Step::Absorb(n) => {
                let elems: Vec<Fr03> = (0..n)
                    .map(|_| {
                        let v = Fr03::from(next);
                        next += 1;
                        v
                    })
                    .collect();
                sponge.absorb(&elems);
            }
            Step::Squeeze(n) => {
                for v in sponge.squeeze_native_field_elements(n) {
                    out.push(hex_le(&v.into_repr().to_bytes_le()));
                }
            }
        }
    }
    out
}

fn run_05(steps: &[Step]) -> Vec<String> {
    let mds: Vec<Vec<Fr05>> = MDS
        .iter()
        .map(|r| r.iter().map(|&v| Fr05::from(v)).collect())
        .collect();
    let ark: Vec<Vec<Fr05>> = (0..(FULL_ROUNDS + PARTIAL_ROUNDS) as usize)
        .map(|i| ark_row(i).iter().map(|&v| Fr05::from(v)).collect())
        .collect();
    let config = PoseidonConfig::new(
        FULL_ROUNDS as usize,
        PARTIAL_ROUNDS as usize,
        ALPHA,
        mds,
        ark,
        RATE,
        CAPACITY,
    );
    let mut sponge = Sponge05::<Fr05>::new(&config);

    let mut out = Vec::new();
    let mut next = 1u64;
    for step in steps {
        match *step {
            Step::Absorb(n) => {
                let elems: Vec<Fr05> = (0..n)
                    .map(|_| {
                        let v = Fr05::from(next);
                        next += 1;
                        v
                    })
                    .collect();
                sponge.absorb(&elems);
            }
            Step::Squeeze(n) => {
                for v in sponge.squeeze_native_field_elements(n) {
                    out.push(hex_le(&v.into_bigint().to_bytes_le()));
                }
            }
        }
    }
    out
}

fn hex_le(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn json_list(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("\"{s}\"")).collect();
    format!("[{}]", inner.join(", "))
}

fn main() {
    println!("{{");
    println!("  \"_generated_by\": \"hash_frx/adapter/testing/reference/arkvec (see its README)\",");
    println!("  \"width\": {WIDTH}, \"rate\": {RATE}, \"capacity\": {CAPACITY},");
    println!("  \"alpha\": {ALPHA}, \"full_rounds\": {FULL_ROUNDS}, \"partial_rounds\": {PARTIAL_ROUNDS},");
    let mds_rows: Vec<String> = MDS
        .iter()
        .map(|r| format!("[{}]", r.map(|v| v.to_string()).join(", ")))
        .collect();
    println!("  \"mds\": [{}],", mds_rows.join(", "));
    let ark_rows: Vec<String> = (0..(FULL_ROUNDS + PARTIAL_ROUNDS) as usize)
        .map(|i| format!("[{}]", ark_row(i).map(|v| v.to_string()).join(", ")))
        .collect();
    println!("  \"round_constants\": [{}],", ark_rows.join(", "));
    println!("  \"scripts\": {{");

    let all = scripts();
    for (idx, (name, steps)) in all.iter().enumerate() {
        let steps_json: Vec<String> = steps
            .iter()
            .map(|s| match s {
                Step::Absorb(n) => format!("[\"absorb\", {n}]"),
                Step::Squeeze(n) => format!("[\"squeeze\", {n}]"),
            })
            .collect();
        let v03 = run_03(steps);
        let v05 = run_05(steps);
        let comma = if idx + 1 == all.len() { "" } else { "," };
        println!("    \"{name}\": {{");
        println!("      \"steps\": [{}],", steps_json.join(", "));
        println!("      \"ark_0_3\": {},", json_list(&v03));
        println!("      \"ark_0_5\": {},", json_list(&v05));
        println!("      \"differs\": {}", v03 != v05);
        println!("    }}{comma}");
    }
    println!("  }}");
    println!("}}");
}

# LPSCounter — (6,3,2) search engine and near-miss instrument

Searches for counterexamples to the Lander–Parkin–Selfridge conjecture at k=6
in the form

    x^6 + y^6 + z^6 = a^6 + b^6        (positive integers)

and simultaneously instruments the search to reveal structure: ±1 near misses,
nearest-gap statistics on a deterministic sample, and (2,2) repeated pair sums.
The realistic outcomes, in order of likelihood: reproducible code + a new
certified search bound, new near misses, empirical gap statistics that tell us
where (or whether) to look next — and, at long odds, the counterexample.

## Search contract

- **Bound convention: exclusive.** All bases lie in `[1, B)`. This matches the
  "radius" semantics of powersums.jorisperrenet.com coverage claims
  (`0 <= x_i < R`).
- **Exact campaign** (`--near 0`): enumerates all *primitive* solutions with
  bases in `[1,B)`. Primitivity-derived pruning (class strides mod 42, pair
  admissibility) discards only tuples that provably cannot be primitive
  (verified exhaustively over the finite residue domains by `selftest`). Any
  non-primitive solution scales down to a primitive one with strictly smaller
  bases, so absence of primitive solutions excludes non-primitive ones within
  the bound. Byproduct: primitive (6,2,2) repeated pair sums.
- **Near campaign** (`--near 1`): full pair table, no pruning. Finds every
  solution and every ±1 near miss with bases in `[1,B)`, plus all (6,2,2)
  repeated pair sums. Class restrictions are *unsound* for near misses (e.g.
  1+1+1 = 1+1 + 1 violates them), hence the separate campaign.
- Both campaigns: a `2^-sampleShift` deterministic sample of (x,y) columns is
  probed unfiltered, recording a log2 histogram of the distance to the nearest
  pair sum and every candidate with `|lhs-rhs| <= 2^deltaBits` (`gap` records).
  In the exact campaign these statistics are measured against admissible pairs
  only (labeled `gapPopulation` in stats.json).
- Every reported identity is re-derived in exact unsigned 128-bit arithmetic
  before being written, and can be re-verified independently with `verify.py`
  (Python big-int arithmetic).
- **In-memory only.** The pair-key table (8 bytes × ~B²/2) must fit in RAM.
  Scaling beyond RAM is explicitly out of scope for v1.

## Prior art and implied coverage (audited 2026-08-09)

| System | Bound | Source / status |
|---|---|---|
| (6,3,2) exact, explicit | common sum ≤ 1.15×10^16 (max base ≈ 475) | Ekl, Math. Comp. 67 (1998) |
| (6,3,2) exact, IMPLIED | bases < 33,225 (via (6,3,3) coverage with a zero term, per the archive's 0 ≤ x_i convention) | Moore's (6,3,3) archive claim (contributor-reported) |
| (6,3,2; +1) misses, IMPLIED | bases < 33,225 (a +1 miss is a (6,3,3) solution with a 1^6 term) | same |
| (6,3,2; −1) misses | radius 17,800 (secondhand); a −1 miss is (6,4,2)-shaped, NOT covered by (6,3,3) | Moore via Piezas |
| (6,2,2) | sum ≤ 2^96 (max base ≈ 58,385) | Schoenfield, OEIS A046881 |

Notation trap: LPS/Ekl write our (6,3,2) as (6.2.3). Archive claims are
contributor-reported and not independently verified; implied-coverage rows
depend on the archive's zero-term convention and are labeled accordingly.

## Results to date (2026-08-09)

An INTERNALLY CROSS-VERIFIED (two oracles by the same author — not
independent third-party verification) exhaustive search found:

- No new (6,3,2;±1) identity for bases below 30,001, recovering all five
  known +1 identities and finding no −1 identity. Relative to prior work this
  is the first explicit archive-format coverage of the combined ±1 residuals
  and, on the −1 side, an extension of Moore's secondhand radius-17,800
  claim; the +1 side below 30,001 is implied by existing (6,3,3) coverage.
- No (6,3,2) solution with all bases below 50,001 (B=150,001 in progress) —
  ~1.5× past the implied (6,3,3)-derived frontier of 33,225, ~105× past
  Ekl's explicit 1998 bound (comparisons labeled, not conflated).
- The (2,2) byproduct scan found no repeated pair sums, consistent with (and
  below) Schoenfield's 2^96 bound; it is NOT a record and the admissibility
  argument for the exact campaign's pair filtering is spelled out in
  lps632.cpp before any (2,2) claim should be made from it.
- A deterministic 1/256-column SAMPLED residual corpus (millions of signed
  gap records, |lhs−rhs| ≤ 2^40) accompanies each run — a sampled instrument,
  never gap-coverage, and not claimed as a first without a literature audit.

## Build & run (Linux; g++ with __int128 and OpenMP)

```bash
g++ -O3 -march=native -fopenmp -o lps632 lps632.cpp
g++ -O1 -g -fsanitize=address,undefined -fopenmp -o lps632_asan lps632.cpp

./lps632 selftest            # must print ALL SELFTESTS PASSED
./lps632_asan selftest       # same, under ASan/UBSan
./lps632 bench --B 10001,20001,30001

# campaign 1 (exact):
./lps632 search --B 150001 --near 0 --outdir run_exact_150k
# campaign 2 (±1 near misses), after costing it via bench:
./lps632 search --B 50001 --near 1 --outdir run_near_50k

# resume after interruption (same parameters, config hash enforced):
./lps632 search --B 150001 --near 0 --outdir run_exact_150k --resume

# independent verification of everything reported:
python3 verify.py run_exact_150k/results.jsonl
```

Production bounds are chosen from `bench` extrapolation, never guessed.

## Validation battery (`selftest`)

- SHA-256 FIPS vectors (config/results hashing is vendored).
- `rootFloor` boundary properties, exhaustive at every exact power.
- Exhaustive class-mask soundness: all 42³ triple classes × all admissible
  pair classes mod 42 — every excluded class is congruence-impossible, every
  allowed class has a witness. Class density must equal 9/49 exactly.
- Exhaustive no-false-negative check of every chain filter over its full
  residue domain (both campaigns, all three probe offsets).
- Brute-force oracle equivalence at small B for K=2,3,4,6 (K=2 is dense with
  genuine hits), near and exact campaigns, including the gap stream and forced
  false-candidate storms via `--key-shift-extra`.
- Rediscovery gates: 147⁶+92⁶+71⁶ = 133⁶+132⁶+1 at B=148, and all five known
  Wroblewski near misses (and no exact hit) at B=1256.
- Crash/resume/merge: partial run + config-mismatch refusal + resume must
  reproduce the uninterrupted run's records bit-for-bit.

Deploy gate additionally cross-checks the engine against `oracle.py`
(independent Python implementation) via bit-exact JSONL diff at B=401.

## Output

`<outdir>/tiles/tile_*.jsonl` (atomic per-tile records), `manifest.txt`
(SHA-256 config header + fsync'd tile completions), `results.jsonl` (merged,
deduplicated), `results.jsonl.sha256`, `stats.json` (counters + gap histogram).
Record types: `exact`, `near` (diff = lhs−rhs = ±1), `dup22`, `gap`
(delta = lhs−rhs, sampled stream), `tilestats`.

## Publishing

Coverage claims go to powersums.jorisperrenet.com ("Report a search with no
new results"; category `(6, 3, 2; ±1)` for the near campaign — state which
residuals are complete in the notes). New identities go through "Contribute a
result" (server re-verifies exactly). Link this repository as the tool.

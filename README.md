# LPSCounter — a (6,3,2) search engine and near-miss instrument

An exhaustive search for counterexamples to the **Lander–Parkin–Selfridge
conjecture** at k=6, in the form

    x⁶ + y⁶ + z⁶ = a⁶ + b⁶        (all bases positive integers)

together with an instrument that measures *how close* the equation gets when it
fails: ±1 near misses, signed residual statistics, and (6,2,2) repeated pair
sums, all as byproducts of the same sweep.

No counterexample was found. The useful output is a certified search bound, an
explicit coverage claim for a residual nobody had searched, and a quantitative
congruence obstruction that explains the shape of the near-miss data.

> **Verification status.** Everything here is *internally cross-verified*: two
> independently written oracles (C++ and Python) by the same author, plus exact
> 128-bit re-derivation of every emitted record. That is not third-party
> verification. Artifacts are published so anyone can check.

## Results

**Exact search, bases in [1, 150001): no solutions.**
Completed 2026-08-14 in 4 d 20 h wall (3,081 CPU-hours), 4,687 tiles,
1.03×10¹⁴ filtered probes, 6,259,053 emitted records, **0 verification
failures** under the independent Python verifier.
Config SHA-256 `cc565bf1ef6be2…`, artifacts in [`artifacts/`](artifacts/).

**±1 near misses, bases in [1, 30001): exactly the five known identities, all +1.**
9,594,478 records verified. The five recovered are Wroblewski's:

```
 147⁶ +  92⁶ +  71⁶ =  133⁶ +  132⁶ + 1
 295⁶ + 154⁶ +  75⁶ =  294⁶ +  173⁶ + 1
 311⁶ + 268⁶ + 159⁶ =  330⁶ +   55⁶ + 1
 556⁶ + 409⁶ + 197⁶ =  515⁶ +  500⁶ + 1
1255⁶ + 815⁶ + 573⁶ = 1143⁶ + 1123⁶ + 1
```

No −1 identity exists below 30,001. **The −1 residual is the genuinely new
coverage here** — a −1 near miss is (6,4,2)-shaped and is not implied by any
prior search, explicit or otherwise.

### How this compares to prior work

| System | Prior bound | Source |
|---|---|---|
| (6,3,2) exact, explicit | complete base-box coverage through bases ≤ 423 (common sum ≤ 1.15×10¹⁶; the 475 figure is a single-term ceiling, **not** a base-box radius) | Ekl, *Math. Comp.* 67 (1998) |
| (6,3,2) exact, *implied* | bases < 33,225, via (6,3,3) coverage with a zero term under the archive's `0 ≤ xᵢ` convention | Moore's (6,3,3) claim (contributor-reported) |
| (6,3,2; +1), *implied* | bases < 33,225 — a +1 miss is a (6,3,3) solution with a 1⁶ term | same |
| (6,3,2; −1) | radius 17,800, secondhand and unpublished | Moore via Piezas |
| (6,2,2) | max base ≈ 58,385 (sum ≤ 2⁹⁶) | Schoenfield, [OEIS A046881](https://oeis.org/A046881) |

So the exact result is **≈4.51× past the implied (6,3,3)-derived frontier** and
≈354× past Ekl's complete base-box coverage. These are different comparisons
against different conventions and are deliberately not conflated.

Conventions: bases are **strictly positive** and the bound is **exclusive**
(`1 ≤ xᵢ < B`). The archive's formal convention permits zero, which is what
makes the implied-coverage rows above possible.

## The congruence obstruction

The interesting mathematics is *why* the near misses look the way they do.

For each p ∈ {2,3,7} there is a modulus q ∈ {8,9,7} where x⁶ mod q is exactly
the indicator `[p ∤ x]`. So modulo q, the equation stops being about sixth
powers and becomes a **counting statement**:

    δ ≡ (# left bases coprime to p) − (# right bases coprime to p)   (mod q)

The generic value — nothing divisible by anything — is 3 − 2 = **+1**. A +1
near miss costs zero divisibility coincidences; a −1 costs two. Exact local
density ratios: 13.77 (mod 7) × 2.00 (mod 8) × 3.07 (mod 9) = **84.6039 : 1**
in favour of +1. Mod 13 contributes nothing to that ratio but suppresses δ=0
separately: sixth powers mod 13 lie in {0, ±1}, and five nonzero terms force an
odd sum, so an exact solution requires a base divisible by 13.

Consequences, all verified against the corpora:

- **288 of the 504 residue classes mod 504 are unreachable for δ.** Zero hits
  across 15.8M records — a free integrity test that any claimed discovery must
  pass. (A "find" landing in a forbidden class is a bug announcing itself.)
- The class-stride density for the exact campaign is exactly **9/49**, and the
  engine's own iteration counters reproduce it to four digits.
- Census of δ mod 504: observed +1:−1 of **89.64** (all-pairs corpus) and
  **58.79** (admissible corpus) against model values 84.60 and 54.98. Consistent
  with the model; *not* an independent replication — the corpora are
  conditioned on `|δ| ≤ 2⁴⁰`, ordered bases, a fixed deterministic column
  sample, and shared-base removal, and the mod-504/6552 tests are nested.

Extending the local product over more primes (with primitive conditioning and
prime-power ladders) gives partial products stabilising near **R(0) ≈ 138** and
**R(−1) ≈ 864** — see [`r0_indep.py`](r0_indep.py). **No rigorous tail bound has
been established**, so these are computed partial products, not proven limits.

The dimensional heuristic (5 terms, degree 6 → X^(5−6) per dyadic shell) says
the expected count in each shell decays like A/X, so the tail is finite and
deepening the search yields progressively less. The near-miss data are
consistent with that picture — five identities clustered at bases 147–1255,
then a certified desert — but five events are far too sparse to calibrate a
reliable absolute probability, and no such probability is claimed here.

## Corrections

Kept visible on purpose; the audit trail is part of the result.

- **R(0) ≈ 70 / R(−1) ≈ 712 (commit `354d305`) are superseded.**
  [`r0_compute.py`](r0_compute.py) summed only primes ≡ 1 (mod 6) and used
  `N(1)/N(0)` without removing the all-zero residue tuple. Both are wrong for
  primitive solutions: p ≡ 5 (mod 6) primes also contribute (x⁶ ranges over
  quadratic residues there), and the correct primitive factor is
  `N(1)/(N(0)−1)`. Superseded by `r0_indep.py`.
- **An earlier draft claimed a "first expected counterexample near 10⁴⁵."**
  Withdrawn — it assumed a constant rate per octave, which contradicts the
  X⁻¹ shell law. Absolute expectation figures derived from it are withdrawn too.
- **The gap histogram is key-quantized.** Its distances come from truncated
  keys, so bin 0 means "a pair sum shares the probe's key bucket," not
  "|δ| ≤ 1". Renamed `nearestKeyGapHistogramLog2` in v1.5.0; runs before that
  emit the same data under the old name. The emitted `gap` *records* are exact.

## Engine

Single-file C++/OpenMP meet-in-the-middle search (`lps632.cpp`) with an
optional CUDA port (`lps632_gpu.cu`).

- Sorted **truncated value keys** rather than opaque hashes, so the pair table
  supports range queries — which is what makes exact hits, ±1 hits, gap
  statistics and (6,2,2) detection fall out of one structure.
- Congruence chain from runtime-detected structural moduli; class strides mod
  42 in the exact campaign (density 9/49), disabled in the near campaign where
  primitivity reasoning is **unsound** (1+1+1 = 1+1+1 violates it).
- Blocked Bloom prefilter; deterministic tiles; per-tile atomic commits
  (tmp → fsync → rename → dir-fsync); SHA-256 config identity gating resume;
  merge completeness gate.
- **The GPU never decides anything.** It generates candidates; every one is
  re-derived on the CPU in exact unsigned-128 arithmetic. Windows partition
  triple space by s₃ with a ±1 value halo so no probe can fall off the edge of
  a slice.

### GPU status: gated, not yet production-viable

All 12 oracle-equivalence gates pass, including forced multi-window (5/8/16
windows) and forced-overflow (64 KiB cap) configurations, byte-identical to the
CPU engine. But a measured 6-block run at B=150001 says the architecture is
**verify-bound**:

| | per 1.2×10¹⁰ probes |
|---|---|
| GPU kernel | 1.3–2.1 s |
| device→host transfer | 0.2–0.8 s |
| **CPU re-derivation** | **271–372 s** |

That is 99.4% of wall time on the host with both GPUs at 0% utilisation. The
cause is key truncation (`keyShift=42`): ~2–6% of probes hit an occupied key
band that does not contain s₃. The fix is a 16-bit fingerprint stored beside
each key (~65,000× fewer false candidates), not a larger buffer — adaptive
subdivision only repackages the flood. **The near-150k campaign is stopped
pending that change.**

## Build & run

```bash
g++ -O3 -march=native -fopenmp -o lps632 lps632.cpp
g++ -O1 -g -fsanitize=address,undefined -fopenmp -o lps632_asan lps632.cpp
nvcc -O2 -arch=sm_120 -Xcompiler -fopenmp -o lps632gpu lps632_gpu.cu   # optional

./lps632 selftest          # must print ALL SELFTESTS PASSED
./lps632_asan selftest     # same, under ASan/UBSan
./lps632gpu gputest        # GPU vs CPU oracle equivalence

./lps632 search --B 150001 --near 0 --outdir run_exact_150k
./lps632 search --B  30001 --near 1 --outdir run_near_30k
./lps632 search --B 150001 --near 0 --outdir run_exact_150k --resume

python3 verify.py run_exact_150k/results.jsonl     # independent re-derivation
```

Memory: the pair-key table is ~8 bytes × B²/2 (84 GB at B=150001) and must fit
in RAM for the CPU engine.

## Validation battery

- SHA-256 FIPS vectors (config/result hashing is vendored).
- Exhaustive class-mask soundness over all 42³ triple classes × all admissible
  pair classes: every excluded class is congruence-impossible, every allowed
  class has a witness, density exactly 9/49. Run for K ∈ {2,3,4,6}.
- Exhaustive no-false-negative check of every chain filter over its complete
  residue domain, both campaigns, all three probe offsets.
- Brute-force oracle equivalence at small B for K ∈ {2,3,4,6} — K=2 is dense
  with genuine solutions, so every code path carries real positives — plus
  forced fingerprint-collision storms via `--key-shift-extra`.
- Rediscovery gates: the 147/92/71 identity at B=148, all five known identities
  and no exact hit at B=1256.
- Crash/resume/merge determinism: a partial run plus resume must reproduce an
  uninterrupted run bit-for-bit; config-hash mismatch must be refused.
- `oracle.py`, an independently written Python implementation, must produce a
  bit-exact JSONL diff at B=401.

A defect this battery caught, as a sample of what it is for: primitivity
pruning was unsound whenever a structural prime folded with q=3, which drops
the genuine primitive solution 50⁴+29⁴+7⁴ = 51⁴+21⁴ at K=4. The exact-mode
oracle had replicated the same filter and was blind to it; it now uses
gcd-primitivity ground truth.

## Artifacts

[`artifacts/run_exact_150k/`](artifacts/run_exact_150k/) holds the manifest
(config hash + every tile completion), `stats.json`, and the results checksum
for the 150k exact run. The full 6.26M-record corpus is large; a durable
archive copy is pending.

## Output format

`<outdir>/tiles/tile_*.jsonl` (atomic per-tile records), `manifest.txt`
(SHA-256 config header, total tile count, fsync'd completions),
`results.jsonl` (merged, deduplicated), `results.jsonl.sha256`, `stats.json`.
Record types: `exact`, `near` (δ = lhs−rhs = ±1), `dup22`, `gap`
(sampled residual stream), `tilestats`.

Note that "6,259,053 records verified" means every emitted record is
arithmetically valid. *Absence* of solutions rests on something different: the
completeness manifest, the no-false-negative filter proofs, and the
candidate-generation validation above.

## License

MIT — see [LICENSE](LICENSE).

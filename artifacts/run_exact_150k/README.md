# Artifacts: exact campaign, bases in [1, 150001)

Run completed 2026-08-14 14:21 PDT on an i9-14900K (32 threads, 185 GB DDR5),
engine `lps632-1.4.1`, launched as a systemd unit from source commit `87416c7`
(binary built from the tree at that commit; the 1.4.1 version string is
embedded in the manifest's hashed config line).

| item | value |
|---|---|
| command | `lps632 search --B 150001 --near 0 --sample-shift 8 --outdir run_exact_150k --quiet` |
| config SHA-256 | `cc565bf1ef6be25b3a67a4912d1263bd0aac913695015132eee4332c442e1bdb` |
| wall time | 4 d 20 h 33 m (3,081 CPU-hours) |
| tiles | 4,687 — manifest contains exactly `done 0` … `done 4686`, no gaps, no duplicates |
| filtered probes | 103,120,822,088,814 |
| emitted records | 6,259,053 (all `gap`; **zero `exact`, zero `near`, zero `dup22`**) |
| verifier | `python3 verify.py results.jsonl` → `verified 6259053 records ({'gap': 6259053}); 0 failures` |
| results.jsonl SHA-256 | `b36b9f4ec3571dfb9afd703e78b2ed3ced5770f161ee5f3ce144e1f2cae94e73` |
| compiler | g++ 14.3.1 (Rocky Linux 10.2), `-O3 -march=native -fopenmp` |

Files here: `manifest.txt` (config header + all tile completions),
`stats.json` (aggregate counters + key-quantized gap histogram),
`results.jsonl.sha256`.

Notes for auditors:

- This run predates engine 1.5.0, so the manifest has **no `total N` line**
  (that completeness feature landed after launch). Completeness is instead
  auditable directly: the tile count for this config is 4,687 and the done-set
  is exactly {0,…,4686}.
- `stats.json` names the histogram `gapHistogramLog2` (pre-rename). It is the
  key-quantized `nearestKeyGapHistogramLog2` described in the top-level README:
  bin 0 means "same truncated-key bucket," not `|δ| ≤ 1`.
- The full 6.26M-record `results.jsonl` (~1 GB) is not in git; its SHA-256 is
  above and a durable archive copy (Zenodo) is pending. Until then it is
  available from the run host on request.
- What "no solutions" rests on: the completeness manifest above, the engine's
  exhaustively-verified filter soundness (see the validation battery in the
  top-level README), and candidate-generation validation — not on the record
  verification alone, which only shows emitted records are valid.

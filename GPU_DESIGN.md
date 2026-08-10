# lps632-gpu design v1 (adversarial-review-integrated)

Target: dual RTX 5090 (sm_120, 32GB) port of the (6,3,2) probe engine.
Measured target: **10–20×** per GPU over the 14900K (30× only if probe-path
optimizations pan out; VRAM random-access ceiling is ~4–5×10^10 triples/s/GPU).
CUDA 13.3, driver 610, native sm_120 SASS only (`CUDA_DISABLE_PTX_JIT=1`).

## Invariants (unchanged from CPU contract)

1. GPU = candidate generator; every candidate re-derived on CPU via `recover()`
   in exact u128. GPU false negatives are gated by oracle equivalence.
2. Coverage audited via the tile manifest, SHA-256 config identity.
3. Filter tables built by the selftested CPU code, copied to device (residue
   tables as the m-entry prefix slices `r_m[i] = res[i], i < m`).

## Review-mandated design decisions

### Windows (correctness-critical)
- **Key-balanced windows**, boundaries a pure function of (B, near, window
  count, fixed VRAM budget constant) — all in `configString()`. NEVER derived
  from measured free VRAM or GPU count. Resume with a different plan is
  refused by the existing hash check. ≥8 windows; dynamic (work-stealing)
  assignment across GPUs, tile ids = f(window, x-block), GPU-independent.
- **±1 value halo**: each window's key+bloom tables cover pair sums in
  [winLo−1, winHi+1] in near mode. Triple partition spans [3, maxPair+1]
  (the s3 = maxPair+1 triple still probes the maximal pair at diff=+1).
- Near-150k: build the global sorted key table ONCE on host (90GB, fits
  185GB); windows are contiguous slices + per-slice bloom (17GB total at
  12 bits/key — not 11). Exact-300k (later): per-window pair regeneration;
  host RAM is the binding constraint, documented as such.
- Companion CPU gap pass: disjoint manifest namespace (`gapdone N`), explicit
  sampling rate in config; runs concurrently on host (shares the read-only
  table); at 1/256 it is hours-scale — start it first.

### Kernel
- **Segments generated on device** (host lists would be ~200GB at 150k):
  upload per-window (x, yLo, yHi) descriptors (KBs) + POW table (2.4MB,
  L2-resident); zLo/zHi per column via warp-uniform binary search.
- **Branchless AND of all chain stages in near mode** (warp-death probability
  ~(0.4)^32 — early exit only adds branches); exact mode: branchless first
  3–4 stages then `__ballot_sync` vote. Lengthen the chain on GPU (shared-mem
  table loads are cheap; add moduli until marginal reject < cost).
- Device arithmetic: `unsigned __int128` (compiler-emulated, textually
  mirrors CPU expressions). Truncated keys computed on device identically to
  `keyOf()`.
- Probe path: blocked bloom (VRAM line) → **per-window 2^24-bucket index**
  (u32 offsets, 64–128MB, largely L2) → ~7–8 dependent loads within a 1.4KB
  bucket span. Process 2–4 z per thread iteration to keep ≥2 probes in
  flight per warp.
- Candidates append to a device buffer via atomic cursor; overflow sets a
  flag and the tile re-runs with a larger buffer (never dropped, never
  marked done on overflow).

### Integrity (ECC-less VRAM defense)
- Host computes per-slice digest pairs (mod-2^64 sum + XOR; XOR provably
  catches any single-bit flip) for keys, bloom, residue tables, POW.
- A device checksum kernel re-verifies all resident tables **before each
  manifest commit batch** (~25ms per sweep, <1% overhead). Mismatch ⇒
  re-upload once; second mismatch ⇒ `_exit(1)`.
- Crash-only failure semantics: any CUDA error, Xid, hang (per-tile deadline
  = 10× predicted duration via `cudaEventQuery` polling; no kernel watchdog
  headless), or unresolved checksum ⇒ log and `_exit(1)`. systemd restarts
  with `--auto-resume` (resume iff matching manifest exists, else create).
- merge() gains a completeness gate: total tile count in the manifest header;
  refuse merge unless all tiles done (explicit `--partial` flag watermarks
  stats.json). **Backport this to the CPU engine too.**

### Operations
- Devices bound by UUID (`CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=GPU-<uuid>`); UUIDs + driver/toolkit/ptxas/VBIOS in
  run metadata. Production unit: `Conflicts=vllm` (auto-evicts GPU0),
  `After=nvidia-persistenced`.
- **Power-cap both cards to 450W** (`nvidia-smi -pl 450` in ExecStartPre) —
  dual 575W + host ≈ 1.4–1.5kW wall, at the edge of a 15A/120V circuit and
  the 12V-2x6 risk envelope; this integer/memory-bound workload likely never
  sustains 575W (measure in gate 6). NVML telemetry per tile: temps, power,
  clocks, throttle reasons.
- Host is the verifier of record: check 14900K microcode ≥ 0x12B + Intel
  Default profile (Raptor Lake Vmin degradation), one memtest86+ pass over
  the 185GB, and `./lps632 selftest` in ExecStartPre of every unit start.
- One CUDA context per device for the campaign; device arenas allocated once
  at plan-maximum window size; pinned buffers sized modestly.
- PCIe topology check before committing to upload overlap (x8/x8 vs chipset
  x4 — the latter makes uploads 20+ min/window; prefer serial upload +
  compute, ~2–4% wall cost).

### Gates (all must pass per PHYSICAL card, re-run after driver updates)
1. Oracle equivalence sweep (K=2 B=61 dense positives; K=6 B=401; K=6
   B=1256 five Wroblewski identities; near and exact) — byte-identical to
   CPU engine output. **Run with forced tiny windows (≥8 windows at B=401)**
   so halo, slice indexes, and boundary conventions are oracle-covered.
2. Bloom on/off equivalence on device.
3. Candidate-buffer overflow path (tiny buffer forcing re-runs).
4. Window audits: Σ per-window triple iterations == CPU zcap closed form
   (derived independently from maxProbe semantics); Σ per-window pair counts
   == global nPairs; small-B multiset equality of concatenated window keys
   vs the full sorted table, with boundaries deliberately placed ON pair-sum
   values and on perfect sixth powers.
5. Cross-engine: full 30k near run must reproduce run_near_30k/results.jsonl
   byte-for-byte (minus tilestats), per card.
6. Microbench suite (1h, GPU1 first): random 64B VRAM read rate, shared-mem
   chain throughput, u128 add/compare, bucket search latency, actual power
   draw — then bench 20k/30k near vs CPU. Publish measured speedups only.

### Out of scope for GPU v1
Gap stream on device; 7^6 root-table and pair-difference engines; >2.2M
arithmetic; per-GPU failure containment (crash-only instead).

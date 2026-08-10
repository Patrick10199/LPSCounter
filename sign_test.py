#!/usr/bin/env python3
"""Signed-gap asymmetry test over a lps632 gap corpus.

Question: are near-approaches with s3 > s2 (delta > 0, the '+1 side') and
s3 < s2 (delta < 0) equally common at each scale of |delta|? The five known
(6,3,2) +/-1 near misses are ALL +1 — 1-in-16 under symmetry. If the whole
delta<0 side is depressed, that is one mechanism; if parity holds except at
tiny |delta|, the obstruction is arithmetic and possibly provable.

WARNING (audit-corrected): density-gradient bias is NOT negligible — record
mass concentrates at s3 within a few multiples of 2^40, where pair-sum density
varies by O(1) across the window. The engine-faithful null is a large,
s3-dependent curvature profile (deep negative at small s3, ~+0.48 peak near
s3 ~ 2^40, decaying to 0), NOT 50/50. Compare against that null (see
census_504.py binned output), never against 0.5.

Usage: python3 sign_test.py results.jsonl [results2.jsonl ...]
"""
import json
import math
import sys
from collections import defaultdict


def main(paths):
    pos = defaultdict(int)  # log2 bin -> count, delta > 0 (pair below s3)
    neg = defaultdict(int)
    residues_pos = defaultdict(int)  # delta mod small m for the smallest bins
    residues_neg = defaultdict(int)
    small = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if '"gap"' not in line:
                    continue
                r = json.loads(line)
                d = int(r["delta"])
                b = int(math.log2(abs(d))) if abs(d) > 1 else 0
                if d > 0:
                    pos[b] += 1
                else:
                    neg[b] += 1
                if abs(d) < 1 << 24:
                    small.append((d, r))
                    m = abs(d) % 504
                    (residues_pos if d > 0 else residues_neg)[m] += 1
    print("POPULATION NOTE: gap records come from a SAMPLED stream (2^-sampleShift")
    print("of (x,y) columns; see the run's stats.json config). Coverage claims rest")
    print("on the full-rate stream, never on this corpus.")
    print("CORRELATION NOTE: records cluster in (x,y,a,b) families (delta = D + z^6");
    print("sweeps); naive_z assumes independence and OVERSTATES significance. Use a")
    print("family-blocked null before believing any asymmetry.")
    print(f"{'log2|d|':>8} {'d>0':>10} {'d<0':>10} {'pos_frac':>9} {'naive_z':>8}")
    for b in sorted(set(pos) | set(neg)):
        p, n = pos[b], neg[b]
        t = p + n
        if t == 0:
            continue
        frac = p / t
        z = (p - t / 2) / math.sqrt(t / 4) if t >= 4 else float("nan")
        print(f"{b:>8} {p:>10} {n:>10} {frac:>9.4f} {z:>7.2f}")
    tp, tn = sum(pos.values()), sum(neg.values())
    tt = tp + tn
    print(f"\ntotal: +{tp} vs -{tn}  pos_frac={tp/tt:.5f}  naive_z={(tp-tt/2)/math.sqrt(tt/4):.2f}"
          " (correlated records; 50/50 null unvalidated)")
    if small:
        small.sort(key=lambda t: abs(t[0]))
        print(f"\nsmallest |delta| records ({min(20,len(small))} of {len(small)}):")
        for d, r in small[:20]:
            print(f"  delta={d:>12}  ({r['x']},{r['y']},{r['z']}) vs ({r['a']},{r['b']})")
        print("\n|delta| mod 504 for |delta| < 2^24 (pos side):",
              dict(sorted(residues_pos.items())[:12]))
        print("|delta| mod 504 for |delta| < 2^24 (neg side):",
              dict(sorted(residues_neg.items())[:12]))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Pattern analysis over lps632 run output.

Reads results.jsonl (+ optional stats.json) and reports the structure signals:
  1. +1 vs -1 near-miss asymmetry (all 5 known misses are +1 — is that law or luck?)
  2. Residue fingerprints of near/gap records mod 7, 8, 9, 13, 63 (which
     congruence needles do near-solutions thread, and how tightly?)
  3. Geometry: z/x, y/x, b/a distributions — do near misses concentrate at
     small z (which would motivate a pair-difference z-targeted campaign)?
  4. Gap-scaling: counts of |delta| <= D vs D from the sampled stream, compared
     with the random model (expected count linear in D). A deficit at small D
     is an obstruction to find and prove; an excess is a family to parametrize.
     The fitted density calibrates c in E[#solutions beyond B] ~ c/B.

Usage: python3 analyze.py <outdir>/results.jsonl [<outdir>/stats.json]
"""
import json
import math
import sys
from collections import Counter


def main(results_path, stats_path=None):
    near = []   # (x,y,z,a,b,diff)
    gaps = []   # (x,y,z,a,b,delta)
    dups = []
    k = 6
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("type")
            k = r.get("k", k)
            if t == "near":
                near.append((r["x"], r["y"], r["z"], r["a"], r["b"], r["diff"]))
            elif t == "exact":
                print("!!! EXACT SOLUTION RECORD PRESENT — verify immediately:", line)
            elif t == "gap":
                gaps.append((r["x"], r["y"], r["z"], r["a"], r["b"], int(r["delta"])))
            elif t == "dup22":
                dups.append(r)

    print(f"records: {len(near)} near(+/-1), {len(gaps)} gap, {len(dups)} dup22")
    print("NOTE: gap records are a SAMPLED instrument (2^-sampleShift of (x,y) columns,")
    print("see stats.json config) — never a coverage claim. near/exact/dup22 records")
    print("come from the full-rate exhaustive stream.\n")

    if near:
        plus = sum(1 for n in near if n[5] == 1)
        minus = len(near) - plus
        print(f"[1] near-miss sign asymmetry: +1: {plus}   -1: {minus}")
        if minus == 0 and plus >= 3:
            print("    -1 side still empty — check gap records below for the nearest -1 approach.\n")
        else:
            print()

    def residue_table(rows, label):
        print(f"[2] residue fingerprints ({label}, {len(rows)} records)")
        for m in (7, 8, 9, 13, 63):
            cnt = Counter()
            for (x, y, z, a, b, _d) in rows:
                lhs = (pow(x, k, m) + pow(y, k, m) + pow(z, k, m)) % m
                cnt[lhs] += 1
            top = ", ".join(f"{r}:{c}" for r, c in cnt.most_common(6))
            print(f"    s3 mod {m:>2}: {top}")
        print()

    if near:
        residue_table(near, "near +/-1")
    if gaps:
        import random
        random.seed(42)
        residue_table(random.sample(gaps, min(30000, len(gaps))), "gap random sample")

    def geometry(rows, label):
        if not rows:
            return
        zx = sorted(r[2] / r[0] for r in rows)
        yx = sorted(r[1] / r[0] for r in rows)
        ba = sorted(r[4] / r[3] for r in rows)
        q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
        print(f"[3] geometry ({label}): quartiles [p25 p50 p75]")
        print(f"    z/x: {q(zx,.25):.3f} {q(zx,.5):.3f} {q(zx,.75):.3f}   "
              f"y/x: {q(yx,.25):.3f} {q(yx,.5):.3f} {q(yx,.75):.3f}   "
              f"b/a: {q(ba,.25):.3f} {q(ba,.5):.3f} {q(ba,.75):.3f}")
        small_z = sum(1 for v in zx if v < 0.35) / len(zx)
        print(f"    fraction with z/x < 0.35: {small_z:.2f} "
              f"(high => pair-difference z-targeted campaign is the right next move)\n")

    geometry(near, "near +/-1")
    geometry(gaps, "gap records")

    if gaps:
        print("[4] gap scaling: count of |delta| <= D (sampled stream)")
        deltas = sorted(abs(g[5]) for g in gaps)
        lo = deltas[0]
        print(f"    smallest sampled |delta|: {lo} (~2^{math.log2(lo):.1f})" if lo > 0 else "")
        for bits in range(8, 45, 4):
            D = 1 << bits
            c = sum(1 for d in deltas if d <= D)
            if c:
                print(f"    D=2^{bits:>2}: {c:>8}")
        n = len(deltas)
        if n >= 10:
            dmax = deltas[-1]
            # random model: counts linear in D => density lambda ~ n / Dmax per probe-window
            half = sum(1 for d in deltas if d <= dmax / 2)
            print(f"    linearity check: N(Dmax/2)/N(Dmax) = {half/n:.3f} (random model: 0.500)")
            print("    deficit at small D = congruence obstruction to identify;")
            print("    excess/clustering = candidate family — inspect those records by hand.")

    if stats_path:
        with open(stats_path) as f:
            st = json.load(f)
        hist = st.get("gapHistogramLog2", [])
        tot = sum(hist)
        if tot:
            print(f"\n[stats.json] nearest-gap log2 histogram ({tot} sampled probes, "
                  f"population: {st.get('gapPopulation')}):")
            first = next((i for i, c in enumerate(hist) if c), 0)
            last = max(i for i, c in enumerate(hist) if c)
            for i in range(first, min(last + 1, 127)):
                if hist[i]:
                    bar = "#" * max(1, int(50 * hist[i] / max(hist)))
                    print(f"    2^{i:>3}: {hist[i]:>10} {bar}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))

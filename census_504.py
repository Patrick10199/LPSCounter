#!/usr/bin/env python3
"""Decisive corpus tests for the congruence-asymmetry model (audited analysis).

Per corpus, one pass:
  1. delta mod 504 census: ratio of records with delta === +1 vs -1 (mod 504).
     Model predicts 84.60 on an all-pairs population (near campaign) and
     54.98 on an admissible-pairs population (exact campaign).
  2. Forbidden-residue integrity alarm: delta mod 7 in {4}, mod 8 in {4,5},
     mod 9 in {4,5,6} can NEVER occur; any hit = corpus/engine bug.
  3. Signed excess, total and binned by log2(s3 / 2^40), for comparison with
     the engine-faithful curvature null (~+6-7% total on all-pairs pilots).

Usage: python3 census_504.py <results.jsonl> [...]
"""
import json
import math
import sys
from collections import Counter

W = 1 << 40


def run(path):
    plus1 = minus1 = 0
    forbidden = Counter()
    pos = neg = 0
    binned = Counter()  # bin -> [pos, neg] via two counters
    binpos, binneg = Counter(), Counter()
    total = 0
    with open(path) as f:
        for line in f:
            if '"gap"' not in line:
                continue
            r = json.loads(line)
            d = int(r["delta"])
            total += 1
            m = d % 504
            if m == 1:
                plus1 += 1
            elif m == 503:
                minus1 += 1
            if d % 7 == 4:
                forbidden["mod7=4"] += 1
            if d % 8 in (4, 5):
                forbidden[f"mod8={d % 8}"] += 1
            if d % 9 in (4, 5, 6):
                forbidden[f"mod9={d % 9}"] += 1
            s3 = r["x"] ** 6 + r["y"] ** 6 + r["z"] ** 6
            b = round(math.log2(s3 / W))
            if d > 0:
                pos += 1
                binpos[b] += 1
            else:
                neg += 1
                binneg[b] += 1
    print(f"== {path}")
    print(f"records: {total}")
    print(f"delta===+1 (mod 504): {plus1}   delta===-1 (mod 504): {minus1}   "
          f"ratio: {plus1 / minus1 if minus1 else float('inf'):.2f}")
    print(f"forbidden residues (MUST be empty): {dict(forbidden) or 'NONE - integrity OK'}")
    exc = (pos - neg) / (pos + neg)
    print(f"sign excess total: {exc:+.4f}  (+{pos} / -{neg})")
    print("binned excess by log2(s3/2^40):")
    for b in sorted(set(binpos) | set(binneg)):
        p, n = binpos[b], binneg[b]
        t = p + n
        if t < 100:
            continue
        print(f"  bin {b:>3}: {(p - n) / t:+.4f}  (n={t})")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)

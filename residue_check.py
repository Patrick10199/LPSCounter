#!/usr/bin/env python3
"""Random-sample residue audit of a gap corpus (bias-free, unlike head-slices)."""
import json
import random
import sys
from collections import Counter

random.seed(42)
path = sys.argv[1]
lines = []
with open(path) as f:
    for line in f:
        if '"gap"' in line:
            lines.append(line)
n = min(30000, len(lines))
sample = random.sample(lines, n)
c7, c9, c8, zx = Counter(), Counter(), Counter(), Counter()
xband = Counter()
for line in sample:
    r = json.loads(line)
    x, y, z = r["x"], r["y"], r["z"]
    c7[(pow(x, 6, 7) + pow(y, 6, 7) + pow(z, 6, 7)) % 7] += 1
    c9[(pow(x, 6, 9) + pow(y, 6, 9) + pow(z, 6, 9)) % 9] += 1
    c8[(pow(x, 6, 8) + pow(y, 6, 8) + pow(z, 6, 8)) % 8] += 1
    zx[round(z / x, 1)] += 1
    xband[x // 5000] += 1
print("total gap lines:", len(lines), " sampled:", n)
print("s3 mod 7 (count of bases coprime to 7):", dict(sorted(c7.items())))
print("  expected if unconstrained: 3->63%, 2->31.5%, 1->5.2%, 0->0.3%")
print("s3 mod 9:", dict(sorted(c9.items())), " expected: 3->29.6%, 2->44.4%, 1->22.2%, 0->3.7%")
print("s3 mod 8:", dict(sorted(c8.items())))
print("x bands (x//5000):", dict(sorted(xband.items())))
print("z/x rounded:", dict(sorted(zx.items())))

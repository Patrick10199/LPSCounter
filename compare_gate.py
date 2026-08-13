#!/usr/bin/env python3
"""Cross-engine gate: compare non-gap, non-tilestats records of two runs."""
import sys


def load(path):
    keep = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or '"tilestats"' in line or '"type":"gap"' in line:
                continue
            keep.add(line)
    return keep


a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"{sys.argv[1]}: {len(a)} records   {sys.argv[2]}: {len(b)} records")
if a == b:
    print("GATE PASS: record sets identical")
    for line in sorted(a):
        print("  ", line[:120])
    sys.exit(0)
print("GATE FAIL")
for x in sorted(b - a):
    print("  missing from first:", x[:150])
for x in sorted(a - b):
    print("  extra in first:   ", x[:150])
sys.exit(1)

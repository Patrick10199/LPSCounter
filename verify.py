#!/usr/bin/env python3
"""Independent exact verifier for lps632 results.jsonl.

Recomputes every reported identity with Python arbitrary-precision integers.
Exits nonzero if ANY record fails. This is the last line of defense: a record
that passes here is a true statement about integers regardless of any bug in
the search engine.

Usage: python3 verify.py <results.jsonl> [more.jsonl ...]
"""
import json
import sys


def check(rec):
    t = rec.get("type")
    k = rec.get("k", 6)
    if t == "tilestats":
        return True, "skip"
    if t in ("exact", "near"):
        x, y, z, a, b, diff = rec["x"], rec["y"], rec["z"], rec["a"], rec["b"], rec["diff"]
        if not (x >= y >= z >= 1 and a >= b >= 1):
            return False, "ordering"
        lhs = x**k + y**k + z**k
        rhs = a**k + b**k
        if lhs - rhs != diff:
            return False, f"lhs-rhs={lhs-rhs} != diff={diff}"
        if t == "exact" and diff != 0:
            return False, "exact with nonzero diff"
        if t == "near" and abs(diff) != 1:
            return False, "near with |diff| != 1"
        if str(lhs) != rec["lhs"] or str(rhs) != rec["rhs"]:
            return False, "decimal strings mismatch"
        if {x, y, z} & {a, b}:
            return False, "sides share a base (reducible; must not be reported)"
        return True, "ok"
    if t == "dup22":
        c, d, a, b = rec["c"], rec["d"], rec["a"], rec["b"]
        if not (c >= d >= 1 and a >= b >= 1):
            return False, "ordering"
        v1 = c**k + d**k
        v2 = a**k + b**k
        if v1 != v2 or str(v1) != rec["value"]:
            return False, "sums differ"
        if (c, d) == (a, b):
            return False, "identical pair"
        return True, "ok"
    if t == "gap":
        x, y, z, a, b = rec["x"], rec["y"], rec["z"], rec["a"], rec["b"]
        if not (x >= y >= z >= 1 and a >= b >= 1):
            return False, "ordering"
        delta = int(rec["delta"])
        if delta == 0:
            return False, "zero delta in gap record"
        if {x, y, z} & {a, b}:
            return False, "sides share a base (reducible; must not be reported)"
        lhs = x**k + y**k + z**k
        rhs = a**k + b**k
        if lhs - rhs != delta:
            return False, f"lhs-rhs={lhs-rhs} != delta={delta}"
        return True, "ok"
    return False, f"unknown type {t}"


def main(paths):
    total = bad = 0
    counts = {}
    for path in paths:
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ok, why = check(rec)
                if why == "skip":
                    continue
                total += 1
                counts[rec["type"]] = counts.get(rec["type"], 0) + 1
                if not ok:
                    bad += 1
                    print(f"FAIL {path}:{ln}: {why}: {line}")
    print(f"verified {total} records ({counts}); {bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))

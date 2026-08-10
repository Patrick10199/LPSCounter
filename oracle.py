#!/usr/bin/env python3
"""Independent brute-force oracle for lps632.

Enumerates x^K+y^K+z^K vs a^K+b^K over bases in [1, B) (exclusive bound) with
no search filters, no fingerprints, no shortcuts — Python integers throughout —
and prints records in the engine's exact JSONL format, one per line, unsorted.
The only rule mirrored from the search contract is REPORTING convention:
identities whose sides share a base are reducible and are not emitted.

Intended use (near campaign, bit-exact comparison):
    ./lps632 search --B 401 --near 1 --sample-shift 2 --delta-bits 30 --outdir run_check
    python3 oracle.py --B 401 --near 1 --sample-shift 2 --delta-bits 30 > oracle.jsonl
    grep -v tilestats run_check/results.jsonl | sort > a; sort oracle.jsonl > b; diff a b

Do NOT diff against an exact-campaign (--near 0) run: the engine deliberately
prunes classes that cannot be primitive there; that logic is verified by the
C++ selftest's oracle-equivalence and exhaustive class tests instead.
"""
import argparse
import sys

M64 = (1 << 64) - 1


def sm64(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def sampled_xy(x, y, shift):
    if shift < 0:
        return False
    if shift == 0:
        return True
    return (sm64(((x * 0x100000001B3) & M64) ^ y) & ((1 << shift) - 1)) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, required=True, help="exclusive bound: bases in [1, B)")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--near", type=int, required=True, choices=[0, 1])
    ap.add_argument("--sample-shift", type=int, default=-1)
    ap.add_argument("--delta-bits", type=int, default=30)
    a = ap.parse_args()
    B, K = a.B, a.K
    bmax = B - 1

    pw = [0] * (bmax + 1)
    for i in range(1, bmax + 1):
        pw[i] = i ** K

    # pair sums: value -> list of (a,b), a>=b, in ascending-a order
    pairs = {}
    for p in range(1, bmax + 1):
        pp = pw[p]
        for q in range(1, p + 1):
            pairs.setdefault(pp + pw[q], []).append((p, q))

    out = sys.stdout
    emitted = set()

    def emit(line):
        if line not in emitted:
            emitted.add(line)
            out.write(line + "\n")

    # (2,2) repeated pair sums (later pair listed as c,d — matches engine)
    for v, reps in pairs.items():
        for i in range(len(reps) - 1):
            c, d = reps[i + 1]
            aa, bb = reps[i]
            emit(f'{{"type":"dup22","k":{K},"c":{c},"d":{d},"a":{aa},"b":{bb},"value":"{v}"}}')

    max_pair = 2 * pw[bmax]
    delta = 1 << a.delta_bits
    sorted_vals = sorted(pairs) if a.sample_shift >= 0 else []
    import bisect

    for x in range(1, bmax + 1):
        px = pw[x]
        for y in range(1, x + 1):
            pxy = px + pw[y]
            if pxy + 1 > max_pair + (1 if a.near else 0):
                break
            do_sample = sampled_xy(x, y, a.sample_shift)
            for z in range(1, y + 1):
                s3 = pxy + pw[z]
                if s3 > max_pair + (1 if a.near else 0):
                    break
                targets = [(0, s3)]
                if a.near:
                    targets += [(1, s3 - 1), (-1, s3 + 1)]
                for diff, v in targets:
                    reps = pairs.get(v)
                    if not reps:
                        continue
                    lhs = s3
                    for (pa, pb) in reps:
                        if {x, y, z} & {pa, pb}:
                            continue  # reducible: sides share a base
                        typ = "exact" if diff == 0 else "near"
                        emit(
                            f'{{"type":"{typ}","k":{K},"diff":{diff},"x":{x},"y":{y},"z":{z},'
                            f'"a":{pa},"b":{pb},"lhs":"{lhs}","rhs":"{v}"}}'
                        )
                if do_sample:
                    lo, hi = s3 - delta, s3 + delta
                    i0 = bisect.bisect_left(sorted_vals, lo)
                    for vi in range(i0, len(sorted_vals)):
                        v = sorted_vals[vi]
                        if v > hi:
                            break
                        if v == s3:
                            continue
                        d = s3 - v  # positive: pair below; negative: pair above
                        for (pa, pb) in pairs[v]:
                            if {x, y, z} & {pa, pb}:
                                continue
                            emit(
                                f'{{"type":"gap","k":{K},"x":{x},"y":{y},"z":{z},'
                                f'"a":{pa},"b":{pb},"delta":"{d}"}}'
                            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

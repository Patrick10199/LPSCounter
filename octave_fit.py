#!/usr/bin/env python3
# octave_fit.py -- corpus-side scaling + census instrument, independent of
# analyze.py. Reads results.jsonl (or tile files) and answers two questions:
#
#  1. SCALING: does the near-miss/gap intensity decay like the classical
#     density heuristic (rate per base-octave ~ 1/b; median log2-gap slope
#     ~ 2/3 bits per sum-octave = 4 bits per base-octave) or is it
#     scale-invariant (constant rate per base-octave; slope ~ 1/2 per
#     sum-octave = 3 bits per base-octave)? This is the fork between
#     "first expected counterexample at a computable-ish scale" and
#     "heuristically true forever".
#  2. CENSUS: empirical delta-channel ratios mod 504 / 6552 vs the exact
#     local predictions in r0_indep_results.json.
#
# Usage: python octave_fit.py RUNDIR_OR_JSONL... [--B 1501] [--pred r0_indep_results.json]

import argparse
import json
import math
import os
import sys
from collections import defaultdict

K = 6


def rec_stream(paths):
    for p in paths:
        if os.path.isdir(p):
            f = os.path.join(p, "results.jsonl")
            if os.path.exists(f):
                yield from open(f, "r", encoding="utf-8")
            else:
                tdir = os.path.join(p, "tiles")
                for t in sorted(os.listdir(tdir)):
                    if t.endswith(".jsonl"):
                        yield from open(os.path.join(tdir, t), "r", encoding="utf-8")
        else:
            yield from open(p, "r", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--B", type=int, default=0, help="Bexcl (else parsed from manifest)")
    ap.add_argument("--pred", default="r0_indep_results.json")
    ap.add_argument("--census-mods", default="504,6552")
    ap.add_argument("--census-min-octave", type=int, default=0,
                    help="only census gap records with sum-octave >= this")
    args = ap.parse_args()

    B = args.B
    if not B and os.path.isdir(args.paths[0]):
        man = os.path.join(args.paths[0], "manifest.txt")
        if os.path.exists(man):
            head = open(man).readline()
            for tok in head.split("|"):
                if tok.startswith("Bexcl="):
                    B = int(tok.split("=")[1])
    if not B:
        sys.exit("need --B or a run dir with manifest.txt")

    mods = [int(m) for m in args.census_mods.split(",")]
    census = {m: defaultdict(int) for m in mods}
    gap_hist = defaultdict(lambda: defaultdict(int))  # sum-octave -> bits -> n
    near = []           # (maxbase, diff, sum)
    exact = []
    n_gap = 0

    for line in rec_stream(args.paths):
        if '"type":"gap"' in line:
            r = json.loads(line)
            d = int(r["delta"])
            s3 = r["x"] ** K + r["y"] ** K + r["z"] ** K
            o = s3.bit_length() - 1
            if d == 0:
                exact.append(r)  # should never appear in the gap stream
                continue
            gap_hist[o][abs(d).bit_length() - 1] += 1
            if o >= args.census_min_octave:
                for m in mods:
                    census[m][d % m] += 1
            n_gap += 1
        elif '"type":"near"' in line:
            r = json.loads(line)
            near.append((max(r["x"], r["a"]), r["diff"], int(r["lhs"])))
        elif '"type":"exact"' in line:
            exact.append(json.loads(line))

    print(f"parsed: {n_gap} gap, {len(near)} near, {len(exact)} exact  (B={B})")
    if exact:
        print("!! EXACT RECORDS PRESENT:")
        for r in exact:
            print("  ", r)

    # ---------------- near-miss listing + sign split + power-law fit
    plus = sum(1 for _, d, _ in near if d == 1)
    print(f"\n== near misses: {len(near)} total, +1:{plus}  -1:{len(near)-plus} ==")
    for b, d, s in sorted(near):
        print(f"  maxbase={b:>7}  diff={d:+d}  sum~2^{s.bit_length()-1}")
    if near:
        bs = sorted(b for b, _, _ in near)
        blo, bhi = 64.0, float(B)
        print("\n  intensity model lambda(b) = C*b^alpha on "
              f"[{blo:.0f},{bhi:.0f}); profile log-likelihood:")
        # per-octave mass ~ b^(alpha+1): classical (mass ~ 1/b) => alpha=-2,
        # scale-invariant (constant mass per octave)          => alpha=-1
        best = (-1e18, None)
        for a10 in range(-40, 11):
            al = a10 / 10.0
            integ = (bhi ** (al + 1) - blo ** (al + 1)) / (al + 1) if al != -1 else math.log(bhi / blo)
            ll = sum(al * math.log(b) for b in bs) - len(bs) * math.log(integ)
            if ll > best[0]:
                best = (ll, al)
            if al in (-2.0, -1.0):
                print(f"    alpha={al:+.1f}  loglik={ll:+.3f}   "
                      f"{'<- classical (per-octave ~ 1/b)' if al == -2 else '<- scale-invariant (constant/octave)'}")
        print(f"    MLE alpha = {best[1]:+.1f}  (loglik {best[0]:+.3f}); "
              f"n={len(bs)} so treat the CI as wide")

    # ---------------- pair-density slope from the gap-record tail rate
    # The gap stream emits every (probe, pair) candidate with |delta| <=
    # 2^deltaBits, so n_gap(o) / probes(o) estimates 2^(deltaBits+1) times
    # the pair-sum density at height 2^o, immune to the stream's censoring.
    # Classical heuristic: density ~ t^(-2/3)  -> slope -2/3 per sum-octave.
    # Scale-invariant miss rate needs        -> slope -1/2 per sum-octave.
    def iroot6(H):
        if H < 1:
            return 0
        r = int(H ** (1.0 / 6.0))
        while r ** 6 > H:
            r -= 1
        while (r + 1) ** 6 <= H:
            r += 1
        return r

    max_probe = 2 * (B - 1) ** 6 + 1
    probes = defaultdict(float)
    stride = max(1, (B * B // 2) // 20_000_000)  # cap work ~2e7 columns
    if stride > 1:
        print(f"\n(probe census strided: every {stride}th x, counts scaled)")
    for x in range(1, B, stride):
        px = x ** K
        for y in range(1, x + 1):
            c2 = px + y ** K
            if c2 + 1 > max_probe:
                break
            zcap = min(y, iroot6(max_probe - c2))
            if zcap < 1:
                continue
            smax = c2 + zcap ** K
            o = (c2 + 1).bit_length() - 1
            while (1 << o) <= smax:
                hi = min((1 << (o + 1)) - 1, smax)
                lo = 1 << o
                cnt = min(zcap, iroot6(hi - c2)) - (min(zcap, iroot6(lo - 1 - c2)) if lo - 1 >= c2 + 1 else 0)
                if cnt > 0:
                    probes[o] += cnt * stride
                o += 1

    print("\n== pair-density slope (gap-record rate per probe, per sum-octave) ==")
    print("   classical decay ~ t^(-2/3): slope -0.667 | scale-invariant: -0.500")
    top_ok = 6 * math.log2(B - 1)
    pts = []
    for o in sorted(gap_hist):
        n = sum(gap_hist[o].values())
        pr = probes.get(o, 0)
        if pr == 0:
            continue
        lr = math.log2(n / pr)
        flag = ""
        if o + 1 > top_ok:
            flag = "  (trimmed: pair-box boundary)"
        elif n < 1000 or o < 30:
            flag = "  (trimmed: thin/discrete)"
        else:
            pts.append((o, lr))
        print(f"  sum-octave {o:>3}  n_gap={n:>9}  probes={pr:>13}  log2(rate)={lr:+8.3f}{flag}")
    if len(pts) >= 4:
        xs = [o for o, _ in pts]
        ys = [r for _, r in pts]
        n = len(pts)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        resid = [y - (sy / n + slope * (x - sx / n)) for x, y in zip(xs, ys)]
        rms = (sum(r * r for r in resid) / n) ** 0.5
        print(f"\n  fitted slope = {slope:+.4f} per sum-octave over {n} octaves (rms resid {rms:.3f})")
        print(f"  classical -0.667 | scale-invariant -0.500")

    # ---------------- census vs exact predictions
    pred = {}
    if os.path.exists(args.pred):
        pred = json.load(open(args.pred)).get("census", {})
    print("\n== delta-channel census (gap stream) ==")
    for m in mods:
        c = census[m]
        c0, c1, cm1 = c[0], c[1], c[m - 1]
        pk = next((v for k, v in pred.items() if k.startswith(str(m))), None)
        want = f"   predicted +1:0 = {pk['R0']:.3f}" if pk else ""
        r = (c1 / c0) if c0 else float("inf")
        print(f"  mod {m:>5}:  n(0)={c0:>7}  n(+1)={c1:>7}  n(-1)={cm1:>7}  "
              f"+1:0 = {r:.3f}{want}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# r0_indep.py -- INDEPENDENT re-derivation of the local delta-channel ratios
# for x^6+y^6+z^6 - a^6 - b^6 (the (6,3,2) hunt), sharing no code or method
# with r0_compute.py. All per-modulus probabilities are EXACT integers
# (counts out of m^5); only the cumulative products are floats.
#
# Method:
#   naive path (any modulus m): full length-m count arrays, direct O(m^2)
#     circular convolution, Python bigints. Used for prime-power ladders and
#     to cross-validate the coset path.
#   coset path (prime p >= 5): the distribution of x^6 mod p is invariant
#     under multiplication by H = image(x->x^6) on units (|H| = (p-1)/d,
#     d = gcd(6, p-1) in {2,6}); every convolution of H-invariant
#     distributions is H-invariant, so distributions live on d+1 slots
#     (zero + d cosets) and one O(p) "coset addition" tensor per prime
#     drives all convolutions. Exact, O(p) per prime.
#
# Output: table + JSON dump (r0_indep_results.json). Self-asserts the four
# audited fractions (mod 8, 9, 7, 13) before doing anything else.
#
# Usage: python r0_indep.py [--cutoff 20000] [--out r0_indep_results.json]

import argparse
import json
import math
import sys
from fractions import Fraction

K = 6  # exponent


# ---------------------------------------------------------------- naive path

def pow_dist(m):
    """count array c[v] = #{x in [0,m) : x^K == v (mod m)}"""
    c = [0] * m
    for x in range(m):
        c[pow(x, K, m)] += 1
    return c


def conv(f, g, m):
    """circular convolution over Z_m, exact ints"""
    out = [0] * m
    for u, fu in enumerate(f):
        if fu:
            for v, gv in enumerate(g):
                if gv:
                    out[(u + v) % m] += fu * gv
    return out


def delta_dist_naive(m):
    """count array of delta = x^K+y^K+z^K-a^K-b^K over (Z_m)^5"""
    p1 = pow_dist(m)
    pneg = [0] * m
    for v, c in enumerate(p1):
        pneg[(-v) % m] += c
    d2 = conv(p1, p1, m)
    d3 = conv(d2, p1, m)
    d2n = conv(pneg, pneg, m)
    d5 = conv(d3, d2n, m)
    assert sum(d5) == m ** 5
    return d5


def channels_naive(m):
    d5 = delta_dist_naive(m)
    return d5[0], d5[1 % m], d5[(-1) % m]  # counts out of m^5


# ---------------------------------------------------------------- coset path

def channels_coset(p):
    """exact (P0,P1,Pm1) counts out of p^5 for prime p >= 5, O(p)."""
    d = math.gcd(K, p - 1)
    # image H of x->x^K on units, and coset index for every unit
    H = sorted({pow(x, K, p) for x in range(1, p)})
    nH = len(H)
    assert nH * d == p - 1
    idx = [-1] * p
    reps = []
    for v in range(1, p):
        if idx[v] < 0:
            reps.append(v)
            r = len(reps) - 1
            for h in H:
                idx[v * h % p] = r
    assert len(reps) == d
    # addition tensor: cnt[i][j][slot] = #{h in H : reps[i] + reps[j]*h in slot}
    # slots: 0..d-1 = cosets, d = zero
    cnt = [[[0] * (d + 1) for _ in range(d)] for _ in range(d)]
    for i in range(d):
        ri = reps[i]
        for j in range(d):
            rj = reps[j]
            row = cnt[i][j]
            for h in H:
                w = (ri + rj * h) % p
                row[d if w == 0 else idx[w]] += 1
    # distributions: (z, [total mass per coset]); per-element mass = total/nH
    def convC(f, g):
        fz, fA = f
        gz, gA = g
        oz = fz * gz
        oA = [fz * gA[t] + gz * fA[t] for t in range(d)]
        for i in range(d):
            fe = fA[i] // nH          # per-element mass, exact by invariance
            assert fe * nH == fA[i]
            for j in range(d):
                fg = fe * gA[j]
                row = cnt[i][j]
                for t in range(d):
                    if row[t]:
                        oA[t] += fg * row[t]
                if row[d]:
                    oz += fg * row[d]
        return oz, oA

    base = (1, [d * nH if t == idx[1] else 0 for t in range(d)])
    # base dist of x^K: mass 1 at zero, d per element on H (coset of 1)
    negrep = [idx[(p - reps[t]) % p] for t in range(d)]
    bz, bA = base
    nA = [0] * d
    for t in range(d):
        nA[negrep[t]] += bA[t]
    nbase = (bz, nA)

    d2 = convC(base, base)
    d3 = convC(d2, base)
    d2n = convC(nbase, nbase)
    dz, dA = convC(d3, d2n)
    total = dz + sum(dA)
    assert total == p ** 5
    def per_elt(v):
        if v == 0:
            return dz
        t = idx[v]
        pe = dA[t] // nH
        assert pe * nH == dA[t]
        return pe
    return per_elt(0), per_elt(1), per_elt(p - 1)


# ------------------------------------------------- primitive conditioning
# Non-primitive-at-p tuples (p | all five bases) form a self-similar branch
# whose delta is p^6 * delta', so its entire mass lands on delta == 0 for
# modulus p^k with k <= 6, and on p^6-divisible residues generally -- never
# on +-1. Conditioning it out = subtracting its count from the 0-channel.

_P0_CACHE = {}


def all_div_zero_count(p, k):
    """# of all-divisible-by-p tuples mod p^k with delta == 0 (mod p^k)."""
    n_alldiv = (p ** (k - 1)) ** 5
    j = k - 6
    if j <= 0:
        return n_alldiv
    if p ** j not in _P0_CACHE:
        _P0_CACHE[p ** j] = delta_dist_naive(p ** j)[0]
    c0j = _P0_CACHE[p ** j]
    return n_alldiv * c0j // (p ** j) ** 5


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=20000)
    ap.add_argument("--ladder-cap", type=int, default=2500)
    ap.add_argument("--out", default="r0_indep_results.json")
    args = ap.parse_args()

    # ---- self-assertions against the audited hand-derived fractions
    audits = {
        8: (Fraction(10, 32), Fraction(10, 32), Fraction(5, 32)),
        9: (Fraction(73, 243), Fraction(86, 243), Fraction(28, 243)),
        7: (Fraction(4105, 16807), Fraction(9090, 16807), Fraction(660, 16807)),
        13: (Fraction(39601, 371293), Fraction(84270, 371293), Fraction(84270, 371293)),
    }
    for m, want in audits.items():
        c0, c1, cm1 = channels_naive(m)
        tot = m ** 5
        got = (Fraction(c0, tot), Fraction(c1, tot), Fraction(cm1, tot))
        if want[0] is not None:
            assert got[0] == want[0], (m, "P0", got[0], want[0])
        if want[1] is not None:
            assert got[1] == want[1], (m, "P1", got[1], want[1])
        if want[2] is not None:
            assert got[2] == want[2], (m, "P-1", got[2], want[2])
    # cross-validate coset path against naive path
    for p in [5, 7, 11, 13, 19, 31, 37, 43, 61, 97, 103, 127, 151, 193]:
        assert channels_coset(p) == channels_naive(p), ("coset!=naive", p)
    print("SELF-CHECKS PASSED (audited fractions + coset==naive on 14 primes)")

    results = {"K": K, "measure": "all-pairs (uniform residues)", "moduli": {}}

    def record(mod, c0, c1, cm1, note=""):
        tot = mod ** 5 if isinstance(mod, int) else None
        r0 = c1 / c0 if c0 else float("inf")
        rm1 = c1 / cm1 if cm1 else float("inf")
        results["moduli"][str(mod)] = {
            "P0": [c0, tot], "P1": [c1, tot], "Pm1": [cm1, tot],
            "R0": r0, "Rm1": rm1, "note": note,
        }
        return r0, rm1

    # ---- prime-power ladders (finest modulus per prime is what CRT wants)
    ladders = {2: [8, 16, 32, 64, 128, 256],
               3: [9, 27, 81, 243, 729],
               7: [7, 49, 343, 2401],
               13: [13, 169, 2197]}
    finest = {}
    print("\n== prime-power ladders: all-pairs vs PRIMITIVE-conditioned (exact) ==")
    for p, ms in ladders.items():
        prevp = None
        for m in ms:
            if m > args.ladder_cap:
                break
            k = round(math.log(m, p))
            c0, c1, cm1 = channels_naive(m)
            r0, rm1 = record(m, c0, c1, cm1, "ladder")
            c0p = c0 - all_div_zero_count(p, k)
            r0p = c1 / c0p
            rm1p = c1 / cm1
            results["moduli"][str(m)]["R0_prim"] = r0p
            drift = "" if prevp is None else f"  (prim delta {r0p - prevp:+.2e})"
            print(f"  m={m:>5}  R0={r0:12.9f}  R0_prim={r0p:12.9f}{drift}")
            prevp = r0p
            finest[p] = (m, r0p, rm1p)

    # ---- prime scan
    print(f"\n== prime scan to {args.cutoff} (coset path, exact) ==")
    sieve = bytearray([1]) * (args.cutoff)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(args.cutoff ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:: i] = b"\x00" * len(sieve[i * i:: i])
    primes = [i for i in range(5, args.cutoff) if sieve[i]]

    logR0, logRm1 = 0.0, 0.0     # refined basis: finest ladder rung per prime
    logR0_p1 = 0.0               # comparison basis matching the audit: {8, 9, p^1}
    for p in (2, 3):
        m, r0, rm1 = finest[p]
        logR0 += math.log(r0)
        logRm1 += math.log(rm1)
    for m in (8, 9):
        c0, c1, _ = channels_naive(m)
        logR0_p1 += math.log(c1 / c0)
    checkpoints = [127, 250, 500, 1000, 2500, 5000, 10000, 20000, 50000]
    top = []
    plateau = []
    band = {}
    for p in primes:
        c0, c1, cm1 = channels_coset(p)
        r0p1 = c1 / c0
        if p in (7, 13):
            m, r0, rm1 = finest[p]  # refined + primitive-conditioned by ladder
        else:
            r0, rm1 = c1 / (c0 - 1), c1 / cm1  # subtract the all-zero tuple
            if p < 200 or abs(math.log(r0)) > 0.01:
                record(p, c0, c1, cm1, "scan")
        logR0 += math.log(r0)
        logRm1 += math.log(rm1)
        logR0_p1 += math.log(r0p1)
        b = int(math.log10(p) * 2)  # half-decade bands for the bias diagnostic
        band.setdefault(b, []).append(p * (r0 - 1.0))
        if abs(math.log(r0)) > 0.005:
            top.append((abs(math.log(r0)), p, r0))
        while checkpoints and p >= checkpoints[0]:
            cp = checkpoints.pop(0)
            plateau.append((cp, math.exp(logR0), math.exp(logRm1), math.exp(logR0_p1)))
            print(f"  cutoff {cp:>6}:  R(0) = {math.exp(logR0):10.4f}   R(-1) = {math.exp(logRm1):10.4f}   [audit-basis R(0) = {math.exp(logR0_p1):9.4f}]")
    plateau.append((args.cutoff, math.exp(logR0), math.exp(logRm1), math.exp(logR0_p1)))
    print(f"  cutoff {args.cutoff:>6}:  R(0) = {math.exp(logR0):10.4f}   R(-1) = {math.exp(logRm1):10.4f}   [audit-basis R(0) = {math.exp(logR0_p1):9.4f}]   << final")

    top.sort(reverse=True)
    print("\n== largest per-prime contributors ==")
    for _, p, r0 in top[:15]:
        print(f"  p={p:>6}  R0_p = {r0:.6f}")

    # convergence diagnostic: if mean of p*(R0_p - 1) stays positive and flat
    # across bands, log R(0) grows like sum c/p (log log divergence); if it
    # decays toward 0, the product converges.
    print("\n== divergence diagnostic: mean p*(R0_p - 1) per half-decade ==")
    for b in sorted(band):
        v = band[b]
        lo, hi = 10 ** (b / 2), 10 ** ((b + 1) / 2)
        print(f"  p in [{lo:8.0f},{hi:8.0f})  n={len(v):>4}  mean={sum(v)/len(v):+9.3f}")

    # ---- census predictions for the corpus (CRT products of exact channels)
    def crt_ratio(mods):
        n0 = Fraction(1)
        n1 = Fraction(1)
        nm1 = Fraction(1)
        for m in mods:
            c0, c1, cm1 = channels_naive(m)
            tot = m ** 5
            n0 *= Fraction(c0, tot)
            n1 *= Fraction(c1, tot)
            nm1 *= Fraction(cm1, tot)
        return float(n1 / n0), float(n1 / nm1)
    for label, mods in [("504 = 8*9*7", [8, 9, 7]), ("6552 = 8*9*7*13", [8, 9, 7, 13])]:
        r0, rm1 = crt_ratio(mods)
        results["census"] = results.get("census", {})
        results["census"][label] = {"R0": r0, "Rm1": rm1}
        print(f"\ncensus prediction mod {label}:  +1:0 = {r0:.6f}   +1:-1 = {rm1:.6f}")

    results["plateau"] = plateau
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())

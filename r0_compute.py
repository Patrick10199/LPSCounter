#!/usr/bin/env python3
"""SUPERSEDED — kept for the audit trail. Use r0_indep.py instead.

Two errors make this script's extended products wrong for PRIMITIVE solutions:

  1. It sums only primes p == 1 (mod 6). Primes p == 5 (mod 6) also contribute:
     there gcd(6, p-1) = 2, so x^6 = (x^3)^2 ranges over the quadratic
     residues — still non-uniform, still a density factor.
  2. It uses N(1)/N(0) for the exact channel. The primitive factor is
     N(1)/(N(0) - 1): the all-zero residue tuple must be removed, since every
     base divisible by p is exactly the non-primitive configuration the
     coverage argument already excludes. (For t = +-1 primitivity is automatic.)

The "extended product R(0) ~ 70 / R(-1) ~ 712" reported in commit 354d305 came
from this script and is WITHDRAWN. Corrected partial products with primitive
conditioning, prime-power ladders, and all primes are R(0) ~ 138, R(-1) ~ 864
(no rigorous tail bound). See r0_indep.py and the Corrections section of
README.md.

The CORE per-modulus algebra below (mod 8, 9, 7, 13) is correct and agrees with
the independent implementation; only the primitive normalization and the prime
selection are wrong.

Original docstring follows.

R(0): local congruence suppression of exact (6,3,2) solutions vs the +1 channel.

Part 1 — exact model: for each modulus m, distribute delta = X1+X2+X3-Y1-Y2
with each variable drawn from the multiset {x^6 mod m : x in [0,m)} (all-pairs
measure), or with (Y1,Y2) conditioned on admissibility for m in {8,9,7}
(admissible measure: not both divisible by the indicator prime). Report
P(delta===+1)/P(delta===0) and /P(delta===-1) per modulus and cumulatively.

Part 2 — corpus census: count gap records with delta === 0, +1, -1 modulo 504
and 6552 and compare ratios against the model's prediction for that corpus's
pair population.

Usage: python3 r0_compute.py [results.jsonl ...]
"""
import json
import sys
from collections import Counter
from fractions import Fraction


def sixth_power_dist(m):
    c = Counter(pow(x, 6, m) for x in range(m))
    return {v: Fraction(n, m) for v, n in c.items()}


def convolve(d1, d2, m, negate2=False):
    out = Counter()
    for v1, p1 in d1.items():
        for v2, p2 in d2.items():
            v = (v1 - v2) % m if negate2 else (v1 + v2) % m
            out[v] += p1 * p2
    return dict(out)


def delta_dist(m, admissible_prime=None):
    d1 = sixth_power_dist(m)
    left = convolve(convolve(d1, d1, m), d1, m)
    if admissible_prime:
        p = admissible_prime
        # pair (a,b) with not both divisible by p; residue of a^6 mod m
        # correlates with divisibility only when p | m (indicator structure)
        pairs = Counter()
        tot = Fraction(0)
        for a in range(m):
            for b in range(m):
                if a % p == 0 and b % p == 0:
                    continue
                pairs[(pow(a, 6, m) + pow(b, 6, m)) % m] += 1
                tot += 1
        right = {v: Fraction(n) / tot for v, n in pairs.items()}
    else:
        right = convolve(d1, d1, m)
    return convolve(left, right, m, negate2=True)


def report_modulus(m, admissible_prime=None):
    dd = delta_dist(m, admissible_prime)
    p0 = dd.get(0, Fraction(0))
    pp = dd.get(1 % m, Fraction(0))
    pn = dd.get((m - 1) % m, Fraction(0))
    tag = f"m={m}" + (f" (adm p={admissible_prime})" if admissible_prime else "")
    r0 = float(pp / p0) if p0 else float("inf")
    rn = float(pp / pn) if pn else float("inf")
    print(f"  {tag:>16}: P(+1)/P(0) = {r0:8.4f}   P(+1)/P(-1) = {rn:8.4f}")
    return pp, p0, pn


print("== per-modulus channel densities (all-pairs measure) ==")
R0_all = Fraction(1)
Rm_all = Fraction(1)
core = {}
for m in (8, 9, 7, 13):
    pp, p0, pn = report_modulus(m)
    core[m] = (pp, p0, pn)
    R0_all *= pp / p0
    Rm_all *= pp / pn
print(f"  core product {{8,9,7,13}}: R(0) = {float(R0_all):.3f}   R(-1) = {float(Rm_all):.3f}")

print("\n== higher primes p === 1 (mod 6), heuristic independence ==")
R0_ext, Rm_ext = R0_all, Rm_all
for m in (19, 31, 37, 43, 61, 67, 73, 79, 97, 103, 109, 127):
    pp, p0, pn = report_modulus(m)
    R0_ext *= pp / p0
    Rm_ext *= pp / pn
print(f"  extended product: R(0) = {float(R0_ext):.3f}   R(-1) = {float(Rm_ext):.3f}")

print("\n== admissible-pairs measure (exact campaign) for indicator primes ==")
R0_adm = Fraction(1)
for m, p in ((8, 2), (9, 3), (7, 7)):
    pp, p0, pn = report_modulus(m, admissible_prime=p)
    R0_adm *= pp / p0
pp13, p013, _ = core[13]
R0_adm *= pp13 / p013
print(f"  admissible product {{8,9,7}}+13: R(0) = {float(R0_adm):.3f}")

# mod-504 and mod-6552 census predictions per measure
def pred(ms, adm):
    r = Fraction(1)
    for m in ms:
        if adm and m in (8, 9, 7):
            dd = delta_dist(m, {8: 2, 9: 3, 7: 7}[m])
        else:
            dd = delta_dist(m)
        r *= dd[1 % m] / dd[0]
    return float(r)


print(f"\n== census predictions: count(delta===+1)/count(delta===0) ==")
print(f"  mod  504, all-pairs (near corpus):  {pred((8, 9, 7), False):.3f}")
print(f"  mod  504, admissible (exact corpus): {pred((8, 9, 7), True):.3f}")
print(f"  mod 6552, all-pairs:                 {pred((8, 9, 7, 13), False):.3f}")
print(f"  mod 6552, admissible:                {pred((8, 9, 7, 13), True):.3f}")

# Part 2: corpus census
for path in sys.argv[1:]:
    c504 = Counter()
    c6552 = Counter()
    n = 0
    with open(path) as f:
        for line in f:
            if '"gap"' not in line:
                continue
            d = int(json.loads(line)["delta"])
            n += 1
            c504[d % 504] += 1
            c6552[d % 6552] += 1
    print(f"\n== {path} ({n} records) ==")
    for mod, c in ((504, c504), (6552, c6552)):
        z, p1, m1 = c[0], c[1], c[mod - 1]
        print(f"  mod {mod}: 0:{z}  +1:{p1}  -1:{m1}   +1/0 = {p1 / z if z else float('nan'):.3f}"
              f"   +1/-1 = {p1 / m1 if m1 else float('nan'):.2f}")

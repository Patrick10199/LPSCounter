// lps632.cpp — meet-in-the-middle search + near-miss instrument for
//   x^K + y^K + z^K = a^K + b^K   over positive integers with all bases < B
// specialized for K=6: the Lander-Parkin-Selfridge counterexample hunt at (6,3,2).
//
// SEARCH CONTRACT (frozen):
//   Bound convention: EXCLUSIVE. All bases lie in [1, B).
//   Campaign "exact" (--near 0):
//     Enumerates all PRIMITIVE solutions of x^K+y^K+z^K = a^K+b^K with bases in
//     [1,B). Primitivity-derived class restrictions discard only tuples that
//     provably cannot be primitive. Any non-primitive solution scales down to a
//     primitive one with strictly smaller bases (also < B), so absence of
//     primitive solutions excludes non-primitive ones within the bound.
//     Byproduct: primitive (2,2) repeated pair sums a^K+b^K = c^K+d^K.
//   Campaign "near" (--near 1):
//     Full pair table, no primitivity pruning. Finds EVERY solution and every
//     +/-1 near miss (|lhs-rhs| = 1) with bases in [1,B), plus all (2,2)
//     repeated pair sums.
//   Disjoint sides: identities whose two sides share a base value are
//     reducible (subtract the shared power) and are SUPPRESSED from all record
//     streams, counted in degenerateSkipped. For exact solutions at K >= 3
//     this loses nothing: a shared base would contradict FLT.
//   Both campaigns: a deterministic sample of (x,y) columns (2^-sampleShift of
//     them, selected by a fixed hash) is probed unfiltered against the pair
//     table to record nearest-gap statistics and all candidates with
//     |lhs-rhs| <= 2^deltaBits. In exact mode these statistics are measured
//     against the ADMISSIBLE pair population only (labeled in output).
//   Every reported identity is re-derived in exact unsigned 128-bit arithmetic
//   before being written. Filters can only skip work; the false-negative risk
//   is confined to filter/table construction, which is verified exhaustively
//   over each finite residue domain by `selftest`, plus brute-force oracle
//   equivalence at small B.
//
// This build searches IN MEMORY ONLY: the pair-key table must fit in RAM,
// which caps B. Scaling beyond that is out of scope for v1.
//
// Durability: work is split into deterministic x-range tiles. Each tile's
// records go to tiles/tile_<i>.jsonl via write-tmp -> fsync -> atomic rename,
// then the tile is marked done in manifest.txt (fsync'd). A crash can only
// duplicate a tile's records, never lose a completed tile; `merge` (run
// automatically when all tiles complete) deduplicates into results.jsonl and
// writes stats.json and results.jsonl.sha256. Config identity is SHA-256.
//
// Build:   g++ -O3 -march=native -fopenmp -o lps632 lps632.cpp
// Sanit:   g++ -O1 -g -fsanitize=address,undefined -fopenmp -o lps632_asan lps632.cpp
// Races:   clang++ -O1 -g -fsanitize=thread -fopenmp -o lps632_tsan lps632.cpp
//          (needs clang's libomp; TSan + libgomp produces unusable noise)
//
//   ./lps632 selftest
//   ./lps632 bench  --B 10000,20000,30000
//   ./lps632 search --B 150000 --near 0 [--outdir DIR] [--resume] [--threads N]
//                   [--tiles N] [--max-tiles N] [--sample-shift 6] [--delta-bits 40]
//                   [--mem-gb G] [--force-mem] [--key-shift-extra N] [--quiet]
//   ./lps632 merge  --outdir DIR
#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>
#ifdef __unix__
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned __int128 u128;

static const char *CODE_VERSION = "lps632-1.5.0";

// ---------------------------------------------------------------- utilities

static std::string dec(u128 v) {
  if (v == 0) return "0";
  std::string s;
  while (v) { s += char('0' + (int)(v % 10)); v /= 10; }
  std::reverse(s.begin(), s.end());
  return s;
}

static inline u64 sm64(u64 x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static u64 powmod(u64 x, u64 e, u64 m) {
  u64 r = 1 % m; x %= m;
  while (e) { if (e & 1) r = r * x % m; x = x * x % m; e >>= 1; }
  return r;
}

static void die(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
#ifdef __unix__
  _exit(1); // avoid exit()'s cross-thread stdio flush race; stderr is unbuffered
#else
  exit(1);
#endif
}

// strict integer parsing: full consumption + range check (atoi's silent 0 on
// garbage could flip a campaign from near to exact)
static long long parseInt(const char *s, long long lo, long long hi, const char *what) {
  char *end = nullptr;
  long long v = strtoll(s, &end, 10);
  if (!end || *end != 0 || end == s || v < lo || v > hi)
    die("bad value for %s: '%s' (allowed %lld..%lld)", what, s, lo, hi);
  return v;
}

// ------------------------------------------------------------ SHA-256
// Vendored implementation; selftest checks FIPS test vectors before anything
// downstream trusts it.
struct SHA256 {
  u32 h[8]; u64 len; u8 buf[64]; size_t bufLen;
  static const u32 K[64];
  SHA256() { init(); }
  void init() {
    static const u32 H0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(h, H0, sizeof h); len = 0; bufLen = 0;
  }
  static inline u32 rotr(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const u8 *p) {
    u32 w[64];
    for (int i = 0; i < 16; i++)
      w[i] = (u32)p[4 * i] << 24 | (u32)p[4 * i + 1] << 16 | (u32)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
      u32 s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      u32 s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
      u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      u32 ch = (e & f) ^ (~e & g);
      u32 t1 = hh + S1 + ch + K[i] + w[i];
      u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      u32 mj = (a & b) ^ (a & c) ^ (b & c);
      u32 t2 = S0 + mj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  void update(const void *data, size_t n) {
    const u8 *p = (const u8 *)data;
    len += n;
    while (n) {
      size_t t = std::min(n, (size_t)64 - bufLen);
      memcpy(buf + bufLen, p, t);
      bufLen += t; p += t; n -= t;
      if (bufLen == 64) { block(buf); bufLen = 0; }
    }
  }
  std::string final() {
    u64 bits = len * 8;
    u8 one = 0x80, z = 0;
    update(&one, 1);
    while (bufLen != 56) update(&z, 1);
    u8 L[8];
    for (int i = 0; i < 8; i++) L[i] = (u8)(bits >> (56 - 8 * i));
    update(L, 8);
    char out[65];
    for (int i = 0; i < 8; i++) snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 64);
  }
  static std::string of(const std::string &s) { SHA256 c; c.update(s.data(), s.size()); return c.final(); }
};
const u32 SHA256::K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static double memAvailableGB() {
#ifdef __unix__
  FILE *f = fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  double gb = 0;
  while (fgets(line, sizeof line, f)) {
    unsigned long long kb;
    if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { gb = kb / 1048576.0; break; }
  }
  fclose(f);
  return gb;
#else
  return 0;
#endif
}

// instrumentation only: counts a-loop iterations inside collectPairsInRange so
// verify cost can be MEASURED per recovery instead of estimated
static thread_local u64 tl_aIters = 0;

// deterministic sampling predicate for the unfiltered gap stream; replicated
// exactly by the oracles (C++ and Python). Samples 2^-shift of (x,y) columns.
static inline bool sampledXY(u64 x, u64 y, int shift) {
  if (shift < 0) return false;
  if (shift == 0) return true;
  return (sm64(x * 0x100000001B3ull ^ y) & ((1ull << shift) - 1)) == 0;
}

// ---------------------------------------------------------------- params

struct Params {
  int K = 6;
  u64 B = 0;             // EXCLUSIVE bound: bases in [1, B)
  int near = -1;         // 0 exact campaign, 1 near campaign; must be set explicitly for search
  int sampleShift = 6;   // gap stream samples 2^-shift of (x,y); -1 disables
  int deltaBits = 40;    // report sampled candidates with |delta| <= 2^deltaBits
  int keyShiftExtra = 0; // test hook: inflate key truncation to force false candidates
  int threads = 0;
  int tiles = 0;
  int maxTiles = 0;      // test hook: process at most N tiles this invocation
  double memGB = 0;
  bool forceMem = false;
  bool resume = false;
  bool quiet = false;
  bool noBloom = false;      // disable the Bloom prefilter (bench comparison)
  bool inMemoryOnly = false; // selftest/bench: no files
  int gpuWindows = 0;        // GPU engine: value-space window count (0 = CPU/single)
  std::string outDir = "";

  std::string configString() const {
    char buf[512];
    int n = snprintf(buf, sizeof buf,
             "%s|K=%d|Bexcl=%" PRIu64 "|near=%d|sampleShift=%d|deltaBits=%d|keyShiftExtra=%d|tiles=%d",
             CODE_VERSION, K, B, near, sampleShift, deltaBits, keyShiftExtra, tiles);
    // window plan is part of the coverage identity: resume with a different
    // plan must be refused (GPU_DESIGN.md; layout never derives from free VRAM)
    if (gpuWindows > 0 && n > 0 && n < (int)sizeof buf)
      snprintf(buf + n, sizeof buf - n, "|gpuWindows=%d", gpuWindows);
    return buf;
  }
  std::string configHash() const { return SHA256::of(configString()); }
};

// a reported identity: x^K+y^K+z^K = a^K+b^K + diff, diff in {0,+1,-1};
// or (kind=1) a (2,2) repeated pair sum x^K+y^K = a^K+b^K.
struct Hit {
  int kind; // 0 = (3,2) exact/near, 1 = (2,2) dup
  u64 x, y, z, a, b;
  long long diff;
  bool operator<(const Hit &o) const {
    return std::tie(kind, x, y, z, a, b, diff) < std::tie(o.kind, o.x, o.y, o.z, o.a, o.b, o.diff);
  }
  bool operator==(const Hit &o) const {
    return kind == o.kind && x == o.x && y == o.y && z == o.z && a == o.a && b == o.b && diff == o.diff;
  }
};

// sampled gap-stream report: pair sum within 2^deltaBits of a sampled triple sum
struct GapHit {
  u64 x, y, z, a, b;
  bool neg;   // true: pair sum below s3
  u128 delta; // |s3 - s2| > 0, exact
  bool operator<(const GapHit &o) const {
    return std::tie(x, y, z, a, b, neg) < std::tie(o.x, o.y, o.z, o.a, o.b, o.neg);
  }
  bool operator==(const GapHit &o) const {
    return std::tie(x, y, z, a, b, neg) == std::tie(o.x, o.y, o.z, o.a, o.b, o.neg);
  }
};

// ---------------------------------------------------------------- engine

struct Engine {
  Params P;
  u64 Bmax; // largest base = B - 1
  int K;

  std::vector<u128> POW; // POW[i] = i^K, 0..Bmax
  u128 maxPair;          // 2*POW[Bmax]
  u128 maxProbe;         // maxPair + (near?1:0)
  int keyShift = 0;      // key = value >> keyShift, fits 63 bits
  int bucketShift = 0;   // bucket = key >> bucketShift, 16-bit bucket space

  // --- structural (q,p): x^K mod q == (p|x ? 0 : 1), detected at runtime ---
  struct QP { u32 q, p; };
  std::vector<QP> structQP;
  u32 structComposite = 1; // product of qualifying q (504 for K=6)
  u32 strideM = 1;         // exact mode: product of folded p (42 for K=6); 1 in near mode
  std::vector<u64> classMask; // [strideM^2] bit cz => class triple admissible
  double strideDensity = 1.0;

  // --- congruence chain (scalar residue arrays; packing is a future, separately
  // benchmarked optimization) ---
  struct Mod {
    u32 m;
    std::vector<u8> F;    // size 3m; bit0: v=s3 feasible; bit1: v=s3-1; bit2: v=s3+1
    std::vector<u16> res; // x^K mod m for x in [0,Bmax]
    double pass;          // expected pass for uniform triples (convolution-exact)
  };
  std::vector<Mod> chain;

  // --- pair table: globally sorted truncated value keys, bucket-indexed ---
  std::vector<u64> keys;
  std::vector<u64> bucketOfs; // 65537
  u64 nPairs = 0;
  double pairBuildSec = 0;

  // --- blocked Bloom prefilter in front of the bucket binary search ---
  // Pure accelerator: a Bloom "no" is only trusted because every stored key
  // was inserted (same loop); a false negative here would lose solutions, so
  // the oracle-equivalence selftests (dense K=2 positives) gate it hard.
  std::vector<u64> bloomBits; // 64-byte blocks, 8 u64 each
  u64 bloomBlocks = 0;

  static inline u64 bloomBlockOf(u64 h, u64 nblk) { return (u64)(((u128)h * nblk) >> 64); }
  inline void bloomInsert(u64 k) {
    u64 h = sm64(k);
    u64 *w = &bloomBits[bloomBlockOf(h, bloomBlocks) * 8];
    u64 h2 = sm64(h ^ 0xA5A5A5A5DEADBEEFull);
    for (int i = 0; i < 4; i++) {
      u32 c = (u32)(h2 >> (9 * i)) & 511;
      __atomic_fetch_or(&w[c >> 6], 1ull << (c & 63), __ATOMIC_RELAXED);
    }
  }
  inline bool bloomMay(u64 k) const {
    u64 h = sm64(k);
    const u64 *w = &bloomBits[bloomBlockOf(h, bloomBlocks) * 8];
    u64 h2 = sm64(h ^ 0xA5A5A5A5DEADBEEFull);
    for (int i = 0; i < 4; i++) {
      u32 c = (u32)(h2 >> (9 * i)) & 511;
      if (!(w[c >> 6] >> (c & 63) & 1)) return false;
    }
    return true;
  }

  // --- results (always kept in memory; files additionally when !inMemoryOnly) ---
  std::vector<Hit> hits;
  std::vector<GapHit> gapHits;
  std::set<Hit> seen;
  std::vector<std::string> pairScanLines; // dup22 records from table build
  u64 gapDropped = 0; // gap records not mirrored in memory (files have them all)

  struct Stats {
    u64 iters = 0, chainPass = 0, keySearches = 0, keyMatches = 0, recoveries = 0,
        sampledProbes = 0, exactHits = 0, nearHits = 0, gapReports = 0, degenerateSkipped = 0,
        aIters = 0; // a-loop iterations inside collectPairsInRange (verify cost)
  };
  Stats total;

  struct TileBuf {
    std::vector<std::string> lines;
    Stats st;
    u64 hist[128] = {0};
  };

  FILE *manifestF = nullptr;
  int tilesDirFd = -1; // for directory fsync after tile renames

  std::vector<u64> tileLo;
  std::vector<double> tileMass;
  std::vector<u8> tileDone;
  double massTotal = 0;

  Engine(const Params &p) : P(p), K(p.K) {
    if (P.B < 3) die("B must be >= 3 (exclusive bound)");
    Bmax = P.B - 1;
  }

  // ---------- base tables ----------
  void buildPow() {
    if (Bmax > 1000000) die("B capped at 1,000,001 exclusive (u128 overflow margin)");
    POW.resize(Bmax + 1);
    POW[0] = 0;
    for (u64 i = 1; i <= Bmax; i++) {
      u128 v = 1;
      for (int j = 0; j < K; j++) v *= i;
      POW[i] = v;
    }
    if (POW[Bmax] > (~(u128)0) / 4) die("overflow margin violated");
    maxPair = 2 * POW[Bmax];
    maxProbe = maxPair + (P.near == 1 ? 1 : 0);
    keyShift = 0;
    while ((maxProbe >> keyShift) > (u128)0x7FFFFFFFFFFFFFFFull) keyShift++;
    keyShift += P.keyShiftExtra;
    if (keyShift > 100) die("keyShiftExtra too large");
    u64 maxKey = (u64)(maxProbe >> keyShift);
    bucketShift = 0;
    while ((maxKey >> bucketShift) > 0xFFFF) bucketShift++;
  }

  inline u64 keyOf(u128 v) const { return (u64)(v >> keyShift); }

  // largest i in [0,Bmax] with POW[i] <= v
  inline u64 rootFloor(u128 v) const {
    if (v >= POW[Bmax]) return Bmax;
    u64 lo = 0, hi = Bmax; // POW[lo] <= v < POW[hi]
    while (hi - lo > 1) {
      u64 mid = lo + (hi - lo) / 2;
      if (POW[mid] <= v) lo = mid; else hi = mid;
    }
    return lo;
  }

  // ---------- structural detection ----------
  void buildStructural() {
    structQP.clear();
    const QP basket[] = {{16, 2}, {8, 2}, {27, 3}, {9, 3}, {49, 7}, {7, 7},
                         {25, 5}, {5, 5}, {13, 13}, {11, 11}, {3, 3}};
    std::set<u32> used;
    for (auto qp : basket) {
      if (used.count(qp.p)) continue;
      bool ok = true;
      for (u64 x = 0; x < qp.q && ok; x++)
        ok = (powmod(x, K, qp.q) == (x % qp.p == 0 ? 0u : 1u));
      if (ok) { structQP.push_back(qp); used.insert(qp.p); }
    }
    structComposite = 1;
    for (auto qp : structQP) structComposite *= qp.q;
    if (structComposite > 65535)
      die("structComposite %u exceeds u16 residue tables; unsupported K", structComposite);

    strideM = 1;
    if (P.near == 0)
      for (auto qp : structQP) {
        // SOUNDNESS: folding p requires q > 3. The descent argument needs
        // "c2 = 0 forces c3 = 0", i.e. c3 ≡ 0 (mod q), c3 <= 3  =>  c3 = 0 —
        // false for q = 3 (c3 = 3 works: e.g. 50^4+29^4+7^4 = 51^4+21^4 is a
        // primitive K=4 solution whose pair is divisible by 3 on both sides).
        if (qp.q <= 3) continue;
        if ((u64)strideM * qp.p <= 64) strideM *= qp.p; // class mask must fit u64
      }

    classMask.assign((size_t)strideM * strideM, 1);
    strideDensity = 1.0;
    if (strideM > 1) {
      u64 allowed = 0;
      for (u32 cx = 0; cx < strideM; cx++)
        for (u32 cy = 0; cy < strideM; cy++) {
          u64 mask = 0;
          for (u32 cz = 0; cz < strideM; cz++) {
            bool ok = true;
            for (auto qp : structQP) {
              if (strideM % qp.p) continue;
              int c3 = (cx % qp.p != 0) + (cy % qp.p != 0) + (cz % qp.p != 0);
              bool any = false;
              for (int c2 = 0; c2 <= 2 && !any; c2++)
                if ((u32)(c3 % (int)qp.q) == (u32)(c2 % (int)qp.q) && !(c3 == 0 && c2 == 0))
                  any = true;
              if (!any) { ok = false; break; }
            }
            if (ok) { mask |= 1ull << cz; allowed++; }
          }
          classMask[(size_t)cx * strideM + cy] = mask;
        }
      strideDensity = (double)allowed / ((double)strideM * strideM * strideM);
    }
  }

  // exact mode: pair classes a primitive solution can use (not both divisible
  // by any folded structural prime)
  inline bool pairAdmissible(u64 a, u64 b) const {
    if (P.near == 1) return true;
    for (auto qp : structQP) {
      if (strideM % qp.p) continue;
      if (a % qp.p == 0 && b % qp.p == 0) return false;
    }
    return true;
  }

  // ---------- congruence chain ----------
  void buildChainMod(Mod &M, u32 m) {
    M.m = m;
    std::vector<u32> r(m);
    for (u32 x = 0; x < m; x++) r[x] = (u32)powmod(x, K, m);
    std::vector<u8> inPair(m, 0);
    bool restrict = (P.near == 0);
    for (u32 a = 0; a < m; a++)
      for (u32 b = 0; b <= a; b++) {
        if (restrict) {
          bool ok = true;
          for (auto qp : structQP) {
            if (strideM % qp.p) continue;
            if (m % qp.p) continue; // residues mod m carry no divisibility info for p
            if (a % qp.p == 0 && b % qp.p == 0) { ok = false; break; }
          }
          if (!ok) continue;
        }
        inPair[(r[a] + r[b]) % m] = 1;
      }
    M.F.assign(3 * (size_t)m, 0);
    for (u32 i = 0; i < 3 * m; i++) {
      u8 f = inPair[i % m] ? 1 : 0;
      if (P.near == 1) {
        f |= inPair[(i + m - 1) % m] ? 2 : 0; // v = s3 - 1
        f |= inPair[(i + 1) % m] ? 4 : 0;     // v = s3 + 1
      }
      M.F[i] = f;
    }
    // exact expected pass rate for uniform x,y,z via convolution over the
    // (small) support of x^K mod m
    std::map<u32, double> supp;
    for (u32 x = 0; x < m; x++) supp[r[x]] += 1.0 / m;
    double pass = 0;
    for (auto &i1 : supp)
      for (auto &i2 : supp)
        for (auto &i3 : supp)
          if (M.F[i1.first + i2.first + i3.first]) pass += i1.second * i2.second * i3.second;
    M.pass = pass;
    M.res.resize(Bmax + 1);
    for (u64 v = 0; v <= Bmax; v++) M.res[v] = (u16)powmod(v, K, m);
  }

  void buildChain() {
    chain.clear();
    Mod comp;
    buildChainMod(comp, structComposite);
    std::vector<Mod> cands;
    for (u32 m : {13u, 19u, 31u, 37u, 43u, 61u, 67u, 73u, 79u, 97u, 103u,
                  109u, 127u, 139u, 151u, 157u, 163u, 181u, 193u, 199u}) {
      Mod M;
      buildChainMod(M, m);
      if (M.pass < 0.93) cands.push_back(std::move(M));
    }
    std::sort(cands.begin(), cands.end(), [](const Mod &a, const Mod &b) { return a.pass < b.pass; });
    // near mode: composite (strongest) first. exact mode: stride classes already
    // encode the composite's information, so primes lead and composite goes last
    // as a redundant safety net.
    if (P.near == 1 || strideM == 1) chain.push_back(std::move(comp));
    for (auto &M : cands) {
      if (chain.size() >= 9) break;
      chain.push_back(std::move(M));
    }
    if (P.near == 0 && strideM > 1) chain.push_back(std::move(comp));
  }

  // ---------- pair table ----------
  void buildPairs() {
    double t0 = omp_get_wtime();
    // Two passes (histogram, scatter) must see IDENTICAL per-thread (a,b)
    // partitions. We do not rely on OpenMP schedule reproducibility at all:
    // explicit per-thread a-ranges, one pinned parallel region, and a cursor
    // postcondition that dies on any divergence.
    omp_set_dynamic(0);
    int T = omp_get_max_threads();
    const size_t NB = 1 << 16;
    std::vector<u64> aLo(T + 1);
    aLo[0] = 1;
    aLo[T] = Bmax + 1;
    for (int t = 1; t < T; t++) { // ~equal pair mass: cost of column a is ~a
      u64 v = (u64)((double)Bmax * std::sqrt((double)t / T));
      aLo[t] = std::max<u64>(std::max<u64>(v, 1), aLo[t - 1]);
    }
    for (int t = 1; t <= T; t++) aLo[t] = std::max(aLo[t], aLo[t - 1]);
    std::vector<std::vector<u64>> hist(T, std::vector<u64>(NB, 0));
    std::vector<std::vector<u64>> cur(T, std::vector<u64>(NB, 0));
    int teamSize = 0;
#pragma omp parallel num_threads(T)
    {
      int t = omp_get_thread_num();
#pragma omp master
      teamSize = omp_get_num_threads();
      auto &h = hist[t];
      for (u64 a = aLo[t]; a < aLo[t + 1]; a++)
        for (u64 b = 1; b <= a; b++) {
          if (!pairAdmissible(a, b)) continue;
          h[keyOf(POW[a] + POW[b]) >> bucketShift]++;
        }
#pragma omp barrier
#pragma omp single
      {
        bucketOfs.assign(NB + 1, 0);
        for (size_t bk = 0; bk < NB; bk++) {
          u64 s = 0;
          for (int tt = 0; tt < T; tt++) s += hist[tt][bk];
          bucketOfs[bk + 1] = bucketOfs[bk] + s;
        }
        nPairs = bucketOfs[NB];
        keys.resize(nPairs);
        for (size_t bk = 0; bk < NB; bk++) {
          u64 o = bucketOfs[bk];
          for (int tt = 0; tt < T; tt++) { cur[tt][bk] = o; o += hist[tt][bk]; }
        }
        if (!P.noBloom) {
          bloomBlocks = std::max<u64>(1, (nPairs * 12 + 511) / 512);
          bloomBits.assign(bloomBlocks * 8, 0);
        }
      } // implicit barrier
      auto &c = cur[t];
      for (u64 a = aLo[t]; a < aLo[t + 1]; a++)
        for (u64 b = 1; b <= a; b++) {
          if (!pairAdmissible(a, b)) continue;
          u64 k = keyOf(POW[a] + POW[b]);
          keys[c[k >> bucketShift]++] = k;
          if (bloomBlocks) bloomInsert(k);
        }
    }
    if (teamSize != T) die("OpenMP team size %d != requested %d in buildPairs", teamSize, T);
    // postcondition: every (thread,bucket) cursor landed exactly at its
    // reserved slice end — any divergence means a corrupted key table
    for (size_t bk = 0; bk < NB; bk++) {
      u64 o = bucketOfs[bk];
      for (int t = 0; t < T; t++) {
        o += hist[t][bk];
        if (cur[t][bk] != o) die("buildPairs cursor mismatch (thread %d bucket %zu)", t, bk);
      }
      if (o != bucketOfs[bk + 1]) die("buildPairs bucket accounting mismatch");
    }
#pragma omp parallel for schedule(dynamic, 16)
    for (long long bk = 0; bk < (long long)NB; bk++)
      std::sort(keys.begin() + bucketOfs[bk], keys.begin() + bucketOfs[bk + 1]);
    // buckets are the key's top bits, so the whole array is now globally sorted

    // (2,2) repeated-pair-sum scan: equal adjacent keys are candidates
#pragma omp parallel for schedule(dynamic, 64)
    for (long long bk = 0; bk < (long long)NB; bk++)
      for (u64 i = bucketOfs[bk] + 1; i < bucketOfs[bk + 1]; i++) {
        if (keys[i] != keys[i - 1]) continue;
        if (i >= bucketOfs[bk] + 2 && keys[i] == keys[i - 2]) continue; // once per key run
        verifyDupKey(keys[i]);
      }
    pairBuildSec = omp_get_wtime() - t0;
  }

  void verifyDupKey(u64 k) {
    u128 lo = (u128)k << keyShift;
    u128 hi = lo + (((u128)1 << keyShift) - 1);
    std::map<u128, std::vector<std::pair<u64, u64>>> byVal;
    collectPairsInRange(lo, hi, byVal);
    for (auto &kv : byVal) {
      auto &v = kv.second;
      for (size_t i = 0; i + 1 < v.size(); i++) {
        Hit h{1, v[i + 1].first, v[i + 1].second, 0, v[i].first, v[i].second, 0};
        std::string line = formatHit(h);
        bool fresh;
#pragma omp critical(hits)
        {
          fresh = seen.insert(h).second;
          if (fresh) {
            hits.push_back(h);
            pairScanLines.push_back(line);
          }
        }
        if (fresh && !P.quiet)
          fprintf(stderr, "(2,2) repeated pair sum: %" PRIu64 "^%d+%" PRIu64 "^%d = %" PRIu64 "^%d+%" PRIu64 "^%d\n",
                  h.x, K, h.y, K, h.a, K, h.b, K);
      }
    }
  }

  // all admissible pairs (a>=b>=1, a<=Bmax) with POW[a]+POW[b] in [lo,hi]
  void collectPairsInRange(u128 lo, u128 hi, std::map<u128, std::vector<std::pair<u64, u64>>> &out) const {
    if (hi > maxPair) hi = maxPair;
    if (lo < 2) lo = 2;
    if (lo > hi) return;
    u128 halfCeil = (lo + 1) >> 1; // b <= a  =>  POW[a] >= lo/2
    u64 a = rootFloor(halfCeil);
    if (POW[a] < halfCeil) a++;
    if (a < 1) a = 1;
    for (; a <= Bmax; a++) {
      tl_aIters++;
      if (POW[a] > hi - 1) break;
      u128 bl = lo > POW[a] ? lo - POW[a] : 1;
      u128 bh = hi - POW[a];
      u64 b2 = rootFloor(bh);
      if (b2 > a) b2 = a;
      u64 b1 = 1;
      if (bl > 1) { b1 = rootFloor(bl); if (POW[b1] < bl) b1++; }
      for (u64 b = b1; b <= b2; b++) {
        if (!pairAdmissible(a, b)) continue;
        out[POW[a] + POW[b]].push_back({a, b});
      }
    }
  }

  // ---------- formatting ----------
  std::string formatHit(const Hit &h) const {
    char buf[512];
    if (h.kind == 1) {
      snprintf(buf, sizeof buf,
               "{\"type\":\"dup22\",\"k\":%d,\"c\":%" PRIu64 ",\"d\":%" PRIu64 ",\"a\":%" PRIu64
               ",\"b\":%" PRIu64 ",\"value\":\"%s\"}",
               K, h.x, h.y, h.a, h.b, dec(POW[h.x] + POW[h.y]).c_str());
      return buf;
    }
    u128 l = POW[h.x] + POW[h.y] + POW[h.z];
    u128 r = POW[h.a] + POW[h.b];
    snprintf(buf, sizeof buf,
             "{\"type\":\"%s\",\"k\":%d,\"diff\":%lld,\"x\":%" PRIu64 ",\"y\":%" PRIu64 ",\"z\":%" PRIu64
             ",\"a\":%" PRIu64 ",\"b\":%" PRIu64 ",\"lhs\":\"%s\",\"rhs\":\"%s\"}",
             h.diff == 0 ? "exact" : "near", K, h.diff, h.x, h.y, h.z, h.a, h.b,
             dec(l).c_str(), dec(r).c_str());
    return buf;
  }
  std::string formatGap(const GapHit &g) const {
    char buf[384];
    snprintf(buf, sizeof buf,
             "{\"type\":\"gap\",\"k\":%d,\"x\":%" PRIu64 ",\"y\":%" PRIu64 ",\"z\":%" PRIu64 ",\"a\":%" PRIu64
             ",\"b\":%" PRIu64 ",\"delta\":\"%s%s\"}",
             K, g.x, g.y, g.z, g.a, g.b, g.neg ? "" : "-", dec(g.delta).c_str());
    return buf;
  }

  // ---------- probing ----------
  inline bool keyPresent(u64 k, Stats &st) const {
    st.keySearches++;
    if (k > (u64)(maxPair >> keyShift)) return false; // structural guard, not number-theoretic
    if (bloomBlocks && !bloomMay(k)) return false;    // one cache line instead of a binary search
    size_t bk = k >> bucketShift;
    const u64 *lo = keys.data() + bucketOfs[bk];
    const u64 *hi = keys.data() + bucketOfs[bk + 1];
    const u64 *it = std::lower_bound(lo, hi, k);
    return it != hi && *it == k;
  }

  void recover(u128 s3, u8 flags, u64 x, u64 y, u64 z, TileBuf &tb) {
    tb.st.recoveries++;
    u128 lo = s3 - ((flags & 2) ? 1 : 0);
    u128 hi = s3 + ((flags & 4) ? 1 : 0);
    std::map<u128, std::vector<std::pair<u64, u64>>> byVal;
    collectPairsInRange(lo, hi, byVal);
    for (auto &kv : byVal) {
      u128 v = kv.first;
      long long diff;
      if (v == s3) { if (!(flags & 1)) continue; diff = 0; }
      else if (v + 1 == s3) { if (!(flags & 2)) continue; diff = 1; }
      else if (v == s3 + 1) { if (!(flags & 4)) continue; diff = -1; }
      else continue;
      for (auto &ab : kv.second) {
        u128 l = POW[x] + POW[y] + POW[z];
        u128 r = POW[ab.first] + POW[ab.second];
        bool okEq = (diff == 0 && l == r) || (diff == 1 && l == r + 1) || (diff == -1 && l + 1 == r);
        if (!okEq) die("internal error: recovery equality recheck failed");
        // shared-base identities are reducible (e.g. x^6+y^6+1 = x^6+y^6 + 1)
        // and reported nowhere. For exact hits at K >= 3 a shared base would
        // contradict FLT — only a code bug can produce one. (K=2 has genuine
        // Pythagorean shared-base identities; still suppressed as reducible.)
        bool shared = x == ab.first || x == ab.second || y == ab.first || y == ab.second ||
                      z == ab.first || z == ab.second;
        if (shared) {
          if (diff == 0 && K >= 3) die("internal error: exact hit with shared base contradicts FLT");
          tb.st.degenerateSkipped++;
          continue;
        }
        Hit h{0, x, y, z, ab.first, ab.second, diff};
        bool fresh;
#pragma omp critical(hits)
        { fresh = seen.insert(h).second; if (fresh) hits.push_back(h); }
        if (fresh) {
          tb.lines.push_back(formatHit(h));
          if (diff == 0) tb.st.exactHits++; else tb.st.nearHits++;
          if (!P.quiet) {
            if (diff == 0)
              fprintf(stderr, "\n*** EXACT HIT *** %" PRIu64 "^%d + %" PRIu64 "^%d + %" PRIu64
                              "^%d = %" PRIu64 "^%d + %" PRIu64 "^%d = %s\n",
                      x, K, y, K, z, K, ab.first, K, ab.second, K, dec(l).c_str());
            else
              fprintf(stderr, "near miss (lhs-rhs=%+lld): %" PRIu64 ",%" PRIu64 ",%" PRIu64 " vs %" PRIu64
                              ",%" PRIu64 "\n", diff, x, y, z, ab.first, ab.second);
          }
        }
      }
    }
  }

  void gapProbe(u128 s3, u64 x, u64 y, u64 z, TileBuf &tb) {
    tb.st.sampledProbes++;
    u64 k = keyOf(s3);
    size_t bk = k >> bucketShift;
    const u64 *base = keys.data();
    const u64 *it = std::lower_bound(base + bucketOfs[bk], base + bucketOfs[bk + 1], k);
    size_t idx = (size_t)(it - base);
    // the whole keys array is globally sorted, so neighbors are adjacent slots
    bool hasBelow = idx > 0, hasAbove = idx < nPairs;
    u64 below = hasBelow ? base[idx - 1] : 0, above = hasAbove ? base[idx] : 0;
    u128 gap = ~(u128)0;
    if (hasBelow) { u128 g = (u128)(k - below) << keyShift; if (g < gap) gap = g; }
    if (hasAbove) { u128 g = (u128)(above - k) << keyShift; if (g < gap) gap = g; }
    int bin = 127;
    if (gap != ~(u128)0) {
      bin = 0;
      while (gap > 1 && bin < 126) { gap >>= 1; bin++; }
    }
    tb.hist[bin]++;
    // exact reports within 2^deltaBits
    u128 delta = (u128)1 << P.deltaBits;
    u128 lo = s3 > delta ? s3 - delta : 2;
    u128 hi = s3 + delta;
    u64 kLo = keyOf(lo), kHi = keyOf(hi > maxPair ? maxPair : hi);
    bool anyNear = (hasBelow && below >= kLo) || (hasAbove && above <= kHi);
    if (!anyNear) return;
    std::map<u128, std::vector<std::pair<u64, u64>>> byVal;
    collectPairsInRange(lo, hi, byVal);
    for (auto &kv : byVal) {
      if (kv.first == s3) continue; // exact hits belong to the filtered stream
      bool neg = kv.first < s3;
      u128 d = neg ? s3 - kv.first : kv.first - s3;
      if (d > delta) continue;
      for (auto &ab : kv.second) {
        if (x == ab.first || x == ab.second || y == ab.first || y == ab.second ||
            z == ab.first || z == ab.second) {
          tb.st.degenerateSkipped++; // trivial neighbor (shared base), not signal
          continue;
        }
        GapHit g{x, y, z, ab.first, ab.second, neg, d};
        tb.lines.push_back(formatGap(g));
        tb.st.gapReports++;
#pragma omp critical(gaps)
        {
          if (gapHits.size() < 5000000) gapHits.push_back(g);
          else gapDropped++; // in-memory mirror only; tile files keep everything
        }
      }
    }
  }

  // ---------- hot loop (scalar chain; deliberately simple) ----------
  void runTile(u64 xlo, u64 xhi, TileBuf &tb) {
    const int nChain = (int)chain.size();
    for (u64 x = xlo; x < xhi; x++) {
      u32 cx = (u32)(x % strideM);
      for (u64 y = 1; y <= x; y++) {
        u128 C2 = POW[x] + POW[y];
        if (C2 + 1 > maxProbe) break;
        u64 zcap = rootFloor(maxProbe - C2);
        if (zcap > y) zcap = y;
        if (zcap < 1) continue;
        // sampled gap stream: deliberately UNFILTERED (no stride classes, no
        // chain) so its statistics are unbiased on the triple side
        if (sampledXY(x, y, P.sampleShift))
          for (u64 z = 1; z <= zcap; z++) gapProbe(C2 + POW[z], x, y, z, tb);
        u64 mask = strideM > 1 ? classMask[(size_t)cx * strideM + (u32)(y % strideM)] : 1;
        u64 m = mask;
        while (m) {
          u32 cz = (u32)__builtin_ctzll(m);
          m &= m - 1;
          u64 z0 = strideM > 1 ? (cz == 0 ? strideM : (u64)cz) : 1;
          for (u64 z = z0; z <= zcap; z += strideM) {
            tb.st.iters++;
            u8 flags = 0xFF;
            for (int ci = 0; ci < nChain; ci++) {
              const Mod &M = chain[ci];
              flags &= M.F[(u32)M.res[x] + M.res[y] + M.res[z]];
              if (!flags) break;
            }
            if (flags) {
              tb.st.chainPass++;
              u128 s3 = C2 + POW[z];
              u64 k = keyOf(s3);
              if (keyPresent(k, tb.st)) {
                tb.st.keyMatches++;
                recover(s3, flags & 7, x, y, z, tb);
              } else if (P.near == 1) {
                // s3 -/+ 1 can sit across a key boundary
                if ((flags & 2) && keyOf(s3 - 1) != k && keyPresent(keyOf(s3 - 1), tb.st)) {
                  tb.st.keyMatches++;
                  recover(s3, flags & 2, x, y, z, tb);
                }
                if ((flags & 4) && keyOf(s3 + 1) != k && keyPresent(keyOf(s3 + 1), tb.st)) {
                  tb.st.keyMatches++;
                  recover(s3, flags & 4, x, y, z, tb);
                }
              }
            }
          }
        }
      }
    }
  }

  // ---------- tiles / files ----------
  void buildTiles() {
    int T = P.tiles ? P.tiles : (int)std::min<u64>(8192, std::max<u64>(64, Bmax / 32));
    if ((u64)T > Bmax) T = (int)Bmax;
    tileLo.assign(1, 1);
    double totalMass = 0;
    for (u64 x = 1; x <= Bmax; x++) totalMass += (double)x * x;
    double per = totalMass / T, acc = 0;
    for (u64 x = 1; x <= Bmax; x++) {
      acc += (double)x * x;
      if (acc >= per * tileLo.size() && x + 1 <= Bmax) tileLo.push_back(x + 1);
    }
    tileLo.push_back(Bmax + 1);
    tileMass.assign(tileLo.size() - 1, 0);
    massTotal = 0;
    for (size_t i = 0; i + 1 < tileLo.size(); i++) {
      double mm = 0;
      for (u64 x = tileLo[i]; x < tileLo[i + 1]; x++) mm += (double)x * x;
      tileMass[i] = mm;
      massTotal += mm;
    }
    tileDone.assign(tileMass.size(), 0);
  }

  std::string tilePath(int i, bool tmp) const {
    char buf[600];
    int n = snprintf(buf, sizeof buf, "%s/tiles/tile_%06d.jsonl%s", P.outDir.c_str(), i, tmp ? ".tmp" : "");
    if (n < 0 || n >= (int)sizeof buf) die("outdir path too long");
    return buf;
  }

  // write a fully-checked, fsync'd file and atomically rename it into place;
  // any I/O failure dies BEFORE the rename so a bad file is never committed
  void atomicWrite(const std::string &tmp, const std::string &fin, const std::vector<std::string> &lines,
                   const std::string &extraLine) {
#ifdef __unix__
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) die("cannot write %s", tmp.c_str());
    for (auto &l : lines)
      if (fputs(l.c_str(), f) < 0 || fputc('\n', f) < 0) die("write failed (disk full?): %s", tmp.c_str());
    if (!extraLine.empty())
      if (fputs(extraLine.c_str(), f) < 0 || fputc('\n', f) < 0) die("write failed: %s", tmp.c_str());
    if (fflush(f) != 0) die("fflush failed (disk full?): %s", tmp.c_str());
    if (fsync(fileno(f)) != 0) die("fsync failed: %s", tmp.c_str());
    if (fclose(f) != 0) die("fclose failed: %s", tmp.c_str());
    if (rename(tmp.c_str(), fin.c_str()) != 0) die("rename %s failed", tmp.c_str());
    if (tilesDirFd >= 0 && fsync(tilesDirFd) != 0) die("directory fsync failed");
#endif
  }

  // returns false (with message) instead of dying, so selftest can exercise it
  bool checkResumeManifest(std::string &err) {
    std::string mp = P.outDir + "/manifest.txt";
    FILE *mf = fopen(mp.c_str(), "r");
    if (!mf) { err = "manifest not found: " + mp; return false; }
    char line[600];
    bool headerOk = false;
    while (fgets(line, sizeof line, mf)) {
      if (!strchr(line, '\n')) continue; // torn append (crash mid-write): line is not trustworthy
      char hash[80];
      int tid;
      size_t tot;
      if (sscanf(line, "config %64s", hash) == 1 && strlen(hash) == 64) {
        if (P.configHash() != hash) { fclose(mf); err = "manifest config hash mismatch"; return false; }
        headerOk = true;
      } else if (sscanf(line, "total %zu", &tot) == 1) {
        if (tot != tileDone.size()) { fclose(mf); err = "manifest tile-count mismatch (layout change?)"; return false; }
      } else if (sscanf(line, "done %d", &tid) == 1) {
        if (tid >= 0 && (size_t)tid < tileDone.size()) tileDone[tid] = 1;
      }
    }
    fclose(mf);
    if (!headerOk) { err = "manifest missing config header"; return false; }
    return true;
  }

  void openFiles() {
    if (P.inMemoryOnly) return;
#ifndef __unix__
    die("file-backed search requires a POSIX system");
#else
    mkdir(P.outDir.c_str(), 0755);
    mkdir((P.outDir + "/tiles").c_str(), 0755);
    tilesDirFd = open((P.outDir + "/tiles").c_str(), O_RDONLY | O_DIRECTORY);
    if (tilesDirFd < 0) die("cannot open tiles dir in %s", P.outDir.c_str());
    if (P.resume) {
      std::string err;
      if (!checkResumeManifest(err)) die("resume refused: %s", err.c_str());
      manifestF = fopen((P.outDir + "/manifest.txt").c_str(), "a");
    } else {
      std::string mp = P.outDir + "/manifest.txt";
      FILE *probe = fopen(mp.c_str(), "r");
      if (probe) { fclose(probe); die("%s exists; use --resume or a fresh --outdir", mp.c_str()); }
      // refuse to write into a dir with stale tiles (a foreign config's records
      // would be merged into results.jsonl without warning)
      DIR *d = opendir((P.outDir + "/tiles").c_str());
      if (d) {
        while (dirent *e = readdir(d)) {
          std::string nn = e->d_name;
          if (nn.size() > 6 && (nn.substr(nn.size() - 6) == ".jsonl" ||
                                (nn.size() > 4 && nn.substr(nn.size() - 4) == ".tmp")))
            die("stale tile file %s present; use --resume or a fresh --outdir", nn.c_str());
        }
        closedir(d);
      }
      manifestF = fopen(mp.c_str(), "w");
      if (manifestF) {
        if (fprintf(manifestF, "config %s %s\n", P.configHash().c_str(), P.configString().c_str()) < 0 ||
            fprintf(manifestF, "total %zu\n", tileMass.size()) < 0 ||
            fflush(manifestF) != 0 || fsync(fileno(manifestF)) != 0)
          die("manifest header write failed");
        int dfd = open(P.outDir.c_str(), O_RDONLY | O_DIRECTORY);
        if (dfd >= 0) { fsync(dfd); close(dfd); }
      }
    }
    if (!manifestF) die("cannot open manifest in %s", P.outDir.c_str());
#endif
  }

  void commitTile(int i, TileBuf &tb) {
#pragma omp critical(totals)
    {
      total.iters += tb.st.iters; total.chainPass += tb.st.chainPass;
      total.keySearches += tb.st.keySearches; total.keyMatches += tb.st.keyMatches;
      total.recoveries += tb.st.recoveries; total.sampledProbes += tb.st.sampledProbes;
      total.exactHits += tb.st.exactHits; total.nearHits += tb.st.nearHits;
      total.gapReports += tb.st.gapReports; total.degenerateSkipped += tb.st.degenerateSkipped;
      total.aIters += tb.st.aIters;
    }
    if (P.inMemoryOnly) return;
#ifdef __unix__
    // per-tile stats record (merge aggregates these; resume-safe)
    char sb[4096];
    int n = snprintf(sb, sizeof sb,
                     "{\"type\":\"tilestats\",\"tile\":%d,\"iters\":%" PRIu64 ",\"chainPass\":%" PRIu64
                     ",\"keySearches\":%" PRIu64 ",\"sampledProbes\":%" PRIu64 ",\"hist\":[",
                     i, tb.st.iters, tb.st.chainPass, tb.st.keySearches, tb.st.sampledProbes);
    bool first = true;
    for (int b = 0; b < 128 && n < (int)sizeof sb - 64; b++)
      if (tb.hist[b]) {
        n += snprintf(sb + n, sizeof sb - n, "%s[%d,%" PRIu64 "]", first ? "" : ",", b, tb.hist[b]);
        first = false;
      }
    snprintf(sb + n, sizeof sb - n, "]}");
    atomicWrite(tilePath(i, true), tilePath(i, false), tb.lines, sb);
#pragma omp critical(manifest)
    {
      if (fprintf(manifestF, "done %d\n", i) < 0 || fflush(manifestF) != 0 ||
          fsync(fileno(manifestF)) != 0)
        die("manifest write failed (disk full?)");
    }
#endif
  }

  // ---------- orchestration ----------
  void banner() {
    if (P.quiet) return;
    fprintf(stderr, "%s\n  config sha256=%s\n", P.configString().c_str(), P.configHash().c_str());
    fprintf(stderr, "  bases in [1,%" PRIu64 ")  pairs stored: %" PRIu64 " (%.2f GB keys + %.2f GB bloom)  keyShift=%d  build %.1fs\n",
            P.B, nPairs, nPairs * 8.0 / (1ull << 30), bloomBlocks * 64.0 / (1ull << 30), keyShift, pairBuildSec);
    fprintf(stderr, "  structural (q,p):");
    for (auto qp : structQP) fprintf(stderr, " (%u,%u)", qp.q, qp.p);
    fprintf(stderr, "  strideM=%u classDensity=%.4f mode=%s\n", strideM, strideDensity,
            P.near == 1 ? "near" : "exact-primitive");
    fprintf(stderr, "  chain:");
    double cum = 1;
    for (auto &M : chain) { cum *= M.pass; fprintf(stderr, " %u(%.3f)", M.m, M.pass); }
    fprintf(stderr, "  cumulative~%.5f (uniform-triple estimate)\n", cum);
    fprintf(stderr, "  tiles=%zu sampleShift=%d deltaBits=%d\n", tileMass.size(), P.sampleShift, P.deltaBits);
  }

  void run() {
    double t0 = omp_get_wtime();
    if (P.near != 0 && P.near != 1) die("--near 0|1 must be set explicitly");
    if (P.threads) omp_set_num_threads(P.threads);
    buildPow();
    buildStructural();
    buildChain();
    buildTiles();
    double keysGB = (double)Bmax * (Bmax + 1) / 2 * (P.noBloom ? 8.0 : 9.5) / (1ull << 30) * (P.near ? 1.0 : 0.70);
    double avail = P.memGB ? P.memGB : memAvailableGB();
    // guard only matters at production scale; selftest/bench tables are tiny
    if (keysGB > 4 && avail > 0 && keysGB + 2 > avail - 4 && !P.forceMem)
      die("estimated %.0f GB of keys vs %.0f GB available; lower --B or --force-mem", keysGB, avail);
    openFiles();
    buildPairs();
    // persist the pair-scan (dup22) records so a standalone merge after a crash
    // still has them; deterministic per config, so resume overwrite is harmless
    if (!P.inMemoryOnly)
      atomicWrite(P.outDir + "/tiles/pairscan.jsonl.tmp", P.outDir + "/tiles/pairscan.jsonl",
                  pairScanLines, "");
    banner();

    std::atomic<int> processed{0}, tilesDoneNow{0};
    std::atomic<u64> massDoneMicro{0};
    double lastPrint = omp_get_wtime();
    int nT = (int)tileMass.size(), already = 0;
    for (int i = 0; i < nT; i++) already += tileDone[i];

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < nT; i++) {
      if (tileDone[i]) continue;
      if (P.maxTiles && processed.fetch_add(1) >= P.maxTiles) continue;
      if (!P.maxTiles) processed.fetch_add(1);
      TileBuf tb;
      runTile(tileLo[i], tileLo[i + 1], tb);
      commitTile(i, tb);
      tileDone[i] = 1;
      tilesDoneNow++;
      massDoneMicro += (u64)(tileMass[i] / massTotal * 1e6);
      if (!P.quiet) {
#pragma omp critical(progress)
        {
          double now = omp_get_wtime();
          if (now - lastPrint > 15) {
            lastPrint = now;
            double frac = massDoneMicro.load() / 1e6;
            double el = now - t0;
            fprintf(stderr, "  [%d/%d tiles this run, +%d prior] elapsed %.0fs ETA(this run) %.0fs\n",
                    tilesDoneNow.load(), nT - already, already, el,
                    frac > 1e-9 ? el * (1 - frac - (double)already / nT) / frac : 0);
          }
        }
      }
    }
    if (manifestF) { fclose(manifestF); manifestF = nullptr; }
    bool allDone = true;
    for (int i = 0; i < nT; i++) allDone = allDone && tileDone[i];
    (void)0;
    if (!P.quiet) {
      double el = omp_get_wtime() - t0;
      fprintf(stderr,
              "run finished in %.1fs (%s): iters=%" PRIu64 " chainPass=%" PRIu64 " (%.4f%%) keySearches=%" PRIu64
              " keyMatches=%" PRIu64 " recoveries=%" PRIu64 "\n",
              el, allDone ? "ALL TILES DONE" : "partial", total.iters, total.chainPass,
              total.iters ? 100.0 * total.chainPass / total.iters : 0.0, total.keySearches, total.keyMatches,
              total.recoveries);
      fprintf(stderr, "hits: exact=%" PRIu64 " near=%" PRIu64 " gapReports=%" PRIu64 " sampledProbes=%" PRIu64
                      " degenerateSkipped=%" PRIu64 "\n",
              total.exactHits, total.nearHits, total.gapReports, total.sampledProbes, total.degenerateSkipped);
    if (gapDropped)
      fprintf(stderr, "WARNING: %" PRIu64 " gap records exceeded the in-memory mirror cap "
                      "(tile files are complete; in-memory comparisons are NOT)\n", gapDropped);
    }
    if (!P.inMemoryOnly && allDone) merge();
  }

  // merge tile files -> results.jsonl (deduplicated) + stats.json + sha256.
  // Self-sufficient: config identity and the done-tile audit come from the
  // manifest, so a standalone post-crash merge is safe (dup22 records are on
  // disk in tiles/pairscan.jsonl).
  void merge() {
#ifdef __unix__
    // manifest: authoritative config string + done-tile audit
    std::string cfgStr = P.configString(), cfgHash = P.configHash(), gapPop = "unknown";
    {
      FILE *mf = fopen((P.outDir + "/manifest.txt").c_str(), "r");
      if (!mf) die("merge: no manifest.txt in %s", P.outDir.c_str());
      char line[600];
      std::vector<int> doneIds;
      size_t totalTiles = 0;
      while (fgets(line, sizeof line, mf)) {
        if (!strchr(line, '\n')) continue;
        char hash[80];
        int tid;
        size_t tot;
        if (sscanf(line, "config %64s", hash) == 1 && strlen(hash) == 64) {
          cfgHash = hash;
          const char *rest = strchr(line, ' ');
          rest = rest ? strchr(rest + 1, ' ') : nullptr;
          if (rest) {
            cfgStr = rest + 1;
            while (!cfgStr.empty() && (cfgStr.back() == '\n' || cfgStr.back() == '\r')) cfgStr.pop_back();
          }
        } else if (sscanf(line, "total %zu", &tot) == 1)
          totalTiles = tot;
        else if (sscanf(line, "done %d", &tid) == 1)
          doneIds.push_back(tid);
      }
      fclose(mf);
      gapPop = cfgStr.find("near=1") != std::string::npos ? "all-pairs" : "admissible-pairs-only";
      // completeness gate: a merged results.jsonl must cover every tile, or
      // a dual-engine/dual-GPU partial run could silently masquerade as done
      if (totalTiles > 0) {
        std::vector<u8> seen(totalTiles, 0);
        for (int id : doneIds)
          if (id >= 0 && (size_t)id < totalTiles) seen[id] = 1;
        size_t missing = 0;
        for (size_t i = 0; i < totalTiles; i++) missing += !seen[i];
        if (missing)
          die("merge: %zu of %zu tiles not marked done — incomplete run; rerun search --resume before merging",
              missing, totalTiles);
      } else
        fprintf(stderr, "merge WARNING: manifest has no 'total' line (pre-1.5.0 run) — completeness not verifiable\n");
      for (int id : doneIds) {
        FILE *tf = fopen(tilePath(id, false).c_str(), "r");
        if (!tf)
          die("manifest marks tile %d done but its file is missing — durability violation; rerun search --resume", id);
        fclose(tf);
      }
    }
    std::string tdir = P.outDir + "/tiles";
    DIR *d = opendir(tdir.c_str());
    if (!d) die("no tiles dir in %s", P.outDir.c_str());
    std::vector<std::string> files;
    while (dirent *e = readdir(d)) {
      std::string n = e->d_name;
      if (n.size() > 6 && n.substr(n.size() - 6) == ".jsonl") files.push_back(tdir + "/" + n);
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    std::set<std::string> records;
    u64 mergedHist[128] = {0};
    u64 mIters = 0, mChainPass = 0, mKeySearches = 0, mSampled = 0;
    for (auto &fp : files) {
      FILE *f = fopen(fp.c_str(), "r");
      if (!f) die("merge: cannot read %s", fp.c_str());
      char line[4096];
      while (fgets(line, sizeof line, f)) {
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!L) continue;
        if (strstr(line, "\"tilestats\"")) {
          u64 it = 0, cp = 0, ks = 0, sp = 0;
          const char *q;
          if ((q = strstr(line, "\"iters\":"))) sscanf(q, "\"iters\":%" SCNu64, &it);
          if ((q = strstr(line, "\"chainPass\":"))) sscanf(q, "\"chainPass\":%" SCNu64, &cp);
          if ((q = strstr(line, "\"keySearches\":"))) sscanf(q, "\"keySearches\":%" SCNu64, &ks);
          if ((q = strstr(line, "\"sampledProbes\":"))) sscanf(q, "\"sampledProbes\":%" SCNu64, &sp);
          mIters += it; mChainPass += cp; mKeySearches += ks; mSampled += sp;
          const char *p = strstr(line, "\"hist\":[");
          if (p) {
            p += 8;
            while (*p == '[') {
              int b; u64 c;
              if (sscanf(p, "[%d,%" SCNu64 "]", &b, &c) == 2 && b >= 0 && b < 128) mergedHist[b] += c;
              p = strchr(p, ']');
              if (!p) break;
              p++;
              if (*p == ',') p++;
            }
          }
        } else records.insert(line);
      }
      fclose(f);
    }
    std::string rp = P.outDir + "/results.jsonl";
    std::string rtmp = rp + ".tmp";
    FILE *rf = fopen(rtmp.c_str(), "w");
    if (!rf) die("merge: cannot write %s", rtmp.c_str());
    SHA256 rh;
    for (auto &r : records) {
      if (fputs(r.c_str(), rf) < 0 || fputc('\n', rf) < 0) die("merge: write failed");
      rh.update(r.data(), r.size());
      rh.update("\n", 1);
    }
    if (fflush(rf) != 0 || fsync(fileno(rf)) != 0 || fclose(rf) != 0) die("merge: flush failed");
    if (rename(rtmp.c_str(), rp.c_str()) != 0) die("merge: rename failed");
    std::string rhash = rh.final();
    FILE *hf = fopen((rp + ".sha256").c_str(), "w");
    if (!hf || fprintf(hf, "%s  results.jsonl\n", rhash.c_str()) < 0 || fclose(hf) != 0)
      die("merge: sha256 write failed");
    FILE *sf = fopen((P.outDir + "/stats.json").c_str(), "w");
    if (!sf) die("merge: stats write failed");
    fprintf(sf, "{\"version\":\"%s\",\"config\":\"%s\",\"configSha256\":\"%s\",\"records\":%zu"
                ",\"iters\":%" PRIu64 ",\"chainPass\":%" PRIu64 ",\"keySearches\":%" PRIu64
                ",\"sampledProbes\":%" PRIu64 ",\"gapPopulation\":\"%s\",\"nearestKeyGapHistogramLog2\":[",
            CODE_VERSION, cfgStr.c_str(), cfgHash.c_str(), records.size(),
            mIters, mChainPass, mKeySearches, mSampled, gapPop.c_str());
    for (int i = 0; i < 128; i++) fprintf(sf, "%s%" PRIu64, i ? "," : "", mergedHist[i]);
    // NOTE: bins are log2 of the distance between TRUNCATED KEYS shifted back,
    // not exact nearest-gap distances. Bin 0 means "a pair sum shares the
    // probe's key bucket", NOT "|delta| <= 1". Emitted gap records are exact;
    // this histogram is key-quantized. Hence the field name.
    fprintf(sf, "],\"nearestKeyGapHistogramLog2_note\":\"key-quantized: bin 0 = same truncated-key bucket, not exact gap 0\"}\n");
    if (fclose(sf) != 0) die("merge: stats close failed");
    if (!P.quiet)
      fprintf(stderr, "merged %zu records -> %s (sha256 %s)\n", records.size(), rp.c_str(), rhash.c_str());
#endif
  }
};

// ---------------------------------------------------------------- oracle (C++)

struct OracleResult {
  std::set<Hit> hits;
  std::set<GapHit> gaps;
};

// deliberately written independently of Engine: plain maps, no filters, no keys
static OracleResult oracle(const Params &P) {
  OracleResult R;
  u64 Bmax = P.B - 1;
  int K = P.K;
  std::vector<u128> POW(Bmax + 1);
  POW[0] = 0;
  for (u64 i = 1; i <= Bmax; i++) { u128 v = 1; for (int j = 0; j < K; j++) v *= i; POW[i] = v; }
  // oracle-side admissibility (exact campaign): re-derive the structural primes
  // with its own logic
  std::vector<std::pair<u32, u32>> qps;
  if (P.near == 0) {
    for (auto qp : std::vector<std::pair<u32, u32>>{{16, 2}, {8, 2}, {27, 3}, {9, 3}, {49, 7}, {7, 7},
                                                    {25, 5}, {5, 5}, {13, 13}, {11, 11}, {3, 3}}) {
      bool have = false;
      for (auto &e : qps) have = have || e.second == qp.second;
      if (have) continue;
      bool ok = true;
      for (u64 x = 0; x < qp.first && ok; x++)
        ok = (powmod(x, K, qp.first) == (x % qp.second == 0 ? 0u : 1u));
      if (ok) qps.push_back(qp);
    }
    u64 prod = 1;
    std::vector<std::pair<u32, u32>> folded;
    for (auto &e : qps)
      if (e.first > 3 && prod * e.second <= 64) { prod *= e.second; folded.push_back(e); } // q>3: same soundness guard as the engine
    qps = folded;
  }
  auto adm = [&](u64 a, u64 b) {
    for (auto &e : qps)
      if (a % e.second == 0 && b % e.second == 0) return false;
    return true;
  };
  // hits are checked against the FULL pair population (ground truth); the
  // admissibility-restricted map exists only to mirror the engine's gap-stream
  // population in the exact campaign. Never filter the hit path by the
  // engine's own pruning rule — that blinds the oracle to pruning bugs.
  std::map<u128, std::vector<std::pair<u64, u64>>> pairs, pairsAdm;
  for (u64 a = 1; a <= Bmax; a++)
    for (u64 b = 1; b <= a; b++) {
      pairs[POW[a] + POW[b]].push_back({a, b});
      if (P.near == 0 && adm(a, b)) pairsAdm[POW[a] + POW[b]].push_back({a, b});
    }
  auto &gapPairs = (P.near == 0) ? pairsAdm : pairs;
  for (auto &kv : pairs)
    for (size_t i = 0; i + 1 < kv.second.size(); i++)
      R.hits.insert(Hit{1, kv.second[i + 1].first, kv.second[i + 1].second, 0,
                        kv.second[i].first, kv.second[i].second, 0});
  u128 maxPair = 2 * POW[Bmax];
  for (u64 x = 1; x <= Bmax; x++)
    for (u64 y = 1; y <= x; y++) {
      bool doSample = sampledXY(x, y, P.sampleShift);
      for (u64 z = 1; z <= y; z++) {
        u128 s3 = POW[x] + POW[y] + POW[z];
        if (s3 > maxPair + (P.near == 1 ? 1 : 0)) break;
        for (int diff = -1; diff <= 1; diff++) {
          if (diff != 0 && P.near != 1) continue;
          u128 v = diff == 1 ? s3 - 1 : (diff == -1 ? s3 + 1 : s3);
          auto it = pairs.find(v);
          if (it != pairs.end())
            for (auto &ab : it->second) {
              if (x == ab.first || x == ab.second || y == ab.first || y == ab.second ||
                  z == ab.first || z == ab.second)
                continue; // reducible: sides share a base
              R.hits.insert(Hit{0, x, y, z, ab.first, ab.second, diff});
            }
        }
        if (doSample) {
          u128 delta = (u128)1 << P.deltaBits;
          u128 lo = s3 > delta ? s3 - delta : 2;
          for (auto it = gapPairs.lower_bound(lo); it != gapPairs.end() && it->first <= s3 + delta; ++it) {
            if (it->first == s3) continue;
            bool neg = it->first < s3;
            u128 d = neg ? s3 - it->first : it->first - s3;
            if (d > delta || d == 0) continue;
            for (auto &ab : it->second) {
              if (x == ab.first || x == ab.second || y == ab.first || y == ab.second ||
                  z == ab.first || z == ab.second)
                continue;
              R.gaps.insert(GapHit{x, y, z, ab.first, ab.second, neg, d});
            }
          }
        }
      }
    }
  return R;
}

// ---------------------------------------------------------------- selftest

static int failures = 0;
#define CHECK(cond, ...) \
  do { \
    if (!(cond)) { \
      failures++; \
      fprintf(stderr, "FAIL %d: ", __LINE__); \
      fprintf(stderr, __VA_ARGS__); \
      fprintf(stderr, "\n"); \
    } \
  } while (0)

static u64 gcd2(u64 a, u64 b) { while (b) { u64 t = a % b; a = b; b = t; } return a; }
static u64 gcd5(const Hit &h) { return gcd2(gcd2(gcd2(h.x, h.y), gcd2(h.z, h.a)), h.b); }

static Engine *runEngine(const Params &Pin) {
  Params P = Pin;
  P.quiet = true;
  Engine *e = new Engine(P);
  e->run();
  return e;
}

static void oracleEquivalence(int K, u64 B, int near, int keyShiftExtra, int sampleShift, int deltaBits) {
  Params P;
  P.K = K; P.B = B; P.near = near; P.keyShiftExtra = keyShiftExtra;
  P.sampleShift = sampleShift; P.deltaBits = deltaBits; P.inMemoryOnly = true; P.tiles = 7;
  Engine *e = runEngine(P);
  OracleResult ora = oracle(P);
  CHECK(ora.gaps.size() < 2000000, "TEST BUG: deltaBits=%d floods the gap stream at K=%d B=%" PRIu64
        " (%zu records) — shrink it", deltaBits, K, B, ora.gaps.size());
  CHECK(e->gapDropped == 0, "TEST BUG: engine gap mirror overflowed (K=%d B=%" PRIu64 ")", K, B);
  std::set<Hit> eng(e->hits.begin(), e->hits.end());
  std::set<GapHit> egaps(e->gapHits.begin(), e->gapHits.end());
  if (near == 1) {
    CHECK(eng == ora.hits, "K=%d B=%" PRIu64 " near: engine %zu vs oracle %zu hits", K, B, eng.size(), ora.hits.size());
    if (eng != ora.hits) {
      for (auto &h : ora.hits)
        if (!eng.count(h))
          fprintf(stderr, "  missing: k%d %" PRIu64 ",%" PRIu64 ",%" PRIu64 "=%" PRIu64 ",%" PRIu64 " d%lld\n",
                  h.kind, h.x, h.y, h.z, h.a, h.b, h.diff);
      for (auto &h : eng)
        if (!ora.hits.count(h))
          fprintf(stderr, "  spurious: k%d %" PRIu64 ",%" PRIu64 ",%" PRIu64 "=%" PRIu64 ",%" PRIu64 " d%lld\n",
                  h.kind, h.x, h.y, h.z, h.a, h.b, h.diff);
    }
    CHECK(egaps == ora.gaps, "K=%d B=%" PRIu64 " gaps: engine %zu vs oracle %zu", K, B, egaps.size(), ora.gaps.size());
  } else {
    // exact campaign: every engine (3,2) report must be a genuine exact solution;
    // every primitive exact solution must be reported. gap stream compared
    // against the admissibility-aware oracle.
    std::set<Hit> oraExact, oraPrim;
    for (auto &h : ora.hits)
      if (h.kind == 0) {
        oraExact.insert(h);
        if (gcd5(h) == 1) oraPrim.insert(h);
      }
    for (auto &h : eng)
      if (h.kind == 0) {
        CHECK(h.diff == 0, "exact campaign produced a near record");
        CHECK(oraExact.count(h), "spurious exact hit");
      }
    for (auto &h : oraPrim)
      CHECK(eng.count(h), "missed primitive %" PRIu64 ",%" PRIu64 ",%" PRIu64 "=%" PRIu64 ",%" PRIu64,
            h.x, h.y, h.z, h.a, h.b);
    CHECK(egaps == ora.gaps, "K=%d B=%" PRIu64 " exact-mode gaps: %zu vs %zu", K, B, egaps.size(), ora.gaps.size());
  }
  delete e;
}

static void selftest() {
  // SHA-256 FIPS vectors — everything downstream trusts this
  CHECK(SHA256::of("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 empty");
  CHECK(SHA256::of("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
  // dec()
  CHECK(dec(0) == "0", "dec0");
  CHECK(dec((u128)1234567890123456789ull) == "1234567890123456789", "dec1");
  {
    u128 v = (u128)10000000000000000000ull * (u128)10000000000000000000ull;
    CHECK(dec(v) == "100000000000000000000000000000000000000", "dec 1e38, got %s", dec(v).c_str());
  }
  // rootFloor exhaustive-property test
  {
    Params P; P.K = 6; P.B = 5001; P.near = 1; P.inMemoryOnly = true; P.quiet = true;
    Engine e(P);
    e.buildPow();
    u64 rng = 12345;
    for (int i = 0; i < 200000; i++) {
      rng = sm64(rng);
      u128 v = (((u128)rng << 64) | sm64(rng ^ 7)) % (3 * e.POW[e.Bmax]);
      u64 r = e.rootFloor(v);
      CHECK(e.POW[r] <= v && (r == e.Bmax || e.POW[r + 1] > v), "rootFloor");
    }
    for (u64 i = 1; i <= e.Bmax; i++) { // exact boundaries
      CHECK(e.rootFloor(e.POW[i]) == i, "rootFloor at power");
      CHECK(e.rootFloor(e.POW[i] - 1) == i - 1, "rootFloor below power");
    }
  }
  // structural detection K=6: (8,2),(9,3),(7,7); strideM=42; density exactly 9/49
  {
    Params P; P.K = 6; P.B = 101; P.near = 0; P.inMemoryOnly = true; P.quiet = true;
    Engine e(P);
    e.buildPow(); e.buildStructural();
    CHECK(e.strideM == 42, "K=6 strideM=%u", e.strideM);
    CHECK(e.structComposite == 504, "K=6 composite=%u", e.structComposite);
    CHECK(std::fabs(e.strideDensity - 0.75 * (2.0 / 3.0) * (126.0 / 343.0)) < 1e-9,
          "K=6 class density %.6f", e.strideDensity);
  }
  // EXHAUSTIVE class-exclusion justification for EVERY K exercised below:
  // each excluded (cx,cy,cz) class must have NO admissible pair class matching
  // all structural congruences, and each allowed class must have a witness.
  // Residues of x^K mod q depend only on x mod p (verified by the detection
  // loop itself, exhaustively over [0,q)), so class representatives suffice.
  for (int KK : {2, 3, 4, 6}) {
    Params P; P.K = KK; P.B = 101; P.near = 0; P.inMemoryOnly = true; P.quiet = true;
    Engine e(P);
    e.buildPow(); e.buildStructural();
    if (e.strideM == 1) continue; // no pruning => nothing to audit
    long long bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(dynamic)
    for (long long cxy = 0; cxy < (long long)e.strideM * e.strideM; cxy++) {
      u32 cx = (u32)(cxy / e.strideM), cy = (u32)(cxy % e.strideM);
      for (u32 cz = 0; cz < e.strideM; cz++) {
        bool excluded = !(e.classMask[(size_t)cx * e.strideM + cy] >> cz & 1);
        bool witness = false;
        for (u32 ca = 0; ca < e.strideM && !witness; ca++)
          for (u32 cb = 0; cb < e.strideM && !witness; cb++) {
            bool admis = true;
            for (auto qp : e.structQP)
              if (e.strideM % qp.p == 0 && ca % qp.p == 0 && cb % qp.p == 0) admis = false;
            if (!admis) continue;
            bool all = true;
            for (auto qp : e.structQP) {
              if (e.strideM % qp.p) continue;
              u32 l = (u32)((powmod(cx, e.K, qp.q) + powmod(cy, e.K, qp.q) + powmod(cz, e.K, qp.q)) % qp.q);
              u32 rr = (u32)((powmod(ca, e.K, qp.q) + powmod(cb, e.K, qp.q)) % qp.q);
              if (l != rr) { all = false; break; }
            }
            if (all) witness = true;
          }
        if (excluded && witness) bad++;   // excluded but a pair class works: UNSOUND
        if (!excluded && !witness) bad++; // allowed but nothing can match: wasteful (and suspicious)
      }
    }
    CHECK(bad == 0, "K=%d class mask soundness: %lld inconsistent classes", KK, bad);
  }
  // exhaustive chain-filter no-false-negative over each finite residue domain,
  // both campaigns, every K exercised by the oracle sweeps
  for (int KK : {2, 3, 4, 6})
  for (int near = 0; near <= 1; near++) {
    Params P; P.K = KK; P.B = 4001; P.near = near; P.inMemoryOnly = true; P.quiet = true;
    Engine e(P);
    e.buildPow(); e.buildStructural(); e.buildChain();
    for (auto &M : e.chain) {
      for (u32 a = 0; a < M.m; a++)
        for (u32 b = 0; b < M.m; b++) {
          if (near == 0) {
            bool admis = true;
            for (auto qp : e.structQP)
              if (e.strideM % qp.p == 0 && M.m % qp.p == 0 && a % qp.p == 0 && b % qp.p == 0) admis = false;
            if (!admis) continue; // dropped pairs may be filtered; that is the point
          }
          u32 v = (u32)((powmod(a, e.K, M.m) + powmod(b, e.K, M.m)) % M.m);
          for (u32 t = 0; t < 3; t++) {
            CHECK(M.F[v + t * M.m] & 1, "m=%u exact false negative", M.m);
            if (near == 1) {
              CHECK(M.F[(v + 1) % M.m + t * M.m] & 2, "m=%u diff=+1 false negative", M.m);
              CHECK(M.F[(v + M.m - 1) % M.m + t * M.m] & 4, "m=%u diff=-1 false negative", M.m);
            }
          }
        }
    }
  }
  // oracle equivalence sweeps. K=2 is dense with genuine hits and exercises
  // every path; keyShiftExtra forces false-candidate storms through recovery.
  // deltaBits must stay small relative to the K,B value range or the gap
  // stream degenerates into the full triples-x-pairs cross product
  oracleEquivalence(2, 61, 1, 0, 0, 3);
  oracleEquivalence(2, 61, 1, 20, 0, 3);
  oracleEquivalence(2, 46, 1, 0, 2, 4);
  oracleEquivalence(3, 81, 1, 0, 3, 8);
  oracleEquivalence(4, 121, 0, 0, 2, 12); // exact campaign, generic strides (K=4 -> strideM=10)
  oracleEquivalence(4, 101, 1, 0, 4, 12);
  oracleEquivalence(6, 401, 1, 0, 4, 26);
  oracleEquivalence(6, 401, 0, 0, 3, 26);
  oracleEquivalence(6, 151, 1, 24, 6, 20);
  // K=6 near-miss rediscovery gates. Wroblewski's verified identities
  // (powersums.jorisperrenet.com/6th-powers/6-3-2-pm1/, re-verified exactly by
  // independent research): smallest is 147^6+92^6+71^6 = 133^6+132^6 + 1, and
  // exactly five exist with all bases < 1256 (Moore's exhaustive +/-1 search to
  // radius 17800 found only these five, all +1). Missing any of the five is a
  // hard failure; finding EXTRA ones is loudly reported but nonfatal — that
  // would be a discovery contradicting Moore's claim, not necessarily our bug.
  {
    Params P; P.K = 6; P.B = 148; P.near = 1; P.sampleShift = -1; P.inMemoryOnly = true; P.tiles = 7;
    Engine *e = runEngine(P);
    OracleResult ora = oracle(P);
    std::set<Hit> eng(e->hits.begin(), e->hits.end());
    CHECK(eng == ora.hits, "B=148 oracle equivalence");
    CHECK(eng.count(Hit{0, 147, 92, 71, 133, 132, 1}), "failed to rediscover 147,92,71 = 133,132 + 1");
    delete e;
  }
  {
    Params P; P.K = 6; P.B = 1256; P.near = 1; P.sampleShift = -1; P.inMemoryOnly = true;
    Engine *e = runEngine(P);
    const Hit known[5] = {{0, 147, 92, 71, 133, 132, 1},
                          {0, 295, 154, 75, 294, 173, 1},
                          {0, 311, 268, 159, 330, 55, 1},
                          {0, 556, 409, 197, 515, 500, 1},
                          {0, 1255, 815, 573, 1143, 1123, 1}};
    std::set<Hit> eng;
    u64 exacts = 0;
    for (auto &h : e->hits)
      if (h.kind == 0) { eng.insert(h); if (h.diff == 0) exacts++; }
    for (auto &h : known)
      CHECK(eng.count(h), "failed to rediscover known near miss with max base %" PRIu64, h.x);
    CHECK(exacts == 0, "exact (6,3,2) hit below 1256 would contradict Ekl 1998 — investigate before celebrating");
    if (eng.size() > 5)
      fprintf(stderr, "NOTE: %zu +/-1 records below 1256, literature says exactly 5 — if selftests otherwise pass, "
                      "this contradicts Moore's radius-17800 claim. Verify by hand immediately.\n", eng.size());
    delete e;
  }
  // gap-path witness regression (the sampled stream is NOT exhaustive by
  // design — 2^-sampleShift of columns — so this forces sampleShift=0 and
  // requires the known +1 witness to surface as a delta=+1 gap record in the
  // EXACT campaign, where +-1 full-rate hunting is deliberately off).
  // GapHit.neg==true means the pair sum lies BELOW s3, i.e. delta = lhs-rhs = +1.
  {
    Params P; P.K = 6; P.B = 148; P.near = 0; P.sampleShift = 0; P.deltaBits = 20;
    P.inMemoryOnly = true; P.tiles = 7;
    Engine *e = runEngine(P);
    bool found = false;
    for (auto &g : e->gapHits)
      if (g.x == 147 && g.y == 92 && g.z == 71 && g.a == 133 && g.b == 132 && g.neg && g.delta == 1)
        found = true;
    CHECK(found, "exact-campaign gap stream at sampleShift=0 missed the +1 witness");
    delete e;
  }
#ifdef __unix__
  // crash/resume/merge durability test with real files
  {
    char dir[128];
    snprintf(dir, sizeof dir, "./selftest_tmp_%d", (int)getpid());
    { char rm[160]; snprintf(rm, sizeof rm, "rm -rf %s", dir); if (system(rm)) {} } // idempotent reruns
    Params P;
    P.K = 6; P.B = 1501; P.near = 1; P.sampleShift = 4; P.deltaBits = 34;
    P.tiles = 48; P.outDir = dir; P.quiet = true; P.forceMem = true;
    { // partial run: 17 of 48 tiles
      Params P1 = P; P1.maxTiles = 17;
      Engine e1(P1); e1.run();
    }
    { // resume mismatch must be refused
      Params Pbad = P; Pbad.B = 1502; Pbad.resume = true;
      Engine eb(Pbad);
      eb.buildPow(); eb.buildStructural(); eb.buildChain(); eb.buildTiles();
      std::string err;
      CHECK(!eb.checkResumeManifest(err), "config mismatch not detected");
    }
    Params P2 = P; P2.resume = true;
    Engine e2(P2); e2.run(); // completes the rest; auto-merges
    // merged file must equal a fresh in-memory full run
    Params P3 = P; P3.inMemoryOnly = true; P3.outDir = "";
    Engine *e3 = runEngine(P3);
    std::set<std::string> expect;
    for (auto &h : e3->hits) expect.insert(e3->formatHit(h));
    for (auto &g : e3->gapHits) expect.insert(e3->formatGap(g));
    std::set<std::string> got;
    {
      FILE *f = fopen((std::string(dir) + "/results.jsonl").c_str(), "r");
      CHECK(f != nullptr, "merged results.jsonl missing");
      if (f) {
        char line[4096];
        while (fgets(line, sizeof line, f)) {
          size_t L = strlen(line);
          while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
          if (L) got.insert(line);
        }
        fclose(f);
      }
    }
    CHECK(got == expect, "resume+merge mismatch: got %zu vs expect %zu records", got.size(), expect.size());
    if (got != expect) {
      for (auto &s : expect) if (!got.count(s)) fprintf(stderr, "  missing: %s\n", s.c_str());
      for (auto &s : got) if (!expect.count(s)) fprintf(stderr, "  extra:   %s\n", s.c_str());
    }
    delete e3;
  }
#endif
  if (failures == 0) fprintf(stderr, "ALL SELFTESTS PASSED\n");
  else { fprintf(stderr, "%d SELFTEST FAILURES\n", failures); exit(1); }
}

// ---------------------------------------------------------------- bench

static void bench(std::vector<u64> Bs, bool noBloom) {
  fprintf(stderr, "%-9s %-6s %14s %16s %12s %10s %10s %12s  bloom=%s\n",
          "B", "mode", "pairs", "iters", "iters/s", "buildS", "probeS", "chainPass%", noBloom ? "off" : "on");
  for (u64 b : Bs)
    for (int near = 0; near <= 1; near++) {
      Params P;
      P.K = 6; P.B = b; P.near = near; P.sampleShift = 6; P.deltaBits = 40;
      P.noBloom = noBloom;
      P.inMemoryOnly = true; P.quiet = true;
      double t0 = omp_get_wtime();
      Engine e(P);
      e.run();
      double dt = omp_get_wtime() - t0;
      double probeS = dt - e.pairBuildSec;
      fprintf(stderr, "%-9" PRIu64 " %-6s %14" PRIu64 " %16" PRIu64 " %12.3g %10.1f %10.1f %12.5f\n",
              b, near ? "near" : "exact", e.nPairs, e.total.iters,
              e.total.iters / std::max(probeS, 1e-9), e.pairBuildSec, probeS,
              e.total.iters ? 100.0 * e.total.chainPass / e.total.iters : 0.0);
    }
  fprintf(stderr, "\niters scale ~B^3; pairs ~B^2 (8 bytes each). Extrapolate probe time from the largest B\n"
                  "whose key table exceeds L3 (cache behavior changes with table size), and re-check at each 2x.\n");
}

// ---------------------------------------------------------------- main

static std::vector<u64> parseList(const char *s) {
  std::vector<u64> v;
  u64 cur = 0;
  bool any = false;
  for (const char *p = s;; p++) {
    if (*p >= '0' && *p <= '9') { cur = cur * 10 + (u64)(*p - '0'); any = true; }
    else { if (any) v.push_back(cur); cur = 0; any = false; if (!*p) break; }
  }
  return v;
}

#ifndef LPS632_GPU_EMBED
int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s selftest | bench --B list | search --B n --near 0|1 [opts] | merge --outdir D\n",
            argv[0]);
    return 1;
  }
  std::string cmd = argv[1];
  Params P;
  std::vector<u64> Blist;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) die("missing value for %s", a.c_str());
      return argv[++i];
    };
    if (a == "--B") { Blist = parseList(next()); if (!Blist.empty()) P.B = Blist[0]; }
    else if (a == "--K") P.K = (int)parseInt(next(), 2, 12, "--K");
    else if (a == "--near") P.near = (int)parseInt(next(), 0, 1, "--near");
    else if (a == "--sample-shift") P.sampleShift = (int)parseInt(next(), -1, 63, "--sample-shift");
    else if (a == "--delta-bits") P.deltaBits = (int)parseInt(next(), 0, 120, "--delta-bits");
    else if (a == "--key-shift-extra") P.keyShiftExtra = (int)parseInt(next(), 0, 40, "--key-shift-extra");
    else if (a == "--threads") P.threads = (int)parseInt(next(), 0, 4096, "--threads");
    else if (a == "--tiles") P.tiles = (int)parseInt(next(), 0, 1000000, "--tiles");
    else if (a == "--max-tiles") P.maxTiles = (int)parseInt(next(), 0, 1000000, "--max-tiles");
    else if (a == "--mem-gb") P.memGB = atof(next());
    else if (a == "--force-mem") P.forceMem = true;
    else if (a == "--no-bloom") P.noBloom = true;
    else if (a == "--resume") P.resume = true;
    else if (a == "--quiet") P.quiet = true;
    else if (a == "--outdir") P.outDir = next();
    else die("unknown option %s", a.c_str());
  }
  if (cmd == "selftest") { selftest(); return 0; }
  if (cmd == "bench") {
    if (Blist.empty()) Blist = {1001, 5001, 10001};
    bench(Blist, P.noBloom);
    return 0;
  }
  if (cmd == "search") {
    if (!P.B) die("search requires --B");
    if (P.near != 0 && P.near != 1) die("search requires an explicit --near 0 (exact campaign) or --near 1 (near campaign)");
    if (P.outDir.empty()) {
      char buf[128];
      snprintf(buf, sizeof buf, "run_K%d_B%" PRIu64 "_%s", P.K, P.B, P.near ? "near" : "exact");
      P.outDir = buf;
    }
    Engine e(P);
    e.run();
    fprintf(stderr, "verify with: python3 verify.py %s/results.jsonl\n", P.outDir.c_str());
    return 0;
  }
  if (cmd == "merge") {
    if (P.outDir.empty()) die("merge requires --outdir");
    // merge() is self-sufficient: config identity comes from the manifest and
    // dup22 records from tiles/pairscan.jsonl on disk
    P.B = 3; P.near = 1; P.inMemoryOnly = false;
    Engine e(P);
    e.merge();
    return 0;
  }
  die("unknown command %s", cmd.c_str());
  return 1;
}
#endif // LPS632_GPU_EMBED

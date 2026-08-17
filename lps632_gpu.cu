// lps632_gpu.cu — CUDA probe engine for the (6,3,2) search.
// Phase B: value-space windows with ±1 halo, key-balanced boundaries hashed
// into the config identity, per-window VRAM-resident slices with upload
// digest verification, dual-GPU tile dispatch, parallel host verify + gap
// pass. See GPU_DESIGN.md v1 for the reviewed contract.
//
// Invariants: the GPU only generates candidates; every candidate is
// re-derived on the CPU in exact u128 via the embedded engine's recover().
// Tables are built by the selftested CPU code. Windows partition triples by
// s3; each window's key/bloom slice covers pair sums [winLo-1, winHi] so all
// three probe offsets of any owned triple are present (the halo).
//
// Build: nvcc -O2 -arch=sm_120 -Xcompiler -fopenmp -o lps632gpu lps632_gpu.cu
// Usage: ./lps632gpu gputest [--device N]
//        ./lps632gpu search --B n --near 0|1 [--windows W] [--devices 0,1]
//                    [--cand-cap M] [--auto-resume] [usual flags]
#define LPS632_GPU_EMBED
#include "lps632.cpp"

#include <cuda_runtime.h>
#include <atomic>
#include <memory>

typedef unsigned __int128 du128;

#define CUCHECK(x)                                                                       \
  do {                                                                                   \
    cudaError_t e_ = (x);                                                                \
    if (e_ != cudaSuccess) die("CUDA error %s at %s:%d", cudaGetErrorString(e_), __FILE__, __LINE__); \
  } while (0)

// ---------------------------------------------------------------- device side

struct DevChain {
  int nMods;
  int m[12];
  int rOff[12];
  int fOff[12];
};

struct DevCand { u32 x, y, z, flags; };

struct KernelArgs {
  const du128 *POW;
  const u64 *keys;      // window slice
  u64 sliceStart;       // global index of slice base (bucketOfs clamping)
  u64 sliceLen;
  const u64 *bucketOfs; // GLOBAL 65537-entry bucket table (global indices)
  const u64 *bloom;     // per-slice bloom (may be null)
  u64 bloomBlocks;
  int bucketShift;
  int keyShift;
  du128 maxPair;
  du128 maxProbe;
  du128 winLo, winHi;   // triple ownership: s3 in [winLo, winHi)
  u64 Bmax;
  int near1;
  const u16 *rAll;
  const u8 *fAll;
  DevChain chain;
  u64 xLo, xHi;
  const u64 *colBase;
  u64 nCols;
  DevCand *out;
  unsigned long long *outCount; // 64-bit: a 32-bit counter can wrap at 2^32
                                // and a wrapped value below the cap would look
                                // like success while candidates were dropped
  u64 outCap;
  unsigned long long *iterCount;
};

__device__ inline u64 d_rootFloor(const du128 *POW, u64 Bmax, du128 v) {
  if (v >= POW[Bmax]) return Bmax;
  u64 lo = 0, hi = Bmax;
  while (hi - lo > 1) {
    u64 mid = lo + (hi - lo) / 2;
    if (POW[mid] <= v) lo = mid; else hi = mid;
  }
  return lo;
}

__device__ inline bool d_bloomMay(const u64 *bloom, u64 bloomBlocks, u64 k) {
  u64 h = k + 0x9E3779B97F4A7C15ull;
  h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
  h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
  h ^= h >> 31;
  const u64 *w = &bloom[(u64)(((du128)h * bloomBlocks) >> 64) * 8];
  u64 h2 = h ^ 0xA5A5A5A5DEADBEEFull;
  h2 += 0x9E3779B97F4A7C15ull;
  h2 = (h2 ^ (h2 >> 30)) * 0xBF58476D1CE4E5B9ull;
  h2 = (h2 ^ (h2 >> 27)) * 0x94D049BB133111EBull;
  h2 ^= h2 >> 31;
  for (int i = 0; i < 4; i++) {
    u32 c = (u32)(h2 >> (9 * i)) & 511;
    if (!(w[c >> 6] >> (c & 63) & 1)) return false;
  }
  return true;
}

__device__ inline bool d_keyPresent(const KernelArgs &A, u64 k) {
  if (k > (u64)(A.maxPair >> A.keyShift)) return false;
  if (A.bloom && !d_bloomMay(A.bloom, A.bloomBlocks, k)) return false;
  // global bucket bounds clamped to the slice (keys are globally sorted, so a
  // window slice is a contiguous run and global bucket indices translate)
  u64 gLo = A.bucketOfs[k >> A.bucketShift];
  u64 gHi = A.bucketOfs[(k >> A.bucketShift) + 1];
  u64 lo = gLo > A.sliceStart ? gLo - A.sliceStart : 0;
  u64 hi = gHi > A.sliceStart ? gHi - A.sliceStart : 0;
  if (hi > A.sliceLen) hi = A.sliceLen;
  if (lo > hi) lo = hi;
  while (lo < hi) {
    u64 mid = lo + (hi - lo) / 2;
    u64 v = A.keys[mid];
    if (v < k) lo = mid + 1;
    else if (v > k) hi = mid;
    else return true;
  }
  return false;
}

#define SH_R_MAX 2048
#define SH_F_MAX 6144

__global__ void probeKernel(KernelArgs A) {
  __shared__ u16 shR[SH_R_MAX];
  __shared__ u8 shF[SH_F_MAX];
  {
    int total = 0;
    for (int c = 0; c < A.chain.nMods; c++) total = max(total, A.chain.rOff[c] + A.chain.m[c]);
    for (int i = threadIdx.x; i < total; i += blockDim.x) shR[i] = A.rAll[i];
    total = 0;
    for (int c = 0; c < A.chain.nMods; c++) total = max(total, A.chain.fOff[c] + 3 * A.chain.m[c]);
    for (int i = threadIdx.x; i < total; i += blockDim.x) shF[i] = A.fAll[i];
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const u64 warpId = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const u64 nWarps = ((u64)gridDim.x * blockDim.x) >> 5;
  unsigned long long iters = 0;

  for (u64 col = warpId; col < A.nCols; col += nWarps) {
    u64 lo = 0, hi = A.xHi - A.xLo;
    while (hi - lo > 1) {
      u64 mid = lo + (hi - lo) / 2;
      if (A.colBase[mid] <= col) lo = mid; else hi = mid;
    }
    const u64 x = A.xLo + lo;
    const u64 y = col - A.colBase[lo] + 1;
    const du128 C2 = A.POW[x] + A.POW[y];
    if (C2 + 1 > A.maxProbe) continue;
    // window ownership: s3 = C2 + z^6 in [winLo, winHi), plus s3 <= maxProbe, z <= y
    du128 room = A.maxProbe - C2;
    u64 zcap = d_rootFloor(A.POW, A.Bmax, room);
    if (zcap > y) zcap = y;
    u64 zLo = 1;
    if (A.winLo > C2 + 1) { // smallest z with C2 + z^6 >= winLo
      du128 need = A.winLo - C2;
      u64 zf = d_rootFloor(A.POW, A.Bmax, need - 1);
      zLo = zf + 1;
    }
    u64 zHi = zcap;
    if (C2 + 1 < A.winHi) { // largest z with C2 + z^6 < winHi
      du128 below = A.winHi - C2 - 1;
      u64 zf = d_rootFloor(A.POW, A.Bmax, below);
      if (zf < zHi) zHi = zf;
    } else
      continue; // even z=1 lands at/above winHi
    if (zLo > zHi) continue;

    u32 base[12];
    for (int c = 0; c < A.chain.nMods; c++) {
      int m = A.chain.m[c];
      base[c] = shR[A.chain.rOff[c] + (u32)(x % m)] + shR[A.chain.rOff[c] + (u32)(y % m)];
    }

    for (u64 z = zLo + lane; z <= zHi; z += 32) {
      iters++;
      u32 flags = 0xFF;
      for (int c = 0; c < A.chain.nMods; c++) {
        int m = A.chain.m[c];
        u32 idx = base[c] + shR[A.chain.rOff[c] + (u32)(z % m)];
        flags &= shF[A.chain.fOff[c] + idx];
      }
      if (!flags) continue;
      const du128 s3 = C2 + A.POW[z];
      const u64 k = (u64)(s3 >> A.keyShift);
      u32 found = 0;
      if (d_keyPresent(A, k)) found = flags & 7;
      else if (A.near1) {
        if ((flags & 2) && (u64)((s3 - 1) >> A.keyShift) != k &&
            d_keyPresent(A, (u64)((s3 - 1) >> A.keyShift)))
          found |= 2;
        if ((flags & 4) && (u64)((s3 + 1) >> A.keyShift) != k &&
            d_keyPresent(A, (u64)((s3 + 1) >> A.keyShift)))
          found |= 4;
      }
      if (found) {
        unsigned long long slot = atomicAdd(A.outCount, 1ull);
        if (slot < A.outCap) A.out[slot] = {(u32)x, (u32)y, (u32)z, found};
      }
    }
  }
  atomicAdd(A.iterCount, iters);
}

// ---------------------------------------------------------------- host side

struct Window {
  du128 lo, hi;      // triple ownership [lo, hi)
  u64 sliceStart, sliceEnd; // global key indices, halo included
};

// host-side slice bloom (mirrors Engine::bloomInsert against a local vector)
static void buildSliceBloom(const u64 *keys, u64 n, std::vector<u64> &bits, u64 &blocks) {
  blocks = std::max<u64>(1, (n * 12 + 511) / 512);
  bits.assign(blocks * 8, 0);
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)n; i++) {
    u64 k = keys[i];
    u64 h = sm64(k);
    u64 *w = &bits[(u64)(((du128)h * blocks) >> 64) * 8];
    u64 h2 = sm64(h ^ 0xA5A5A5A5DEADBEEFull);
    for (int j = 0; j < 4; j++) {
      u32 c = (u32)(h2 >> (9 * j)) & 511;
      __atomic_fetch_or(&w[c >> 6], 1ull << (c & 63), __ATOMIC_RELAXED);
    }
  }
}

struct DevCtx {
  int device = 0;
  u64 candCap = 0;
  du128 *dPOW = nullptr;
  u64 *dKeys = nullptr, *dBucketOfs = nullptr, *dBloom = nullptr, *dColBase = nullptr;
  u16 *dRAll = nullptr;
  u8 *dFAll = nullptr;
  DevCand *dOut = nullptr;
  unsigned long long *dOutCount = nullptr;
  unsigned long long *dIter = nullptr;
  u64 keysCapBytes = 0, bloomCapBytes = 0;

  void init(Engine &e, int dev, u64 cap, u64 maxSliceKeys, const std::vector<u16> &rAll,
            const std::vector<u8> &fAll) {
    device = dev;
    candCap = cap;
    CUCHECK(cudaSetDevice(device));
    CUCHECK(cudaMalloc(&dPOW, (e.Bmax + 1) * sizeof(du128)));
    CUCHECK(cudaMemcpy(dPOW, e.POW.data(), (e.Bmax + 1) * sizeof(du128), cudaMemcpyHostToDevice));
    keysCapBytes = maxSliceKeys * sizeof(u64);
    CUCHECK(cudaMalloc(&dKeys, keysCapBytes));
    bloomCapBytes = (std::max<u64>(1, (maxSliceKeys * 12 + 511) / 512)) * 64;
    CUCHECK(cudaMalloc(&dBloom, bloomCapBytes));
    CUCHECK(cudaMalloc(&dBucketOfs, e.bucketOfs.size() * sizeof(u64)));
    CUCHECK(cudaMemcpy(dBucketOfs, e.bucketOfs.data(), e.bucketOfs.size() * sizeof(u64),
                       cudaMemcpyHostToDevice));
    CUCHECK(cudaMalloc(&dRAll, rAll.size() * sizeof(u16)));
    CUCHECK(cudaMemcpy(dRAll, rAll.data(), rAll.size() * sizeof(u16), cudaMemcpyHostToDevice));
    CUCHECK(cudaMalloc(&dFAll, fAll.size()));
    CUCHECK(cudaMemcpy(dFAll, fAll.data(), fAll.size(), cudaMemcpyHostToDevice));
    CUCHECK(cudaMalloc(&dOut, candCap * sizeof(DevCand)));
    CUCHECK(cudaMalloc(&dOutCount, 8));
    CUCHECK(cudaMalloc(&dIter, 8));
    CUCHECK(cudaMalloc(&dColBase, (size_t)(2 + e.Bmax) * sizeof(u64)));
  }

  // upload a window slice + its bloom, then verify both by chunked readback
  // digest (ECC-less VRAM defense; crash-only on mismatch)
  void loadWindow(Engine &e, const Window &w, bool noBloom, u64 &bloomBlocksOut) {
    CUCHECK(cudaSetDevice(device));
    u64 n = w.sliceEnd - w.sliceStart;
    if (n * sizeof(u64) > keysCapBytes) die("slice exceeds device arena — raise --windows");
    const u64 *src = e.keys.data() + w.sliceStart;
    CUCHECK(cudaMemcpy(dKeys, src, n * 8, cudaMemcpyHostToDevice));
    u64 hs = 0, hx = 0;
    for (u64 i = 0; i < n; i++) { hs += src[i]; hx ^= src[i]; }
    verifyDigest(dKeys, n, hs, hx, "keys");
    bloomBlocksOut = 0;
    if (!noBloom) {
      std::vector<u64> bits;
      u64 blocks = 0;
      buildSliceBloom(src, n, bits, blocks);
      if (blocks * 64 > bloomCapBytes) die("slice bloom exceeds device arena");
      CUCHECK(cudaMemcpy(dBloom, bits.data(), blocks * 64, cudaMemcpyHostToDevice));
      u64 bs = 0, bx = 0;
      for (u64 i = 0; i < bits.size(); i++) { bs += bits[i]; bx ^= bits[i]; }
      verifyDigest(dBloom, bits.size(), bs, bx, "bloom");
      bloomBlocksOut = blocks;
    }
  }

  void verifyDigest(const u64 *dPtr, u64 n, u64 expectSum, u64 expectXor, const char *what) {
    const u64 CH = 32ull << 20;
    static thread_local std::vector<u64> buf;
    buf.resize(std::min(CH, n));
    u64 s = 0, xr = 0;
    for (u64 off = 0; off < n; off += CH) {
      u64 c = std::min(CH, n - off);
      CUCHECK(cudaMemcpy(buf.data(), dPtr + off, c * 8, cudaMemcpyDeviceToHost));
      for (u64 i = 0; i < c; i++) { s += buf[i]; xr ^= buf[i]; }
    }
    if (s != expectSum || xr != expectXor)
      die("device %s digest mismatch on GPU %d (VRAM corruption?)", what, device);
  }

  ~DevCtx() {
    for (void *p : std::initializer_list<void *>{dPOW, dKeys, dBucketOfs, dBloom, dRAll, dFAll,
                                                 dOut, dOutCount, dIter, dColBase})
      if (p) cudaFree(p);
  }
};

static void verifyCandidates(Engine &e, std::vector<DevCand> &cands, Engine::TileBuf &tb);
static void gapPass(Engine &e, u64 xLo, u64 xHi, Engine::TileBuf &tb);

struct GpuHarness {
  Engine &e;
  std::vector<u16> rAll;
  std::vector<u8> fAll;
  DevChain chain{};
  std::vector<Window> windows;
  u64 maxSliceKeys = 0;

  GpuHarness(Engine &eng) : e(eng) {}

  void flattenChain() {
    int rOff = 0, fOff = 0;
    chain.nMods = (int)e.chain.size();
    if (chain.nMods > 12) die("chain too long for device struct");
    for (int c = 0; c < chain.nMods; c++) {
      const Engine::Mod &M = e.chain[c];
      chain.m[c] = (int)M.m;
      chain.rOff[c] = rOff;
      chain.fOff[c] = fOff;
      for (u32 i = 0; i < M.m; i++) rAll.push_back(M.res[i]);
      for (u32 i = 0; i < 3 * M.m; i++) fAll.push_back(M.F[i]);
      rOff += M.m;
      fOff += 3 * M.m;
    }
    if (rOff > SH_R_MAX || fOff > SH_F_MAX)
      die("chain tables exceed shared-memory budget (%d/%d)", rOff, fOff);
  }

  // key-balanced window plan: boundaries from sorted keys at deterministic
  // rank cuts — a pure function of (keys, gpuWindows), which the config hash
  // covers via B/near/version/gpuWindows. Never derived from free VRAM.
  void planWindows(int nW) {
    windows.clear();
    du128 prevHi = 0;
    for (int w = 0; w < nW; w++) {
      Window win;
      win.lo = prevHi;
      if (w == nW - 1) win.hi = e.maxProbe + 2;
      else {
        u64 cutIdx = (u64)((double)e.nPairs * (w + 1) / nW);
        if (cutIdx >= e.nPairs) cutIdx = e.nPairs - 1;
        du128 v = (du128)e.keys[cutIdx] << e.keyShift;
        if (v <= win.lo) v = win.lo + 1; // degenerate at tiny B: keep windows ordered
        win.hi = v;
      }
      prevHi = win.hi;
      // slice with ±1 halo: pair values in [lo-1, hi]
      u64 kLo = win.lo <= 1 ? 0 : (u64)((win.lo - 1) >> e.keyShift);
      u64 kHi = (u64)(win.hi >> e.keyShift);
      win.sliceStart = std::lower_bound(e.keys.begin(), e.keys.end(), kLo) - e.keys.begin();
      win.sliceEnd = std::upper_bound(e.keys.begin(), e.keys.end(), kHi) - e.keys.begin();
      windows.push_back(win);
    }
    // coverage audit (review gate 4): slices jointly cover every key, windows
    // partition probe space
    if (windows.front().sliceStart != 0 || windows.back().sliceEnd != e.nPairs)
      die("window slice audit: ends not covered");
    for (size_t i = 1; i < windows.size(); i++) {
      if (windows[i].lo != windows[i - 1].hi) die("window audit: gap in triple space");
      if (windows[i].sliceStart > windows[i - 1].sliceEnd) die("window audit: slice gap");
    }
    maxSliceKeys = 1;
    for (auto &w : windows) maxSliceKeys = std::max(maxSliceKeys, w.sliceEnd - w.sliceStart);
  }

  // run one (window, x-range) batch on a prepared device.
  // On candidate-buffer overflow returns false with `overflowCount` set — the
  // caller subdivides and retries. Overflow NEVER silently drops candidates and
  // never marks a tile done: the batch is re-run in smaller pieces.
  bool tryRun(DevCtx &d, const Window &w, u64 bloomBlocks, u64 xLo, u64 xHi,
              std::vector<DevCand> &out, unsigned long long &iters, u64 &overflowCount) {
    CUCHECK(cudaSetDevice(d.device));
    std::vector<u64> colBase(xHi - xLo + 1);
    colBase[0] = 0;
    for (u64 i = 0; i < xHi - xLo; i++) colBase[i + 1] = colBase[i] + (xLo + i);
    CUCHECK(cudaMemcpy(d.dColBase, colBase.data(), colBase.size() * 8, cudaMemcpyHostToDevice));
    CUCHECK(cudaMemset(d.dOutCount, 0, 8));
    CUCHECK(cudaMemset(d.dIter, 0, 8));
    KernelArgs A{};
    A.POW = d.dPOW;
    A.keys = d.dKeys;
    A.sliceStart = w.sliceStart;
    A.sliceLen = w.sliceEnd - w.sliceStart;
    A.bucketOfs = d.dBucketOfs;
    A.bloom = bloomBlocks ? d.dBloom : nullptr;
    A.bloomBlocks = bloomBlocks;
    A.bucketShift = e.bucketShift;
    A.keyShift = e.keyShift;
    A.maxPair = e.maxPair;
    A.maxProbe = e.maxProbe;
    A.winLo = w.lo;
    A.winHi = w.hi;
    A.Bmax = e.Bmax;
    A.near1 = e.P.near == 1 ? 1 : 0;
    A.rAll = d.dRAll;
    A.fAll = d.dFAll;
    A.chain = chain;
    A.xLo = xLo;
    A.xHi = xHi;
    A.colBase = d.dColBase;
    A.nCols = colBase.back();
    A.out = d.dOut;
    A.outCount = d.dOutCount;
    A.outCap = d.candCap;
    A.iterCount = d.dIter;
    probeKernel<<<1024, 256>>>(A);
    CUCHECK(cudaGetLastError());
    CUCHECK(cudaDeviceSynchronize());
    unsigned long long cnt = 0;
    CUCHECK(cudaMemcpy(&cnt, d.dOutCount, 8, cudaMemcpyDeviceToHost));
    unsigned long long it = 0;
    CUCHECK(cudaMemcpy(&it, d.dIter, 8, cudaMemcpyDeviceToHost));
    iters = it;
    overflowCount = 0;
    if (cnt > d.candCap) { overflowCount = cnt; return false; }
    out.resize(cnt);
    if (cnt) {
      double td = omp_get_wtime();
      CUCHECK(cudaMemcpy(out.data(), d.dOut, (u64)cnt * sizeof(DevCand), cudaMemcpyDeviceToHost));
      xferSec += omp_get_wtime() - td;
      xferBytes += (u64)cnt * sizeof(DevCand);
    }
    return true;
  }
  double xferSec = 0;   // measured device->host candidate transfer
  u64 xferBytes = 0;
  double kernelSec = 0; // measured kernel time (incl. overflowed retries)
  double verifySec = 0; // measured host exact re-derivation

  // adaptive: run [xLo,xHi) in as few batches as the candidate buffer allows,
  // verifying each batch on the CPU before the next launch. Splits by equal
  // triple mass (~x^3) using the measured overflow factor, so one retry
  // normally suffices instead of repeated halving.
  void runRangeAdaptive(DevCtx &d, const Window &w, u64 bloomBlocks, u64 xLo, u64 xHi,
                        Engine::TileBuf &tb, unsigned long long &itersAcc) {
    std::vector<DevCand> cands;
    unsigned long long iters = 0;
    u64 over = 0;
    double tk = omp_get_wtime();
    bool ok = tryRun(d, w, bloomBlocks, xLo, xHi, cands, iters, over);
    kernelSec += omp_get_wtime() - tk;
    if (ok) {
      itersAcc += iters;
      tb.st.keyMatches += cands.size();
      double tv = omp_get_wtime();
      verifyCandidates(e, cands, tb);
      verifySec += omp_get_wtime() - tv;
      return;
    }
    if (xHi - xLo <= 1)
      die("single column x=%llu overflows the candidate buffer (count=%llu cap=%llu); "
          "raise --cand-cap — column-level subdivision is not implemented",
          (unsigned long long)xLo, (unsigned long long)over, (unsigned long long)d.candCap);
    int nSplit = (int)(over / d.candCap) + 2;
    if (nSplit > 4096) nSplit = 4096;
    if ((u64)nSplit > xHi - xLo) nSplit = (int)(xHi - xLo);
    // equal triple mass: mass(x) ~ x^2, so cut on cube roots
    double lo3 = (double)xLo * xLo * xLo, hi3 = (double)xHi * xHi * xHi;
    u64 prev = xLo;
    for (int i = 1; i <= nSplit; i++) {
      u64 cut = i == nSplit ? xHi : (u64)std::cbrt(lo3 + (hi3 - lo3) * i / nSplit);
      if (cut <= prev) cut = prev + 1;
      if (cut > xHi) cut = xHi;
      if (cut > prev) runRangeAdaptive(d, w, bloomBlocks, prev, cut, tb, itersAcc);
      prev = cut;
      if (prev >= xHi) break;
    }
  }
};

static void mergeParts(std::vector<Engine::TileBuf> &parts, Engine::TileBuf &tb) {
  for (auto &pt : parts) {
    tb.lines.insert(tb.lines.end(), pt.lines.begin(), pt.lines.end());
    tb.st.exactHits += pt.st.exactHits;
    tb.st.nearHits += pt.st.nearHits;
    tb.st.recoveries += pt.st.recoveries;
    tb.st.degenerateSkipped += pt.st.degenerateSkipped;
    tb.st.sampledProbes += pt.st.sampledProbes;
    tb.st.gapReports += pt.st.gapReports;
    tb.st.aIters += pt.st.aIters;
    for (int b = 0; b < 128; b++) tb.hist[b] += pt.hist[b];
  }
}

// parallel exact re-derivation of GPU candidates (the only path to a record)
static void verifyCandidates(Engine &e, std::vector<DevCand> &cands, Engine::TileBuf &tb) {
  if (cands.empty()) return;
  std::vector<Engine::TileBuf> parts(omp_get_max_threads());
#pragma omp parallel
  {
    Engine::TileBuf &pt = parts[omp_get_thread_num()];
    u64 aStart = tl_aIters; // measure real verify cost (a-loop iterations)
#pragma omp for schedule(dynamic, 4096)
    for (long long ci = 0; ci < (long long)cands.size(); ci++) {
      auto &c = cands[ci];
      du128 s3 = e.POW[c.x] + e.POW[c.y] + e.POW[c.z];
      e.recover(s3, (u8)(c.flags & 7), c.x, c.y, c.z, pt);
    }
    pt.st.aIters += tl_aIters - aStart;
  }
  mergeParts(parts, tb);
}

// sampled unfiltered gap pass on the host (identical code path to the CPU
// engine, so gap records stay byte-identical across engines)
static void gapPass(Engine &e, u64 xLo, u64 xHi, Engine::TileBuf &tb) {
  if (e.P.sampleShift < 0) return;
  std::vector<Engine::TileBuf> parts(omp_get_max_threads());
#pragma omp parallel
  {
    Engine::TileBuf &pt = parts[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 8)
    for (long long x = (long long)xLo; x < (long long)xHi; x++) {
      for (u64 y = 1; y <= (u64)x; y++) {
        if (!sampledXY((u64)x, y, e.P.sampleShift)) continue;
        du128 C2 = e.POW[x] + e.POW[y];
        if (C2 + 1 > e.maxProbe) break;
        u64 zcap = e.rootFloor(e.maxProbe - C2);
        if (zcap > y) zcap = y;
        for (u64 z = 1; z <= zcap; z++) e.gapProbe(C2 + e.POW[z], (u64)x, y, z, pt);
      }
    }
  }
  mergeParts(parts, tb);
}

// closed-form window count: pure function of B and campaign (never of VRAM)
static int defaultWindows(u64 B, int near) {
  double keys = (double)B * (B - 1) / 2 * (near ? 1.0 : 0.70);
  double bytes = keys * 9.5; // keys + slice bloom
  const double budget = 24e9;
  int w = (int)(bytes / budget) + 1;
  return w < 1 ? 1 : w;
}

static void gpuSearch(Params P, std::vector<int> devices, u64 candCap) {
  if (P.gpuWindows <= 0) P.gpuWindows = defaultWindows(P.B, P.near);
  Engine e(P);
  if (P.threads) omp_set_num_threads(P.threads);
  omp_set_dynamic(0);
  e.buildPow();
  e.buildStructural();
  e.buildChain();
  e.buildTiles();
  int XB = (int)e.tileMass.size();
  int nW = P.gpuWindows;
  // expand tile identity to (window, x-block): id = w*XB + b. The manifest's
  // total line and resume marks are in this namespace.
  {
    double m = e.massTotal / nW;
    std::vector<double> tm = e.tileMass;
    e.tileMass.resize((size_t)nW * XB);
    for (int w = 0; w < nW; w++)
      for (int b = 0; b < XB; b++) e.tileMass[(size_t)w * XB + b] = tm[b] / nW;
    (void)m;
    e.tileDone.assign((size_t)nW * XB, 0);
  }
  e.openFiles();
  e.buildPairs();
  if (!P.inMemoryOnly)
    e.atomicWrite(P.outDir + "/tiles/pairscan.jsonl.tmp", P.outDir + "/tiles/pairscan.jsonl",
                  e.pairScanLines, "");
  e.banner();

  GpuHarness g(e);
  g.flattenChain();
  g.planWindows(nW);
  if (!P.quiet)
    fprintf(stderr, "  gpu: %d windows, max slice %.2f GB keys, devices:%zu candCap=%u\n", nW,
            g.maxSliceKeys * 8.0 / 1e9, devices.size(), candCap);

  std::vector<std::unique_ptr<DevCtx>> ctx;
  for (int d : devices) {
    ctx.emplace_back(new DevCtx());
    ctx.back()->init(e, d, candCap, g.maxSliceKeys, g.rAll, g.fAll);
  }

  double t0 = omp_get_wtime();
  u64 totalIters = 0;
  int processed = 0;
  for (int w = 0; w < nW; w++) {
    if (P.maxTiles && processed >= P.maxTiles) break;
    // skip fully-done windows without touching the devices
    bool any = false;
    for (int b = 0; b < XB; b++) any = any || !e.tileDone[(size_t)w * XB + b];
    if (!any) continue;
    std::vector<u64> bloomBlocks(ctx.size(), 0);
    for (size_t d = 0; d < ctx.size(); d++) ctx[d]->loadWindow(e, g.windows[w], P.noBloom, bloomBlocks[d]);
    // round-robin x-blocks across devices; kernel on device d overlaps host
    // verification of the previous tile from device d^1
    struct Pending { int tile; std::vector<DevCand> cands; unsigned long long iters; u64 xLo, xHi; };
    std::vector<int> todo;
    for (int b = 0; b < XB; b++)
      if (!e.tileDone[(size_t)w * XB + b]) todo.push_back(b);
    for (size_t i = 0; i < todo.size(); i++) {
      if (P.maxTiles && processed >= P.maxTiles) break;
      int b = todo[i];
      int tile = w * XB + b;
      DevCtx &d = *ctx[i % ctx.size()];
      Engine::TileBuf tb;
      unsigned long long iters = 0;
      double tk = omp_get_wtime();
      g.runRangeAdaptive(d, g.windows[w], bloomBlocks[i % ctx.size()], e.tileLo[b], e.tileLo[b + 1],
                         tb, iters);
      double tv = omp_get_wtime();
      tb.st.iters = iters;
      totalIters += iters;
      // gap pass belongs to window 0 only (it is window-independent; running it
      // once per x-block across all windows would duplicate records)
      if (w == 0) gapPass(e, e.tileLo[b], e.tileLo[b + 1], tb);
      double tg = omp_get_wtime();
      e.commitTile(tile, tb);
      e.tileDone[tile] = 1;
      processed++;
      if (!P.quiet && (P.maxTiles || i % 64 == 0)) {
        double kS = g.kernelSec, vS = g.verifySec, xS = g.xferSec;
        g.kernelSec = g.verifySec = g.xferSec = 0;
        u64 xB = g.xferBytes; g.xferBytes = 0;
        fprintf(stderr,
                "  [w%d/%d blk %zu/%zu x=[%" PRIu64 ",%" PRIu64 ")] iters=%llu cands=%" PRIu64
                " (%.3g/iter) | kernel %.1fs xfer %.1fs (%.2f GB) verify %.1fs"
                " (%.3g rec/s, %.1f a-iters/rec) gap %.1fs | wall %.0fs\n",
                w + 1, nW, i, todo.size(), e.tileLo[b], e.tileLo[b + 1], (unsigned long long)iters,
                tb.st.keyMatches, iters ? (double)tb.st.keyMatches / iters : 0.0,
                kS - xS, xS, xB / 1e9, vS,
                vS > 0 ? tb.st.recoveries / vS : 0.0,
                tb.st.recoveries ? (double)tb.st.aIters / tb.st.recoveries : 0.0,
                tg - tv, omp_get_wtime() - t0);
      }
    }
  }
  if (e.manifestF) { fclose(e.manifestF); e.manifestF = nullptr; }
  bool allDone = true;
  for (auto v : e.tileDone) allDone = allDone && v;
  if (!P.quiet)
    fprintf(stderr, "gpu run %s in %.1fs: iters=%llu exact=%" PRIu64 " near=%" PRIu64
                    " gapReports=%" PRIu64 "\n",
            allDone ? "COMPLETE" : "partial", omp_get_wtime() - t0, (unsigned long long)totalIters,
            e.total.exactHits, e.total.nearHits, e.total.gapReports);
  if (!P.inMemoryOnly && allDone) e.merge();
}

// in-memory GPU-vs-CPU equivalence for one config (multi-window aware)
static int gpuVsCpu(int K, u64 B, int near, int sampleShift, int deltaBits, int device,
                    bool noBloom, int nWindows, u64 candCap = 8u << 20) {
  Params P;
  P.K = K; P.B = B; P.near = near; P.sampleShift = sampleShift; P.deltaBits = deltaBits;
  P.inMemoryOnly = true; P.quiet = true; P.tiles = 7; P.noBloom = noBloom;
  P.gpuWindows = nWindows;
  Engine eg(P);
  eg.buildPow(); eg.buildStructural(); eg.buildChain(); eg.buildTiles(); eg.buildPairs();
  GpuHarness g(eg);
  g.flattenChain();
  g.planWindows(nWindows);
  DevCtx d;
  d.init(eg, device, candCap, g.maxSliceKeys, g.rAll, g.fAll);
  unsigned long long itersTotal = 0;
  int XB = (int)eg.tileMass.size();
  for (int w = 0; w < nWindows; w++) {
    u64 bb = 0;
    d.loadWindow(eg, g.windows[w], noBloom, bb);
    for (int b = 0; b < XB; b++) {
      Engine::TileBuf tb;
      unsigned long long iters = 0;
      g.runRangeAdaptive(d, g.windows[w], bb, eg.tileLo[b], eg.tileLo[b + 1], tb, iters);
      itersTotal += iters;
      if (w == 0) gapPass(eg, eg.tileLo[b], eg.tileLo[b + 1], tb);
    }
  }
  Engine *ec = runEngine(P);
  std::set<Hit> hg(eg.hits.begin(), eg.hits.end()), hc(ec->hits.begin(), ec->hits.end());
  std::set<GapHit> gg(eg.gapHits.begin(), eg.gapHits.end()), gc(ec->gapHits.begin(), ec->gapHits.end());
  int bad = 0;
  if (hg != hc) {
    bad++;
    fprintf(stderr, "FAIL K=%d B=%" PRIu64 " near=%d W=%d: hits gpu=%zu cpu=%zu\n", K, B, near,
            nWindows, hg.size(), hc.size());
    for (auto &h : hc) if (!hg.count(h))
      fprintf(stderr, "  gpu missing: %" PRIu64 ",%" PRIu64 ",%" PRIu64 "=%" PRIu64 ",%" PRIu64
                      " d%lld\n", h.x, h.y, h.z, h.a, h.b, h.diff);
    for (auto &h : hg) if (!hc.count(h))
      fprintf(stderr, "  gpu spurious: %" PRIu64 ",%" PRIu64 ",%" PRIu64 "=%" PRIu64 ",%" PRIu64
                      " d%lld\n", h.x, h.y, h.z, h.a, h.b, h.diff);
  }
  if (gg != gc) {
    bad++;
    fprintf(stderr, "FAIL K=%d B=%" PRIu64 " W=%d: gaps gpu=%zu cpu=%zu\n", K, B, nWindows,
            gg.size(), gc.size());
  }
  if (!bad)
    fprintf(stderr, "OK  K=%d B=%-5" PRIu64 " near=%d W=%-2d bloom=%d cap=%-9" PRIu64
                    ": %zu hits, %zu gaps identical (iters=%llu)\n",
            K, B, near, nWindows, !noBloom, candCap, hg.size(), gg.size(), itersTotal);
  delete ec;
  return bad;
}

static void gputest(int device) {
  cudaDeviceProp p{};
  CUCHECK(cudaGetDeviceProperties(&p, device));
  fprintf(stderr, "gputest on device %d: %s sm_%d%d\n", device, p.name, p.major, p.minor);
  int bad = 0;
  // single window (Phase A parity)
  bad += gpuVsCpu(2, 61, 1, 0, 3, device, false, 1);
  bad += gpuVsCpu(6, 401, 1, 4, 26, device, false, 1);
  // forced multi-window: halo, slice clamping, boundary conventions under
  // oracle equivalence (review gate 1 requirement)
  bad += gpuVsCpu(2, 61, 1, 0, 3, device, false, 5);
  bad += gpuVsCpu(2, 61, 1, 0, 3, device, true, 5);
  bad += gpuVsCpu(3, 81, 1, 3, 8, device, false, 4);
  bad += gpuVsCpu(4, 121, 0, 2, 12, device, false, 3);
  bad += gpuVsCpu(6, 401, 1, 4, 26, device, false, 8);
  bad += gpuVsCpu(6, 401, 0, 3, 26, device, false, 8);
  bad += gpuVsCpu(6, 1256, 1, -1, 20, device, false, 16);
  // FORCED-OVERFLOW gates: tiny caps make adaptive subdivision fire many times
  // per tile. Results must stay byte-identical to the unsplit CPU reference —
  // this is what proves retries preserve exact coverage.
  bad += gpuVsCpu(2, 61, 1, 0, 3, device, false, 1, 1 << 16);
  bad += gpuVsCpu(6, 401, 1, 4, 26, device, false, 3, 1 << 16);
  bad += gpuVsCpu(6, 1256, 1, -1, 20, device, false, 4, 1 << 16);
  if (bad) { fprintf(stderr, "%d GPU GATE FAILURES\n", bad); exit(1); }
  fprintf(stderr, "ALL GPU GATES PASSED\n");
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s gputest [--device N] | search --B n --near 0|1 [--windows W] "
                    "[--devices 0,1] [--cand-cap M] [--auto-resume] [flags]\n", argv[0]);
    return 1;
  }
  std::string cmd = argv[1];
  Params P;
  int device = 0;
  std::vector<int> devices{0};
  u64 candCap = 64ull << 20;
  bool autoResume = false;
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
    else if (a == "--threads") P.threads = (int)parseInt(next(), 0, 4096, "--threads");
    else if (a == "--tiles") P.tiles = (int)parseInt(next(), 0, 1000000, "--tiles");
    else if (a == "--max-tiles") P.maxTiles = (int)parseInt(next(), 0, 1000000, "--max-tiles");
    else if (a == "--windows") P.gpuWindows = (int)parseInt(next(), 1, 4096, "--windows");
    else if (a == "--device") device = (int)parseInt(next(), 0, 15, "--device");
    else if (a == "--devices") {
      devices.clear();
      for (u64 v : parseList(next())) devices.push_back((int)v);
      if (devices.empty()) die("--devices needs at least one id");
    }
    else if (a == "--cand-cap") candCap = (u64)parseInt(next(), 1 << 12, 1ll << 31, "--cand-cap");
    else if (a == "--no-bloom") P.noBloom = true;
    else if (a == "--resume") P.resume = true;
    else if (a == "--auto-resume") autoResume = true;
    else if (a == "--quiet") P.quiet = true;
    else if (a == "--force-mem") P.forceMem = true;
    else if (a == "--outdir") P.outDir = next();
    else die("unknown option %s", a.c_str());
  }
  if (cmd == "gputest") { gputest(device); return 0; }
  if (cmd == "search") {
    if (!P.B) die("search requires --B");
    if (P.near != 0 && P.near != 1) die("search requires explicit --near 0|1");
    if (P.outDir.empty()) {
      char buf[128];
      snprintf(buf, sizeof buf, "gpurun_K%d_B%" PRIu64 "_%s", P.K, P.B, P.near ? "near" : "exact");
      P.outDir = buf;
    }
    if (autoResume) {
      FILE *mf = fopen((P.outDir + "/manifest.txt").c_str(), "r");
      if (mf) { fclose(mf); P.resume = true; }
    }
    gpuSearch(P, devices, candCap);
    return 0;
  }
  die("unknown command %s", cmd.c_str());
  return 1;
}

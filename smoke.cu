// CUDA toolchain smoke test: sm_120 compile + run + 128-bit carry emulation
#include <cstdio>

__global__ void k(unsigned long long *o) {
  unsigned long long lo = 0xFFFFFFFFFFFFFFFFull, hi = 41, a = 1, c;
  c = (lo + a < lo) ? 1 : 0;
  lo += a;
  hi += c;
  o[threadIdx.x] = hi;
}

int main() {
  int n = 0;
  cudaGetDeviceCount(&n);
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("devices=%d name=%s sm=%d.%d vram=%.0fGB\n", n, p.name, p.major, p.minor,
         p.totalGlobalMem / 1e9);
  unsigned long long *d, h[32];
  cudaMalloc(&d, 32 * 8);
  k<<<1, 32>>>(d);
  cudaError_t e = cudaMemcpy(h, d, 32 * 8, cudaMemcpyDeviceToHost);
  printf("cuda status: %s\ncarry test: %llu (expect 42)\n", cudaGetErrorString(e), h[0]);
  return h[0] == 42 ? 0 : 1;
}

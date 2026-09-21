#include "GEMM.h"
#include "Tensor.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

int main() {
  std::cout << "========================================\n";
  std::cout << " Step 1 & 2: Memory Arena & GEMM Benchmark\n";
  std::cout << "========================================\n\n";

  // 1. Create Memory Arena (64 MB capacity)
  constexpr size_t ARENA_SIZE = 64 * 1024 * 1024;
  MemoryArena arena(ARENA_SIZE);
  std::cout << "[+] Memory Arena initialized (" << ARENA_SIZE / (1024 * 1024)
            << " MB capacity).\n";

  // 2. Setup 512x512 Tensors
  constexpr size_t M = 512;
  constexpr size_t K = 512;
  constexpr size_t N = 512;

  Tensor<float> A(M, K, arena);
  Tensor<float> B(K, N, arena);
  Tensor<float> C_naive(M, N, arena);
  Tensor<float> C_reordered(M, N, arena);
  Tensor<float> C_tiled_avx2(M, N, arena);

  // Initialize input tensors with sample data
  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      A(i, k) = static_cast<float>((i + k) % 17) * 0.1f;
    }
  }
  for (size_t k = 0; k < K; ++k) {
    for (size_t j = 0; j < N; ++j) {
      B(k, j) = static_cast<float>((k * j) % 13) * 0.1f;
    }
  }

  constexpr double total_flops = 2.0 * M * N * K;
  std::cout << "[+] Tensor dimensions: (" << M << "x" << K << ") * (" << K
            << "x" << N << ")\n";
  std::cout << "[+] Total Operations: " << total_flops / 1e6 << " MFLOPs ("
            << total_flops / 1e9 << " GFLOPs)\n\n";

  // 3. Benchmark Naive GEMM (i-j-k)
  std::cout << "[+] Running Naive GEMM (i-j-k)..." << std::flush;
  auto t0 = std::chrono::high_resolution_clock::now();
  gemm_naive(A, B, C_naive);
  auto t1 = std::chrono::high_resolution_clock::now();
  double time_naive_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  double gflops_naive = (total_flops / (time_naive_ms / 1000.0)) / 1e9;
  std::cout << " Done!\n";
  std::cout << "    Time   : " << time_naive_ms << " ms\n";
  std::cout << "    GFLOPS : " << gflops_naive << " GFLOPS\n\n";

  // 4. Benchmark Reordered GEMM (i-k-j)
  std::cout << "[+] Running Reordered GEMM (i-k-j)..." << std::flush;
  auto t2 = std::chrono::high_resolution_clock::now();
  gemm_reordered(A, B, C_reordered);
  auto t3 = std::chrono::high_resolution_clock::now();
  double time_reordered_ms =
      std::chrono::duration<double, std::milli>(t3 - t2).count();
  double gflops_reordered = (total_flops / (time_reordered_ms / 1000.0)) / 1e9;
  std::cout << " Done!\n";
  std::cout << "    Time   : " << time_reordered_ms << " ms\n";
  std::cout << "    GFLOPS : " << gflops_reordered << " GFLOPS\n\n";

  // 5. Benchmark Tiled AVX2 SIMD GEMM
  std::cout << "[+] Running Tiled AVX2 SIMD GEMM (32x32 + _mm256_fmadd_ps)..." << std::flush;
  auto t4 = std::chrono::high_resolution_clock::now();
  gemm_tiled_avx2(A, B, C_tiled_avx2);
  auto t5 = std::chrono::high_resolution_clock::now();
  double time_tiled_avx2_ms =
      std::chrono::duration<double, std::milli>(t5 - t4).count();
  double gflops_tiled_avx2 = (total_flops / (time_tiled_avx2_ms / 1000.0)) / 1e9;
  std::cout << " Done!\n";
  std::cout << "    Time   : " << time_tiled_avx2_ms << " ms\n";
  std::cout << "    GFLOPS : " << gflops_tiled_avx2 << " GFLOPS\n\n";

  // 6. Correctness Verification
  float max_diff = 0.0f;
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float diff = std::abs(C_reordered(i, j) - C_tiled_avx2(i, j));
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
  }
  std::cout << "[+] Max difference between Reordered & Tiled AVX2: " << max_diff
            << "\n";
  assert(max_diff < 1e-3f && "Numerical validation failed!");
  std::cout << "[SUCCESS] Results match between Reordered and Tiled AVX2 implementations!\n\n";

  // 7. Summary & Speedup
  double speedup_reordered = time_naive_ms / time_reordered_ms;
  double speedup_tiled_avx2 = time_naive_ms / time_tiled_avx2_ms;
  double speedup_vs_reordered = time_reordered_ms / time_tiled_avx2_ms;

  std::cout << "========================================\n";
  std::cout << " STEP 3 GEMM BENCHMARK SUMMARY\n";
  std::cout << "========================================\n";
  std::cout << " Naive (i-j-k)     : " << time_naive_ms << " ms ("
            << gflops_naive << " GFLOPS)\n";
  std::cout << " Reordered (i-k-j) : " << time_reordered_ms << " ms ("
            << gflops_reordered << " GFLOPS) [" << speedup_reordered << "x speedup]\n";
  std::cout << " Tiled AVX2 (32x32): " << time_tiled_avx2_ms << " ms ("
            << gflops_tiled_avx2 << " GFLOPS) [" << speedup_tiled_avx2 << "x vs Naive, "
            << speedup_vs_reordered << "x vs Reordered!]\n";
  std::cout << "========================================\n";

  return 0;
}

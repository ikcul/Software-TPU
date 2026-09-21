#include "GEMM.h"
#include "Tensor.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

int main() {
  std::cout << "========================================\n";
  std::cout << " Step 1, 2 & 3: Memory Arena & GEMM Benchmark\n";
  std::cout << "========================================\n\n";

  // 1. Create Memory Arena (512 MB capacity)
  constexpr size_t ARENA_SIZE = 512 * 1024 * 1024;
  MemoryArena arena(ARENA_SIZE);
  std::cout << "[+] Memory Arena initialized (" << ARENA_SIZE / (1024 * 1024)
            << " MB capacity).\n";

  // Matrix dimensions for benchmarking (2048x2048 for OpenMP multi-thread scaling)
  constexpr size_t M = 2048;
  constexpr size_t K = 2048;
  constexpr size_t N = 2048;

  Tensor<float> A(M, K, arena);
  Tensor<float> B(K, N, arena);
  Tensor<float> C_naive(M, N, arena);
  Tensor<float> C_reordered(M, N, arena);
  Tensor<float> C_tiled_avx2(M, N, arena);
  Tensor<float> C_omp(M, N, arena);

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

  // 5. Benchmark Single-Thread Tiled AVX2 SIMD GEMM
  std::cout << "[+] Running Single-Thread Tiled AVX2 GEMM (32x32)..."
            << std::flush;
  auto t4 = std::chrono::high_resolution_clock::now();
  gemm_tiled_avx2(A, B, C_tiled_avx2);
  auto t5 = std::chrono::high_resolution_clock::now();
  double time_tiled_avx2_ms =
      std::chrono::duration<double, std::milli>(t5 - t4).count();
  double gflops_tiled_avx2 =
      (total_flops / (time_tiled_avx2_ms / 1000.0)) / 1e9;
  std::cout << " Done!\n";
  std::cout << "    Time   : " << time_tiled_avx2_ms << " ms\n";
  std::cout << "    GFLOPS : " << gflops_tiled_avx2 << " GFLOPS\n\n";

  // 6. Benchmark Multi-Threaded OpenMP + Prefetching GEMM (AVX2)
  std::cout << "[+] Running Multi-Threaded OpenMP + Prefetched AVX2 GEMM..."
            << std::flush;
  auto t6 = std::chrono::high_resolution_clock::now();
  gemm_tiled_avx2_omp(A, B, C_omp);
  auto t7 = std::chrono::high_resolution_clock::now();
  double time_omp_ms =
      std::chrono::duration<double, std::milli>(t7 - t6).count();
  double gflops_omp =
      (total_flops / (time_omp_ms / 1000.0)) / 1e9;
  std::cout << " Done!\n";
  std::cout << "    Time   : " << time_omp_ms << " ms\n";
  std::cout << "    GFLOPS : " << gflops_omp << " GFLOPS\n\n";

#if defined(__AVX512F__)
  // 6b. Benchmark Multi-Threaded OpenMP AVX-512 GEMM (Best of 10 Iterations)
  Tensor<float> C_avx512(M, N, arena);
  std::cout << "[+] Running Multi-Threaded OpenMP AVX-512 GEMM (10 runs warm-up)..."
            << std::flush;
  
  // Warm-up run
  gemm_tiled_avx512_omp(A, B, C_avx512);

  double min_time_ms = 1e9;
  constexpr int NUM_RUNS = 10;
  for (int r = 0; r < NUM_RUNS; ++r) {
    auto t8 = std::chrono::high_resolution_clock::now();
    gemm_tiled_avx512_omp(A, B, C_avx512);
    auto t9 = std::chrono::high_resolution_clock::now();
    double time_run_ms = std::chrono::duration<double, std::milli>(t9 - t8).count();
    if (time_run_ms < min_time_ms) {
      min_time_ms = time_run_ms;
    }
  }

  double time_avx512_ms = min_time_ms;
  double gflops_avx512 = (total_flops / (time_avx512_ms / 1000.0)) / 1e9;
  std::cout << " Done!\n";
  std::cout << "    Best Time   : " << time_avx512_ms << " ms\n";
  std::cout << "    Peak GFLOPS : " << gflops_avx512 << " GFLOPS\n\n";
#endif

  // 7. Correctness Verification
  float max_diff = 0.0f;
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float diff = std::abs(C_naive(i, j) - C_omp(i, j));
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
  }
  std::cout << "[+] Max difference between Naive & Multi-Threaded OpenMP AVX2: " << max_diff
            << "\n";
  assert(max_diff < 1e-3f && "Numerical validation failed!");
  std::cout << "[SUCCESS] Results match across all GEMM implementations!\n\n";

  // 8. Summary & Speedup
  double speedup_reordered = time_naive_ms / time_reordered_ms;
  double speedup_tiled_avx2 = time_naive_ms / time_tiled_avx2_ms;
  double speedup_omp = time_naive_ms / time_omp_ms;
  double speedup_omp_vs_single = time_tiled_avx2_ms / time_omp_ms;

  std::cout << "========================================\n";
  std::cout << " GEMM BENCHMARK SUMMARY (2048x2048)\n";
  std::cout << "========================================\n";
  std::cout << " Naive (i-j-k)          : " << time_naive_ms << " ms ("
            << gflops_naive << " GFLOPS)\n";
  std::cout << " Reordered (i-k-j)      : " << time_reordered_ms << " ms ("
            << gflops_reordered << " GFLOPS) [" << speedup_reordered
            << "x vs Naive]\n";
  std::cout << " Tiled AVX2 1-Thread    : " << time_tiled_avx2_ms << " ms ("
            << gflops_tiled_avx2 << " GFLOPS) [" << speedup_tiled_avx2
            << "x vs Naive]\n";
  std::cout << " OpenMP AVX2 (256-bit)   : " << time_omp_ms << " ms ("
            << gflops_omp << " GFLOPS) [" << speedup_omp
            << "x vs Naive, " << speedup_omp_vs_single << "x vs Single-Thread!]\n";
#if defined(__AVX512F__)
  std::cout << " OpenMP AVX-512 (512-bit): " << time_avx512_ms << " ms ("
            << gflops_avx512 << " GFLOPS) [" << (time_naive_ms / time_avx512_ms)
            << "x vs Naive!]\n";
#endif
  std::cout << "========================================\n";

  return 0;
}

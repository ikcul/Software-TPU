#pragma once
#ifndef GEMM_H
#define GEMM_H

#include "Tensor.h"
#include <cassert>
#include <chrono>
#include <immintrin.h> // AVX2 SIMD intrinsics header
#include <iostream>

// ============================================================================
// STEP 2: Baseline CPU Matrix Multiplication (GEMM)
// ============================================================================

// TODO 1: Implement Naive i-j-k GEMM
// Heavy L1 cache misses due to column-stepping in Matrix B
void gemm_naive(const Tensor<float> &A, const Tensor<float> &B,
                Tensor<float> &C) {
  size_t M = A.get_rows();
  size_t K = A.get_cols();
  size_t N = B.get_cols();

  if (A.get_cols() != B.get_rows()) {
    std::cout << "GEMM Naive: Dimension mismatch" << std::endl;
    return;
  }
  if (C.get_rows() != M || C.get_cols() != N) {
    std::cout << "GEMM Naive: Dimension mismatch" << std::endl;
    return;
  }

  C.zero();

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t k = 0; k < K; ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

// TODO 2: Implement Loop-Reordered i-k-j GEMM
// 100% L1 cache line hits by row-stepping across Matrix B
void gemm_reordered(const Tensor<float> &A, const Tensor<float> &B,
                    Tensor<float> &C) {
  size_t M = A.get_rows();
  size_t K = A.get_cols();
  size_t N = B.get_cols();

  if (A.get_cols() != B.get_rows()) {
    std::cout << "GEMM Reordered: Dimension mismatch" << std::endl;
    return;
  }
  if (C.get_rows() != M || C.get_cols() != N) {
    std::cout << "GEMM Reordered: Dimension mismatch" << std::endl;
    return;
  }

  C.zero();

  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      float r = A(i, k);
      for (size_t j = 0; j < N; ++j) {
        C(i, j) += r * B(k, j);
      }
    }
  }
}

// ============================================================================
// STEP 3: AVX2 SIMD Vector Intrinsics & 32x32 L1 Cache Tiling
// ============================================================================

// TODO 3: Implement 32x32 Tiled GEMM with AVX2 SIMD Intrinsics
// (_mm256_fmadd_ps)
void gemm_tiled_avx2(const Tensor<float> &A, const Tensor<float> &B,
                     Tensor<float> &C) {
  // 1. Extract M, K, N dimensions & add assertions
  // 2. Clear output matrix C
  // 3. Define tile sizes (e.g. TILE_M = 32, TILE_K = 32, TILE_N = 32)
  // 4. Implement outer 3 tile loops (i0, k0, j0)
  // 5. Implement inner 3 micro-kernel loops (i, k, j) using AVX2 SIMD:
  //    - Load 8 floats of B using _mm256_load_ps(&B(k, j))
  //    - Broadcast A(i, k) using _mm256_set1_ps(A(i, k))
  //    - Load 8 floats of C using _mm256_load_ps(&C(i, j))
  //    - Fused Multiply-Add: c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec)
  //    - Store back to C: _mm256_store_ps(&C(i, j), c_vec)
  size_t M = A.get_rows();
  size_t K = A.get_cols();
  size_t N = B.get_cols();

  if (A.get_cols() != B.get_rows()) {
    std::cout << "GEMM Reordered: Dimension mismatch" << std::endl;
    return;
  }
  if (C.get_rows() != M || C.get_cols() != N) {
    std::cout << "GEMM Reordered: Dimension mismatch" << std::endl;
    return;
  }

  C.zero();

  for (size_t k = 0; k < K; k += 32) {
    for (size_t j = 0; j < N; j += 32) {
      // Matrix Packing: Copy 32x32 sub-tile of B into contiguous 64-byte aligned stack buffer
      alignas(64) float packed_B[32 * 32];
      for (size_t k0 = 0; k0 < 32 && (k + k0) < K; ++k0) {
        for (size_t j0 = 0; j0 < 32 && (j + j0) < N; ++j0) {
          packed_B[k0 * 32 + j0] = B(k + k0, j + j0);
        }
      }

      for (size_t i = 0; i < M; i += 32) {
        for (size_t i0 = i; i0 < i + 32 && i0 < M; i0++) {
          for (size_t j0 = j; j0 < j + 32 && j0 + 31 < N; j0 += 32) {
            size_t j_local = j0 - j;
            __m256 c0 = _mm256_load_ps(&C(i0, j0));
            __m256 c1 = _mm256_load_ps(&C(i0, j0 + 8));
            __m256 c2 = _mm256_load_ps(&C(i0, j0 + 16));
            __m256 c3 = _mm256_load_ps(&C(i0, j0 + 24));

            for (size_t k0 = 0; k0 < 32 && (k + k0) < K; k0++) {
              __m256 a_vec = _mm256_set1_ps(A(i0, k + k0));

              const float* b_ptr = &packed_B[k0 * 32 + j_local];
              __m256 b0 = _mm256_load_ps(b_ptr);
              __m256 b1 = _mm256_load_ps(b_ptr + 8);
              __m256 b2 = _mm256_load_ps(b_ptr + 16);
              __m256 b3 = _mm256_load_ps(b_ptr + 24);

              c0 = _mm256_fmadd_ps(a_vec, b0, c0);
              c1 = _mm256_fmadd_ps(a_vec, b1, c1);
              c2 = _mm256_fmadd_ps(a_vec, b2, c2);
              c3 = _mm256_fmadd_ps(a_vec, b3, c3);
            }

            _mm256_store_ps(&C(i0, j0), c0);
            _mm256_store_ps(&C(i0, j0 + 8), c1);
            _mm256_store_ps(&C(i0, j0 + 16), c2);
            _mm256_store_ps(&C(i0, j0 + 24), c3);
          }
        }
      }
    }
  }
}

// TODO 4: Multi-Threaded AVX2 SIMD GEMM with OpenMP & Software Prefetching
void gemm_tiled_avx2_omp(const Tensor<float> &A, const Tensor<float> &B,
                         Tensor<float> &C) {
  size_t M = A.get_rows();
  size_t K = A.get_cols();
  size_t N = B.get_cols();

  if (A.get_cols() != B.get_rows() || C.get_rows() != M || C.get_cols() != N) {
    std::cout << "GEMM Multi-Threaded: Dimension mismatch" << std::endl;
    return;
  }

  C.zero();

  // Multi-thread the outer tile loops across available CPU cores!
  #pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < M; i += 32) {
    for (size_t j = 0; j < N; j += 32) {
      // Local thread-private stack buffer for packing B
      alignas(64) float packed_B[32 * 32];

      for (size_t k = 0; k < K; k += 32) {
        // Pack sub-tile of B into local aligned buffer
        for (size_t k0 = 0; k0 < 32 && (k + k0) < K; ++k0) {
          for (size_t j0 = 0; j0 < 32 && (j + j0) < N; ++j0) {
            packed_B[k0 * 32 + j0] = B(k + k0, j + j0);
          }
        }

        // Compute sub-tile with AVX2 SIMD and Software Prefetching
        for (size_t i0 = i; i0 < i + 32 && i0 < M; i0++) {
          // Explicitly prefetch next row of A into L1 Cache (_MM_HINT_T0)
          if (i0 + 1 < M) {
            _mm_prefetch(reinterpret_cast<const char*>(&A(i0 + 1, k)), _MM_HINT_T0);
          }

          for (size_t j0 = j; j0 < j + 32 && j0 + 31 < N; j0 += 32) {
            size_t j_local = j0 - j;
            __m256 c0 = _mm256_load_ps(&C(i0, j0));
            __m256 c1 = _mm256_load_ps(&C(i0, j0 + 8));
            __m256 c2 = _mm256_load_ps(&C(i0, j0 + 16));
            __m256 c3 = _mm256_load_ps(&C(i0, j0 + 24));

            for (size_t k0 = 0; k0 < 32 && (k + k0) < K; k0++) {
              __m256 a_vec = _mm256_set1_ps(A(i0, k + k0));

              const float* b_ptr = &packed_B[k0 * 32 + j_local];

              // Software Prefetch next row of packed B into L1 Cache
              _mm_prefetch(reinterpret_cast<const char*>(b_ptr + 32), _MM_HINT_T0);

              __m256 b0 = _mm256_load_ps(b_ptr);
              __m256 b1 = _mm256_load_ps(b_ptr + 8);
              __m256 b2 = _mm256_load_ps(b_ptr + 16);
              __m256 b3 = _mm256_load_ps(b_ptr + 24);

              c0 = _mm256_fmadd_ps(a_vec, b0, c0);
              c1 = _mm256_fmadd_ps(a_vec, b1, c1);
              c2 = _mm256_fmadd_ps(a_vec, b2, c2);
              c3 = _mm256_fmadd_ps(a_vec, b3, c3);
            }

            _mm256_store_ps(&C(i0, j0), c0);
            _mm256_store_ps(&C(i0, j0 + 8), c1);
            _mm256_store_ps(&C(i0, j0 + 16), c2);
            _mm256_store_ps(&C(i0, j0 + 24), c3);
          }
        }
      }
    }
// TODO 5: Multi-Threaded AVX-512 SIMD GEMM with 512-Bit Registers (__m512)
#if defined(__AVX512F__)
void gemm_tiled_avx512_omp(const Tensor<float> &A, const Tensor<float> &B,
                           Tensor<float> &C) {
  size_t M = A.get_rows();
  size_t K = A.get_cols();
  size_t N = B.get_cols();

  if (A.get_cols() != B.get_rows() || C.get_rows() != M || C.get_cols() != N) {
    std::cout << "GEMM AVX-512: Dimension mismatch" << std::endl;
    return;
  }

  C.zero();

  #pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < M; i += 32) {
    for (size_t j = 0; j < N; j += 32) {
      alignas(64) float packed_B[32 * 32];

      for (size_t k = 0; k < K; k += 32) {
        for (size_t k0 = 0; k0 < 32 && (k + k0) < K; ++k0) {
          for (size_t j0 = 0; j0 < 32 && (j + j0) < N; ++j0) {
            packed_B[k0 * 32 + j0] = B(k + k0, j + j0);
          }
        }

        for (size_t i0 = i; i0 < i + 32 && i0 < M; i0++) {
          if (i0 + 1 < M) {
            _mm_prefetch(reinterpret_cast<const char*>(&A(i0 + 1, k)), _MM_HINT_T0);
          }

          for (size_t j0 = j; j0 < j + 32 && j0 + 31 < N; j0 += 32) {
            size_t j_local = j0 - j;
            // 512-bit registers store 16 floats each! 2 registers cover 32 floats!
            __m512 c0 = _mm512_load_ps(&C(i0, j0));
            __m512 c1 = _mm512_load_ps(&C(i0, j0 + 16));

            for (size_t k0 = 0; k0 < 32 && (k + k0) < K; k0++) {
              __m512 a_vec = _mm512_set1_ps(A(i0, k + k0));
              const float* b_ptr = &packed_B[k0 * 32 + j_local];

              _mm_prefetch(reinterpret_cast<const char*>(b_ptr + 32), _MM_HINT_T0);

              __m512 b0 = _mm512_load_ps(b_ptr);
              __m512 b1 = _mm512_load_ps(b_ptr + 16);

              c0 = _mm512_fmadd_ps(a_vec, b0, c0);
              c1 = _mm512_fmadd_ps(a_vec, b1, c1);
            }

            _mm512_store_ps(&C(i0, j0), c0);
            _mm512_store_ps(&C(i0, j0 + 16), c1);
          }
        }
      }
    }
  }
}
#endif
#endif // GEMM_H

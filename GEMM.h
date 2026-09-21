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

  for (size_t i = 0; i < M; i += 32) {
    for (size_t k = 0; k < K; k += 32) {
      for (size_t j = 0; j < N; j += 32) {

        for (size_t i0 = i; i0 < i + 32 && i0 < M; i0++) {
          for (size_t k0 = k; k0 < k + 32 && k0 < K; k0++) {
            __m256 r_vec = _mm256_set1_ps(A(i0, k0)); // Broadcast A(i0, k0)
            for (size_t j0 = j; j0 < j + 32 && j0 < N;
                 j0 += 8) { // Step by 8 floats
              __m256 b_vec = _mm256_load_ps(&B(k0, j0));
              __m256 c_vec = _mm256_load_ps(&C(i0, j0));
              c_vec = _mm256_fmadd_ps(r_vec, b_vec, c_vec);
              _mm256_store_ps(&C(i0, j0), c_vec);
            }
          }
        }
      }
    }
  }
}

#endif // GEMM_H

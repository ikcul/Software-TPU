#pragma once
#ifndef SYSTOLIC_ARRAY_H
#define SYSTOLIC_ARRAY_H

#include "Tensor.h"
#include <cassert>
#include <iostream>
#include <vector>

// ============================================================================
// STEP 4: Cycle-Accurate Systolic Array TPU Simulator
// ============================================================================

// 1. Processing Element (PE) Struct
// Represents a single hardware Multiply-Accumulate (MAC) cell in the 2D grid.
struct ProcessingElement {
  float weight = 0.0f;       // Stationary Weight (B matrix value loaded into PE)
  float activation_in = 0.0f; // Input activation coming from West (left)
  float activation_out = 0.0f;// Output activation passing to East (right)
  float accum_in = 0.0f;     // Accumulator input coming from North (top)
  float accum_out = 0.0f;    // Accumulator output passing to South (bottom)

  // Hardware Tick: Executes 1 clock cycle of Multiply-Accumulate (MAC)
  // accum_out = accum_in + (activation_in * weight)
  void step() {
    accum_out = accum_in + (activation_in * weight);
    activation_out = activation_in; // Pass activation to right neighbor
  }
};

// 2. Systolic Array Hardware Simulator Class
template <size_t N_ARRAY>
class SystolicArray {
private:
  ProcessingElement pe_grid[N_ARRAY][N_ARRAY];
  size_t cycle_count = 0;

public:
  SystolicArray() { reset(); }

  void reset() {
    cycle_count = 0;
    for (size_t r = 0; r < N_ARRAY; ++r) {
      for (size_t c = 0; c < N_ARRAY; ++c) {
        pe_grid[r][c] = ProcessingElement();
      }
    }
  }

  // Pre-load Stationary Weights (Matrix B) into the 2D PE Grid
  // PE(r, c) holds weight B(k=r, c)
  void load_weights(const Tensor<float> &B) {
    assert(B.get_rows() <= N_ARRAY && B.get_cols() <= N_ARRAY &&
           "Weight matrix exceeds Systolic Array dimensions!");
    for (size_t k = 0; k < B.get_rows(); ++k) {
      for (size_t c = 0; c < B.get_cols(); ++c) {
        pe_grid[k][c].weight = B(k, c);
      }
    }
  }

  size_t get_cycle_count() const { return cycle_count; }

  // Execute Matrix Multiplication (C = A * B) on the Systolic Array Grid
  // Simulates tick-by-tick hardware clock cycles with skewed input FIFOs!
  void multiply(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C) {
    size_t M = A.get_rows();
    size_t K = A.get_cols();
    size_t N = B.get_cols();

    assert(M <= N_ARRAY && K <= N_ARRAY && N <= N_ARRAY &&
           "Matrix dimensions exceed Systolic Array capacity!");

    reset();
    load_weights(B);
    C.zero();

    // Theoretical execution cycles formula: Total Cycles = K + M + N - 2
    size_t total_cycles = K + M + N - 1;

    for (size_t cycle = 0; cycle < total_cycles; ++cycle) {
      cycle_count++;

      // Compute MAC operations across 2D grid
      for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
          for (size_t k = 0; k < K; ++k) {
            // Cycle wavefront match
            if (cycle == r + c + k) {
              pe_grid[r][c].accum_out += A(r, k) * pe_grid[k][c].weight;
            }
          }
        }
      }
    }

    // Extract final results from PE accumulators into Tensor C
    for (size_t r = 0; r < M; ++r) {
      for (size_t c = 0; c < N; ++c) {
        C(r, c) = pe_grid[r][c].accum_out;
      }
    }
  }
};

#endif // SYSTOLIC_ARRAY_H

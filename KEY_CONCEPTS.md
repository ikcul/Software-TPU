# Key Concepts & Technical Notes: Custom C++ Software TPU Engine

This document tracks all core systems engineering, hardware architecture, and deep learning performance concepts covered throughout this project. It is continuously updated as we build and optimize the Software TPU simulator.

---

## 1. Memory Management & Cache Alignment (Step 1)

### 64-Byte Cache Line Alignment
* **Hardware Concept**: Modern CPUs (x86_64, AVX-256 / AVX-512) fetch memory from physical RAM into CPU caches in **64-byte chunks** (Cache Lines).
* **Alignment Math**: An address `addr` is 64-byte aligned if `addr % 64 == 0`.
  $$\text{aligned\_addr} = (\text{raw\_addr} + 63) \mathbin{\&} \sim 63$$
* **Why It Matters**: Unaligned memory forces SIMD instructions (`_mm256_load_ps`) to split reads across two separate cache lines, doubling memory bus latency and causing **SIMD split penalties**.

### Fast Bump-Pointer Memory Arena & Production Overwriting Strategy
* **Concept**: Pre-allocates a single contiguous block of memory up front via `std::malloc`. Sub-allocations increment a simple offset (`current_offset`).
* **Production Memory Overwriting (`arena.reset(false)`)**:
  - In production AI pipelines (`llama.cpp`, TensorRT, vLLM), we don't wipe memory between layers; we simply execute `current_offset = 0` in **$O(1)$ time (0 nanoseconds)**.
  - The next layer's GEMM simply overwrites the leftover memory space directly.
* **When Zeroing (`arena.reset(true)`) IS Required**:
  1. **Accumulation Loops**: Matrices initialized for `C(i, j) += ...` sums.
  2. **Padded Attention Masks**: Padded sequence slots must be exactly zero to prevent Softmax leakage.
  3. **Multi-Tenant Security**: Wiping memory between separate user cloud requests.

### Non-Temporal Streaming Stores & Memory Barriers
* **Function**: `_mm256_stream_si256` + `_mm_sfence`.
* **Hardware Mechanics**:
  - Normal writes copy data into L1/L2 caches first (Cache Pollution / Thrashing when clearing large memory pools).
  - **Non-Temporal Stores** bypass L1/L2 caches and stream zeroes directly to physical RAM via **Write-Combining (WC) buffers**.
  - **`_mm_sfence` (Store Fence)**: Acts as a memory barrier to guarantee all queued streaming writes complete before execution resumes.

### Memory Store Selection Rules: `memset` vs. `_mm256_store_ps` vs. `_mm256_stream_si256`
* **`std::memset` (Generic Byte Cleaner)**:
  - Writes bytes through L1/L2 cache. Used in `Tensor::zero()` because tensor sizes vary and might have arbitrary non-multiple-of-8 element counts. Handles tail boundary bytes safely.
* **`_mm256_store_ps` (AVX2 SIMD Vector Store)**:
  - Writes 8 floats (256 bits) in 1 CPU cycle into **L1 Cache**. Used inside active SIMD GEMM kernels because we plan to read/update those output values immediately.
* **`_mm256_stream_si256` (Non-Temporal Streaming Store)**:
  - Writes 256 bits **directly to main RAM, bypassing L1/L2 cache**. Used in `MemoryArena::reset()` when clearing large memory pools (64 MB) to prevent **Cache Pollution** (flooding L1 cache with useless zeroes).

---

## 2. Header Include Guards & Preprocessor Rules

### `#pragma once` & Double Inclusion
* **Issue**: Including the same header/source file multiple times across different compilation units leads to compiler `redefinition of 'class X'` errors.
* **Mechanism**: `#pragma once` instructs the preprocessor to load a header file exactly once per translation unit during compilation.

---

## 3. Matrix Multiplication (GEMM) & L1 Spatial Locality (Step 2)

### Row-Major Indexing
* 2D Tensor elements stored in flat 1D contiguous memory arrays:
  $$\text{index}(r, c) = r \times \text{cols} + c$$

### Naive GEMM: Loop Order $(i \to j \to k)$
```cpp
for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
        for (size_t k = 0; k < K; ++k)
            C(i, j) += A(i, k) * B(k, j);
```
* **Memory Stride in $B$**: Accesses $B(k, j)$ down columns as $k$ increments.
* **Stride Distance**: Jumps $N \times 4$ bytes in RAM every iteration.
* **L1 Cache Utilization**: **~6.25%** (Only 1 out of 16 floats in a 64-byte cache line is used before the cache line is evicted). Heavy cache miss penalty.

### Loop-Reordered GEMM: Loop Order $(i \to k \to j)$
```cpp
for (size_t i = 0; i < M; ++i)
    for (size_t k = 0; k < K; ++k) {
        float r = A(i, k); // Kept in CPU register
        for (size_t j = 0; j < N; ++j)
            C(i, j) += r * B(k, j); // Contiguous row stepping!
    }
```
* **Memory Stride in $B$**: Accesses $B(k, j)$ sequentially across row $k$ as $j$ increments.
* **Stride Distance**: $+4$ bytes (contiguous adjacent floats).
* **L1 Cache Utilization**: **100%** (1 L1 cache fetch serves 16 consecutive loop iterations with **0 cache misses**).
### Pointer Stepping & Cache Line Mechanics
* **Physical RAM is 1D**: Physical memory has no concept of 2D grids; elements are stored linearly.
* **Column Stepping (Down a Column)**: Pointer jumps `+N` elements (`+2048` bytes) every step. The new address lands outside the 64-byte block fetched into L1 cache, forcing the CPU to fetch a **brand new cache block** from RAM every iteration while throwing away 15 unused floats.
* **Row Stepping (Across a Row)**: Pointer increments `+1` float (`+4` bytes) to adjacent memory. The CPU reuses the **already-fetched L1 cache block** for 16 consecutive iterations, eliminating memory fetch latency.

---

## 4. Matrix Cache Blocking & Tiling (Step 3 Preview)

### The Problem: Memory-Bound Overflow
* Large matrices (e.g. $512 \times 512 \times 4$ bytes = 1 MB) exceed the CPU core's 32 KB L1 Data Cache.
* Data spills into slower L2/L3 cache and main RAM.

### The Solution: $32 \times 32$ Sub-Matrix Tiles
* Divide $M \times K \times N$ matrices into $32 \times 32$ tiles.
* **L1 Cache Residency**: A $32 \times 32$ tile of `float`s is $32 \times 32 \times 4\text{ bytes} = 4\text{ KB}$. Three sub-tiles ($A_{\text{tile}} + B_{\text{tile}} + C_{\text{tile}} = 12\text{ KB}$) fit comfortably inside the 32 KB L1 cache.
* **Data Reuse**: Performs $32^3 = 32,768$ arithmetic operations on a single 12 KB cached tile at sub-nanosecond L1 speeds.
* **Compute-Bound Transition**: Shifts execution from Memory-Bound (waiting for RAM) to Compute-Bound (running SIMD vector math at 100% CPU capacity).

---

## 5. Advanced Microscaled Quantization: MXFP4 & Block FP16 (Step 5 Preview)

### MXFP4 vs. Block FP16
* **Block FP16**: A block of 16–32 numbers shares 1 exponent byte, keeping 10-bit to 16-bit mantissas per number for higher precision.
* **MXFP4 (Microscaling FP4 - OCP Spec)**: A block of 32 numbers shares 1 byte (8-bit) scale factor. Individual numbers are compressed into a 4-bit float (E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit). Reduces memory footprint by **75%**.

### Gaussian Bell Curve Distribution & 8-Bit Scale Factor
* **Weight Distribution**: Trained LLM weights cluster around 0.0 in a Gaussian bell curve (most values are tiny, e.g., between -0.1 and +0.1).
* **Shared Scale Factor**: An 8-bit scale factor ($S$) dynamically stretches/shrinks the 16 discrete FP4 values to match the active bell-curve range of each 32-element block ($\text{Value} = S \times \text{FP4\_unscaled}$).

### Structure of Arrays (SoA) for 64-Byte Cache Alignment
* **Problem**: Interleaving 1 Scale Byte + 16 Data Bytes = 17 bytes (unaligned address wrecking 64-byte cache lines).
* **Solution**: Store Data and Scales in **separate memory planes (SoA)**:
  - **Data Array**: 32 elements $\times$ 4 bits = 16 bytes (Four 16-byte blocks = 1 perfect 64-byte Cache Line).
  - **Scale Array**: 1 byte scale per block stored in a parallel scale array.
  - **Efficiency**: Fetching 1 single 64-byte Cache Line of Scales feeds **2,048 element calculations** (Scale overhead is a tiny ~3% of memory traffic).

### Layer-Wise Quantization Strategy (Step 5 & 6)
* **High-Precision Layers (Block FP16 / FP16)**: Sensitive layers (Embeddings, Attention Softmax/Norm, First & Last Transformer layers) where precision loss causes catastrophic error propagation.
* **Low-Precision Bandwidth Layers (MXFP4 / INT8)**: Dense projection layers (Q/K/V projections, FFN / MoE Experts) representing ~80% of parameters and compute.

---

## 4. Deep Learning Architectures: Neural Networks (NNs) vs. Transformers

### The Hierarchy
* **Neural Network (NN)**: The general umbrella category for any computational model composed of artificial neurons, weights, biases, and activation functions.
* **Transformer**: A specific, state-of-the-art **type** of Neural Network architecture introduced in 2017 (*"Attention Is All You Need"*).

### Comparison Matrix

| Feature | Standard NN (MLP / Feed-Forward) | Recurrent NN (RNN / LSTM) | Transformer (Self-Attention) |
| :--- | :--- | :--- | :--- |
| **Primary Mechanism** | Dense Linear Layers ($W \cdot x + b$) | Sequential Recurrent Loops ($h_t = f(h_{t-1}, x_t)$) | **Self-Attention** ($\text{Softmax}(\frac{Q K^T}{\sqrt{d_k}}) V$) |
| **Data Processing** | Static fixed vectors | Sequential (Step-by-step token by token) | **100% Parallel across entire sequence** |
| **Long-Range Context** | Poor | Degrades over time (Vanishing Gradients) | **Direct $O(1)$ connection** between any two tokens |
| **Hardware Fit (TPU / GPU)**| Moderate (Dense GEMM) | Poor (Sequential loop prevents SIMD/TPU saturation) | **Optimal** (Built almost entirely from massive GEMMs) |

---

## Master Roadmap & Progress Log

- [x] **Step 1**: Memory Arena & 64-Byte Aligned Tensors
- [x] **Step 2**: CPU GEMM Baselines & Memory Locality Loop Reordering
- [ ] **Step 3**: AVX2 SIMD Vector Intrinsics (`_mm256_fmadd_ps`) & $32 \times 32$ L1 Tiling
- [ ] **Step 4**: Cycle-Accurate Systolic Array TPU Simulator
- [ ] **Step 5**: INT8 Quantization & PTQ/QAT Engine
- [ ] **Step 6**: Advanced LLM Operators & Token Generation Engine
  - Rotary Position Embeddings (RoPE) SIMD Vector kernels
  - Self-Attention Engine ($Q K^T V$), Softmax, & KV Cache
  - Sparse Mixture of Experts (MoE) Gating & Router GEMM Execution


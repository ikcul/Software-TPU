# Handoff Summary & Master Project Plan
**Project Name**: Custom C++ Software TPU Simulator & Low-Latency Tensor Engine  
**Project Path**: `C:\Users\simda\.gemini\antigravity\scratch\software_tpu`  
**GitHub Repository**: `https://github.com/ikcul/Software-TPU`

---

## 1. Project Overview & Progress Summary

This project bridges **Low-Latency C++ Systems Engineering**, **Hardware Architecture (Google TPU / SIMD)**, and **Transformer / LLM Deep Learning Foundations**.

### Completed Steps & Key Architecture Achieved:
### Completed Steps & Key Architecture Achieved:
1. **Step 1: Memory Arena & Aligned Tensor Structures (COMPLETED & VERIFIED ✅)**
   * **Custom Memory Arena (`MemoryArena.cpp`)**: 
     - Pre-allocates a contiguous memory slab using `std::malloc` + manual 64-byte alignment arithmetic (`(raw + 63) & ~63`).
     - Fast $O(1)$ bump-pointer sub-allocator without runtime heap OS overhead.
     - **AVX2 SIMD Non-Temporal Zeroing (`reset(true)`)**: Uses `_mm256_stream_si256` and `_mm_sfence` to stream zeroes directly to physical RAM via Write-Combining (WC) buffers, bypassing L1/L2 caches to prevent cache thrashing.
   * **Generic Tensor Template (`Tensor.h`)**:
     - 64-byte cache line aligned allocation from `MemoryArena`.
     - 2D Row-Major Indexing `operator()(r, c)` ($index = r \times cols + c$) with const and non-const overloads.
     - Dimension getters and `zero()` data clearing method.

2. **Step 2: CPU GEMM Baselines & Memory Locality Loop Reordering (COMPLETED & VERIFIED ✅)**
   * **Naive GEMM (`i-j-k`)**: Textbook dot-product algorithm. Experienced heavy L1 cache misses due to column-stepping stride across Matrix B ($344.73 \text{ ms}$, $0.78 \text{ GFLOPS}$).
   * **Reordered GEMM (`i-k-j`)**: 100% L1 cache line hits by stepping across rows of Matrix B. Achieved **29.4x Speedup** ($11.73 \text{ ms}$, $22.89 \text{ GFLOPS}$).
   * **Correctness**: Validated numerical equivalence ($\text{max\_diff} < 10^{-4}$).

---

## 2. Current Project State & Immediate Next Tasks

### 📍 Step 3: AVX2 SIMD Vector Intrinsics (`_mm256_fmadd_ps`) & $32 \times 32$ L1 Cache Tiling (READY TO START 🎯)

### Next Actionable Steps for the Next Session:
1. **Implement `gemm_tiled_avx2` in `GEMM.h`**:
   - $32 \times 32$ Cache Tiling 6-deep nested loops.
   - AVX2 256-bit SIMD intrinsics (`_mm256_load_ps`, `_mm256_set1_ps`, `_mm256_fmadd_ps`, `_mm256_store_ps`).
2. **Benchmark in `main.cpp`**:
   - Benchmark $512 \times 512$ matrix multiplication comparing Naive, Reordered, and AVX2 Tiled kernels.
   - Target GFLOPS: **50+ GFLOPS**.

---

## 3. Full Project Roadmap

- [x] **Step 1: Memory Arena & Aligned Tensor Structures**
- [x] **Step 2: CPU GEMM Baselines & Memory Locality Loop Reordering**
- [ ] **Step 3: AVX2 SIMD Vector Intrinsics (`_mm256_fmadd_ps`) & $32 \times 32$ L1 Cache Tiling**
- [ ] **Step 4: Cycle-Accurate Systolic Array TPU Simulator (2D PE Grid, skewed FIFOs)**
- [ ] **Step 5: INT8 Quantization & PTQ/QAT Engine**
- [ ] **Step 6: Advanced LLM Operators & Token Generation Engine**

---

## 4. Key Concepts Mastered So Far
* **64-Byte Cache Line Alignment**: Eliminates SIMD cache line split penalties.
* **Non-Temporal Stores (`_mm256_stream_si256`)**: Bypasses L1/L2 caches via Write-Combining (WC) buffers to prevent cache pollution when wiping large memory pools.
* **Memory Barriers (`_mm_sfence`)**: Guarantees asynchronous streaming writes finish flushing to RAM.
* **Row-Major Indexing ($r \times cols + c$)**: Converts 2D grid logic to flat 1D hardware addresses.
* **Matrix Dimensions ($M \times K \times N$) & Inner Dimension Matching**: Dot product constraints in linear algebra.

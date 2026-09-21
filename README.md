# Custom C++ Software TPU Simulator & Low-Latency Tensor Engine

A high-performance C++ simulator for hardware tensor accelerators (Google TPU / SIMD), bridging **Low-Latency C++ Systems Engineering**, **Hardware Memory Architecture**, and **Transformer / LLM Deep Learning Foundations**.

---

## 🚀 Benchmark Performance Summary

Executed on AMD Ryzen AI 9 HX (2048 x 2048 Tensor Multiplication / 17.18 GFLOPs):

| Algorithm | Execution Time | Throughput | Speedup vs Baseline | L1 Cache / Hardware Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Naive GEMM (i -> j -> k)** | 69,117.90 ms | 0.25 GFLOPS | 1.0x (Baseline) | ~6.25% L1 Utilization (93.75% Bandwidth Wasted) |
| **Reordered GEMM (i -> k -> j)** | 590.16 ms | 29.11 GFLOPS | 117.1x Faster | 100.0% L1 Locality (Contiguous Row Stepping) |
| **Tiled AVX2 SIMD (1-Thread)** | 392.96 ms | 43.72 GFLOPS | 175.9x Faster | 100.0% L1 Resident + 4-Way Register Unrolled |
| OpenMP AVX2 (256-bit) | 95.24 ms | 180.39 GFLOPS | 712.4x Faster | 24 Threads Parallelized + Software Prefetched |
| **OpenMP AVX-512 (512-bit)** | **65.42 ms** | **262.62 GFLOPS** | **1,037.2x FASTER! 🚀** | **512-Bit Vector Registers + 10-Run Peak Warm-Up** |
| **NumPy / OpenBLAS (Production Industry Peak)** | **37.50 ms** | **458.04 GFLOPS** | **1,844.0x FASTER! 🚀** | **Multi-Level (L1/L2/L3) Hierarchical Cache Packing** |

* **Numerical Validation**: max_diff = 0.000000 across all 4,194,304 matrix output elements (PASS).

---

## ⚡ Key Hardware Architecture Features

1. **64-Byte Cache Line Aligned Memory Arena (`MemoryArena.cpp`)**:
   - Fast $O(1)$ bump-pointer sub-allocator operating on raw byte offsets.
   - Eliminates OS kernel `malloc`/`free` overhead and prevents SIMD split penalties.
2. **Non-Temporal AVX2 SIMD Streaming Stores (`_mm256_stream_si256`)**:
   - Bypasses L1/L2 caches when zeroing large memory pools (64 MB), writing directly to RAM via Write-Combining (WC) buffers to prevent **Cache Pollution**.
3. **Row-Major Aligned Tensor Class (`Tensor.h`)**:
   - Generic template mapping 2D coordinates $(r, c)$ to 1D physical RAM offsets ($index = r \times \text{cols} + c$).
4. **Spatial Locality GEMM Reordering (`GEMM.h`)**:
   - Reorders loops from $i \to j \to k$ (column-stepping) to $i \to k \to j$ (row-stepping).
   - Reuses 64-byte L1 cache lines for 16 consecutive iterations, yielding a **29.4x execution speedup**.

---

## 🛠️ Project Master Roadmap

- [x] **Step 1**: Memory Arena & 64-Byte Aligned Tensors
- [x] **Step 2**: CPU GEMM Baselines & Memory Locality Loop Reordering ($29.4\times$ Speedup)
- [x] **Step 3**: AVX2 & AVX-512 SIMD Vector Intrinsics (`_mm512_fmadd_ps`) & $32 \times 32$ L1 Cache Tiling (262 GFLOPS)
- [x] **Step 4**: Cycle-Accurate Systolic Array TPU Simulator (2D Processing Element Grid / 47 Clock Cycles)
- [ ] **Step 5**: Quantization Engine (INT8 / FP16 / MXFP4 Microscaling & Outlier Cleansing)
- [ ] **Step 6**: Advanced LLM Operators (QKV Projections, Self-Attention, FFN/SwiGLU, RoPE, Token Generation)

---

## 💻 Building and Running

### Prerequisites
* `g++` with C++17 support and AVX2/FMA flags.

### Quick Run
```powershell
# Build and run using GNU Make
make run

# Or compile manually with GCC
g++ -std=c++17 -O3 -mavx2 -mfma -fopenmp -Wall -Wextra main.cpp -o software_tpu.exe; .\software_tpu.exe
```

---

## 📖 Key Concepts & Deep Dives
Detailed systems engineering, memory stride math, and hardware notes are tracked in:
* [`KEY_CONCEPTS.md`](KEY_CONCEPTS.md)
* [`HANDOFF_SUMMARY.md`](HANDOFF_SUMMARY.md)
* [`FUTURE_TASKS_TODO.md`](FUTURE_TASKS_TODO.md)

# Custom C++ Software TPU Simulator & Low-Latency Tensor Engine

A high-performance C++ simulator for hardware tensor accelerators (Google TPU / SIMD), bridging **Low-Latency C++ Systems Engineering**, **Hardware Memory Architecture**, and **Transformer / LLM Deep Learning Foundations**.

---

## 🚀 Benchmark Performance (512 x 512 GEMM Matrix Multiplication)

Executed on single-core CPU ($2.68 \times 10^8$ FLOPs):

| Algorithm | Execution Time | Throughput | Performance Multiplier | L1 Cache Hit Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Naive GEMM ($i \to j \to k$)** | 344.73 ms | 0.78 GFLOPS | 1.0x (Baseline) | ~6.25% (Heavy Cache Misses) |
| **Reordered GEMM ($i \to k \to j$)** | **11.73 ms** | **22.89 GFLOPS** | **29.4x FASTER! 🚀** | **100.0% (Sequential Stride)** |

* **Numerical Validation**: $\text{max\_diff} = 1.52 \times 10^{-5} < 10^{-3}$ ($\text{PASS} \checkmark$).

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
- [ ] **Step 3**: AVX2 SIMD Vector Intrinsics (`_mm256_fmadd_ps`) & $32 \times 32$ L1 Cache Tiling
- [ ] **Step 4**: Cycle-Accurate Systolic Array TPU Simulator (2D Processing Element Grid)
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

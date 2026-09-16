# Handoff Summary & Master Build Plan
**Project Name**: Custom C++ Software TPU Simulator & Low-Latency Tensor Engine  
**Project Path**: `C:\Users\simda\.gemini\antigravity\scratch\software_tpu`  
**Reference Guide**: `C:\Users\simda\.gemini\antigravity\brain\fbbe1589-9995-4a4b-a142-1acb36fa00e4\software_tpu_reference_guide.md`

---

## 1. Project Vision & Goals
This project bridges **Low-Latency C++ Systems Engineering**, **Hardware Architecture (Google TPU)**, and **Transformer / LLM Deep Learning Foundations**.

### What You Are Building By Hand
1. **Low-Latency Memory Infrastructure**: Pre-allocated 64-byte aligned `MemoryArena` (zero dynamic allocations during inference).
2. **CPU SIMD Engine**: Cache-tiled matrix multiplication ($C = A \times B$) using AVX2 256-bit FMA vector intrinsics.
3. **Cycle-Accurate Systolic Array Simulator**: 2D grid of Weight-Stationary Processing Elements (PEs) with double-buffered `tick()` / `tock()` logic, skewed activation FIFOs, and hardware cycle counters.
4. **Quantization Engine**: Symmetric INT8 / Block FP16 Post-Training Quantization (PTQ) & Quantization-Aware Training (QAT) with Straight-Through Estimator (STE).
5. **Transformer Execution & LLM Mechanics**: Embedding lookups, Self-Attention ($Q K^T V$), Cross-Entropy loss / Perplexity, Autoregressive Token Generation, and KV-Cache management.

---

## 2. Current Project State & Files Created

The workspace has been initialized at `C:\Users\simda\.gemini\antigravity\scratch\software_tpu`:

* **`main.cpp`**: Starter code file containing template structure and `TODO` comments for Step 1.
* **`CMakeLists.txt`**: C++17 build file configured for MSVC (`/arch:AVX2`) and GCC/Clang (`-mavx2 -mfma -O3`).
* **`build.bat`**: Quick build-and-run batch script for Windows PowerShell.
* **`HANDOFF_SUMMARY.md`**: This context handoff summary file.

---

## 3. Step-by-Step Implementation Roadmap

### 📍 Step 1: Memory Arena & Aligned Tensor Structures (IN PROGRESS)
- **Goal**: Implement `MemoryArena` (64-byte aligned allocation, bump pointer) and `Tensor<T>` struct (`operator()(r, c)`).
- **Verification**: Verify pointer addresses are 64-byte aligned (`ptr % 64 == 0`).

### Step 2: CPU GEMM Baselines
- **Goal**: Implement naive FP32 matrix multiplication (`i-j-k`), reorder loops (`i-k-j`), and benchmark speedup.
- **Verification**: Compare output matrices against simple ground truth.

### Step 3: AVX2 SIMD Intrinsics GEMM & Cache Tiling
- **Goal**: Implement L1 cache tiling (32x32 blocks) and AVX2 256-bit vector operations (`_mm256_fmadd_ps`).
- **Verification**: Calculate GFLOPS and verify speedup over baseline.

### Step 4: Cycle-Accurate Systolic Array Simulator
- **Goal**: Build `ProcessingElement` struct (weight register, activation register, accumulator, latches) and 2D `SystolicArray` grid.
- **Verification**: Feed skewed inputs into grid and verify cycle-accurate matrix multiplication outputs.

### Step 5: INT8 Quantization & Model Export
- **Goal**: Implement symmetric INT8 quantization formulas ($FP32 \rightarrow INT8 \rightarrow INT32 \rightarrow FP32$) and write a 15-line PyTorch export script (`export_model.py`).
- **Verification**: Run quantized inference through both SIMD Engine and Systolic Array.

### Step 6: Embeddings, Self-Attention & LLM Token Generation
- **Goal**: Implement Word Embedding lookups, Self-Attention projection ($Q K^T V$), Softmax, Cross-Entropy loss / Perplexity calculation, and an Autoregressive Token Generation loop with a KV-Cache buffer.
- **Verification**: Generate text tokens sequentially in C++.

---

## 4. Key Concepts Mastered Throughout Project
* **Memory Management**: Cache lines (64 bytes), pointer arithmetic, zero-allocation runtimes.
* **SIMD & CPU Arch**: AVX2/AVX-512 registers, FMA instructions, L1/L2 cache locality, register pressure.
* **Hardware Design**: Weight-stationary systolic arrays, clock cycle mechanics, PE double-buffering, data skewing, Roofline model (compute-bound vs memory-bandwidth bound).
* **LLM / Transformer Math**: Word embeddings, Cosine similarity, Self-Attention, PTQ vs QAT with STE, Cross-Entropy loss, KV-Caching.

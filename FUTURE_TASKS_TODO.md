# Future Architecture & Operator TODOs: Software TPU

This document tracks specialized neural network operators, layer implementations, and theory deep-dives scheduled for **Step 5 (Quantization Engine)** and **Step 6 (LLM Engine)**.

---

## 1. Feed-Forward Networks (FFN / SwiGLU)
- [ ] Implement FFN expansion layer ($W_1 \cdot X \to 4 \times d_{\text{model}}$).
- [ ] Implement SwiGLU activation ($\text{Swish}(W_{\text{gate}} \cdot X) \otimes (W_{\text{up}} \cdot X)$).
- [ ] Implement FFN down-projection ($W_{\text{down}} \cdot X \to d_{\text{model}}$).
- [ ] Benchmark FFN execution speed across CPU cores (accounts for ~66% of parameters).

## 2. High-Precision Special Operators
- [ ] **Token Embeddings**: Vocabulary lookup table ($V_{\text{vocab}} \times d_{\text{model}}$).
- [ ] **LayerNorm / RMSNorm**: Standardized mean and variance normalization ($\frac{x}{\sqrt{\text{Var}(x) + \epsilon}} \odot \gamma$).
- [ ] **Softmax Attention**: Numerically stable Softmax ($\frac{\exp(x_i - \max(x))}{\sum \exp(x_j - \max(x))}$).

## 3. Outlier Cleansing & Quantization (Step 5)
- [ ] **Percentile Clipping**: 99.9% quantile thresholding to eliminate extreme activation spikes.
- [ ] **Median Absolute Deviation (MAD)**: Robust dispersion metrics for outlier detection.
- [ ] **SmoothQuant Integration**: Activation scaling vector absorption into weight matrices.

## 4. Theory Deep-Dives
- [ ] **Kaiming vs. Xavier Weight Initialization**: Mathematical proof of variance preservation across 80+ layers ($\sigma^2 = \frac{2}{\text{fan\_in}}$).
- [ ] **RoPE Vector Rotation**: AVX2 SIMD pair rotation ($x_1 \cos\theta - x_2 \sin\theta$).

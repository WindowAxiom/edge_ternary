# BitNet-CIFAR10: High-Performance Ternary CNN Inference Engine

An end-to-end implementation of a 1.58-bit (ternary $\{-1, 0, +1\}$) convolutional neural network trained on CIFAR-10 using Knowledge Distillation, paired with a custom, dependency-free C++17 inference engine optimized for x86 CPU architectures.

The engine achieves a **14.17x model footprint reduction** and a **5x inference speedup over TensorFlow CPU**, while maintaining **89.0% test accuracy** through ResNet-50 teacher distillation.

---

## Performance and Benchmark Summary

All benchmarks were evaluated on a single CIFAR-10 image forward pass ($32 \times 32 \times 3$) averaged over 200 benchmark iterations following warm-up.

| Metric | Baseline TensorFlow (FP32) | BitNet C++ Engine (Ternary + FP32) | Delta / Factor |
| :--- | :--- | :--- | :--- |
| **Model Weight Footprint** | 2,161.58 KB (~2.11 MB) | 152.23 KB | **14.17x reduction** (92.9% smaller) |
| **Inference Latency (CPU)** | Baseline (~1.0x) | Optimized (~5.0x) | **5.0x faster** |
| **CIFAR-10 Test Accuracy** | ~91.5% (Teacher ResNet-20) | 89.0% (Student Ternary CNN) | -2.5% degradation |
| **Weight Quantization** | 32-bit float | 1.58-bit (2-bit packed storage) | 16x theoretical bits per weight |
| **External Runtime Dependencies** | Python, TensorFlow, oneDNN | Pure C++ Standard Library, OpenMP | Zero third-party inference dependencies |

---

## CPU Cache Optimization Analysis

### Why In-Cache Execution Matters
In standard FP32 deep learning inference, memory bandwidth to external DRAM is the primary latency bottleneck. Arithmetic execution units often stall while waiting for weights and activations to stream over the memory bus.

### Cache Residency Breakdown
For full hardware optimization, the critical working sets must remain within the CPU cache hierarchy:

1. **Static Model Footprint (152.23 KB):**
   * Packed 2-bit ternary convolution kernels: **130.50 KB**
   * FP32 stem, dense classifier, scale ($\alpha$), and folded batch norm biases: **18.16 KB**
   * **Result:** The entire model parameter set is well within typical per-core L2 cache capacities (512 KB to 1 MB+ on modern x86 architectures). Model weights are loaded once during initialization and never cause external DRAM cache misses during repeated inference.

2. **Ping-Pong Activation Buffers (512 KB Total):**
   * The pipeline uses two reusable contiguous buffers (`buffer_A` and `buffer_B`) of size $32 \times 32 \times 64 \times 4\text{ bytes} = 256\text{ KB}$ each.
   * Alternating output and input between these two buffers avoids dynamic heap allocations (`malloc`/`free`) during runtime and keeps active feature maps residing in L2/L3 cache.

3. **im2col Workspace (~4.7 MB peak):**
   * The workspace buffer fits inside shared L3 cache on modern processors (typically 16 MB to 64+ MB), ensuring low-latency memory access without spilling to main memory.

---

## Architecture and Quantization Scheme

### 1. Straight-Through Estimator (STE) Quantization
Ternary quantization maps continuous weights to $\{-1, 0, +1\}$ during forward passes while maintaining full-precision shadow weights for gradient accumulation:

$$\alpha = \frac{1}{N} \sum \vert{}W\vert{}$$

$$\tilde{W} = \text{clip}\left(\text{round}\left(\frac{W}{\alpha}\right), -1, +1\right)$$

During backpropagation, the gradient passes through unchanged: $\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial \tilde{W}}$.

### 2. Network Topology (540,394 Total Parameters)
* **Stem (FP32):** `Conv2D(3 -> 32, 3x3)` + `BatchNorm` + `ReLU` (864 params)
* **Block 1 (Ternary):** `BitConv2D(32 -> 64, 3x3)` + `BN` + `ReLU` + `MaxPool(2x2)` (18,432 params)
* **Block 2 (Ternary):** `BitConv2D(64 -> 128, 3x3)` + `BN` + `ReLU` (73,728 params)
* **Block 3 (Ternary):** `BitConv2D(128 -> 128, 3x3)` + `BN` + `ReLU` + `MaxPool(2x2)` (147,456 params)
* **Block 4 (Ternary):** `BitConv2D(128 -> 256, 3x3)` + `BN` + `ReLU` (294,912 params)
* **Head:** `GlobalAveragePooling2D` + `Dense(256 -> 10, FP32)` (2,570 params)

### 3. Binary 2-Bit Storage Format
Weights in `{-1, 0, +1}` are bit-packed at 4 weights per `uint8_t`:
* `00` ($0$) $\rightarrow$ $0$
* `01` ($1$) $\rightarrow$ $+1$
* `10` ($2$) $\rightarrow$ $-1$
* `11` ($3$) $\rightarrow$ Unused

---

## Engine Optimization Techniques

* **Folded Batch Normalization:**
  Batch normalization scale ($\gamma$), variance ($\sigma^2$), mean ($\mu$), and bias ($\beta$) are folded directly into affine coefficients prior to inference:
  $$y = x \cdot \text{scale}_{\text{folded}} + \text{bias}_{\text{folded}}$$
* **NHWC Memory Layout:**
  Tensors are organized in channel-last (`NHWC`) order, maximizing sequential memory access along inner filter loops and aligning vector loads.
* **Tiled Branchless Matrix Multiplication:**
  The `BitConv2D` layers execute via an OpenMP-parallelized `im2col` transformation followed by a tiled GEMM with `TILE_M = 32`. This keeps inner accumulator arrays mapped directly to CPU vector registers without spilling to L1.
* **Unrolled FMA SIMD Vectorization:**
  Packed weights are unpacked into aligned float arrays during model initialization, allowing GCC/Clang to issue unrolled Fused Multiply-Accumulate (FMA) instructions during GEMM loops.

---

## Project Structure

```text
.
├── engine.cpp                  # Native C++ inference engine & benchmark harness
├── train_distillation.ipynb    # Model training, STE, distillation & evaluation
├── model_weights.bin           # Exported packed model weights
└── README.md                   # Project documentation

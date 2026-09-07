#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <random>

// ============================================================================
// 1. DATA STRUCTURES & PACKED STORAGE
// ============================================================================

// 2-bit ternary encoding:
// 00 (0) ->  0
// 01 (1) -> +1
// 10 (2) -> -1
// 11 (3) -> Unused
inline int8_t decode_2bit(uint8_t byte, int idx) {
    uint8_t code = (byte >> (idx * 2)) & 0x03;
    if (code == 1) return 1;
    if (code == 2) return -1;
    return 0;
}

inline uint8_t encode_2bit(int8_t w0, int8_t w1, int8_t w2, int8_t w3) {
    auto to_code = [](int8_t w) -> uint8_t {
        if (w == 1)  return 1;
        if (w == -1) return 2;
        return 0;
    };
    return (to_code(w0)) | (to_code(w1) << 2) | (to_code(w2) << 4) | (to_code(w3) << 6);
}

struct LayerWeights {
    bool is_ternary = false;
    int kh = 3, kw = 3, cin = 0, cout = 0;
    
    // Storage
    std::vector<float> fp32_weights;      // Used for Conv1 and Dense
    std::vector<uint8_t> packed_weights;  // Used for BitConv (4 weights per byte)
    std::vector<float> unpacked_ternary;  // Cached as pure floats for FP32 FMA SIMD
    
    // Folded BN affine parameters: y = x * bn_scale + bn_bias
    std::vector<float> bn_scale;
    std::vector<float> bn_bias;
    
    // Layer scale alpha (mean absolute value from TF STE)
    float alpha = 1.0f;
};

struct BitNetModel {
    LayerWeights conv1;      // FP32: 3x3, 3 -> 32
    LayerWeights bitconv1;   // Ternary: 3x3, 32 -> 64
    LayerWeights bitconv2;   // Ternary: 3x3, 64 -> 128
    LayerWeights bitconv3;   // Ternary: 3x3, 128 -> 128
    LayerWeights bitconv4;   // Ternary: 3x3, 128 -> 256
    LayerWeights dense;      // FP32: 256 -> 10
};

// ============================================================================
// 2. TERNARY & FP32 CONVOLUTION KERNELS (NHWC LAYOUT)
// ============================================================================

// Folded Batch Normalization + ReLU activation
inline float activate(float acc, float scale, float bias) {
    float val = acc * scale + bias;
    return (val > 0.0f) ? val : 0.0f;
}

// First Conv Layer (FP32 Input, FP32 Weights, Folded BN + ReLU)
void conv2d_fp32(const float* input, int H, int W, int Cin,
                 const LayerWeights& layer, float* output) {
    int Cout = layer.cout;
    int kh = layer.kh;
    int kw = layer.kw;
    int pad_h = kh / 2;
    int pad_w = kw / 2;

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            float* out_pixel = output + (h * W + w) * Cout;
            for (int co = 0; co < Cout; ++co) {
                float acc = 0.0f;
                for (int ky = 0; ky < kh; ++ky) {
                    int ih = h + ky - pad_h;
                    if (ih < 0 || ih >= H) continue;
                    for (int kx = 0; kx < kw; ++kx) {
                        int iw = w + kx - pad_w;
                        if (iw < 0 || iw >= W) continue;
                        
                        const float* in_ptr = input + (ih * W + iw) * Cin;
                        const float* w_ptr = layer.fp32_weights.data() + (((ky * kw + kx) * Cin) * Cout + co);
                        
                        for (int ci = 0; ci < Cin; ++ci) {
                            acc += in_ptr[ci] * w_ptr[ci * Cout];
                        }
                    }
                }
                out_pixel[co] = activate(acc, layer.bn_scale[co], layer.bn_bias[co]);
            }
        }
    }
}

// BitConv2D: Hyper-Optimized im2col + Tiled Branchless GEMM
void bitconv2d(const float* input, int H, int W, int Cin,
               const LayerWeights& layer, float* output, std::vector<float>& col_buffer) {
    int Cout = layer.cout;
    int kh = layer.kh;
    int kw = layer.kw;
    int pad_h = kh / 2;
    int pad_w = kw / 2;

    int N = H * W;                  // Total spatial pixels
    int K = kh * kw * Cin;          // Patch size (e.g., 3 * 3 * 32 = 288)
    int M = Cout;                   // Output channels

    // ==========================================================
    // PHASE 1: im2col Transformation
    // Flattens the sliding windows into a contiguous matrix (N x K)
    // ==========================================================
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int col_idx = (h * W + w) * K;
            int patch_offset = 0;
            
            for (int ky = 0; ky < kh; ++ky) {
                int ih = h + ky - pad_h;
                for (int kx = 0; kx < kw; ++kx) {
                    int iw = w + kx - pad_w;
                    
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        const float* in_ptr = input + (ih * W + iw) * Cin;
                        std::memcpy(&col_buffer[col_idx + patch_offset], in_ptr, Cin * sizeof(float));
                    } else {
                        // Zero-padding
                        std::memset(&col_buffer[col_idx + patch_offset], 0, Cin * sizeof(float));
                    }
                    patch_offset += Cin;
                }
            }
        }
    }

    // ==========================================================
    // PHASE 2: Tiled Dense Matrix Multiplication (GEMM)
    // Computes Output (N x M) = col_buffer (N x K) * weights (K x M)
    // ==========================================================
    const float* weights_fp32 = layer.unpacked_ternary.data();

    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        float* out_row = output + n * M;
        const float* a_row = col_buffer.data() + n * K;
        
        // TILE_M = 32 floats (128 bytes). This perfectly consumes 4 AVX2 registers.
        // It entirely eliminates register spilling to the L1 cache.
        const int TILE_M = 32; 
        
        for (int m_block = 0; m_block < M; m_block += TILE_M) {
            float acc[TILE_M] = {0.0f}; 
            
            for (int k = 0; k < K; ++k) {
                float a_val = a_row[k];
                const float* b_row = weights_fp32 + k * M + m_block;
                
                // Force aggressive vectorization. 
                // Because both operands are now floats, this compiles to pure FMA.
                #pragma GCC unroll 8
                for (int m = 0; m < TILE_M; ++m) {
                    acc[m] += a_val * b_row[m]; 
                }
            }

            // Apply folded Batch Norm and ReLU to the tile
            for (int m = 0; m < TILE_M; ++m) {
                int out_idx = m_block + m;
                out_row[out_idx] = activate(acc[m], layer.bn_scale[out_idx], layer.bn_bias[out_idx]);
            }
        }
    }
}

// 2x2 Max Pooling (Stride 2)
void maxpool2d(const float* input, int H, int W, int C, float* output) {
    int out_H = H / 2;
    int out_W = W / 2;
    for (int h = 0; h < out_H; ++h) {
        for (int w = 0; w < out_W; ++w) {
            float* out_pixel = output + (h * out_W + w) * C;
            for (int c = 0; c < C; ++c) {
                float v00 = input[((2 * h) * W + (2 * w)) * C + c];
                float v01 = input[((2 * h) * W + (2 * w + 1)) * C + c];
                float v10 = input[((2 * h + 1) * W + (2 * w)) * C + c];
                float v11 = input[((2 * h + 1) * W + (2 * w + 1)) * C + c];
                out_pixel[c] = std::max({v00, v01, v10, v11});
            }
        }
    }
}

// Global Average Pooling: (H, W, C) -> (C)
void global_avg_pool2d(const float* input, int H, int W, int C, float* output) {
    float norm = 1.0f / (H * W);
    std::fill(output, output + C, 0.0f);
    for (int i = 0; i < H * W; ++i) {
        for (int c = 0; c < C; ++c) {
            output[c] += input[i * C + c];
        }
    }
    for (int c = 0; c < C; ++c) {
        output[c] *= norm;
    }
}

// Final Linear Classifier
void dense_forward(const float* input, const LayerWeights& layer, float* output) {
    int Cin = layer.cin;
    int Cout = layer.cout;
    for (int co = 0; co < Cout; ++co) {
        float acc = layer.bn_bias[co]; // Bias
        for (int ci = 0; ci < Cin; ++ci) {
            acc += input[ci] * layer.fp32_weights[ci * Cout + co];
        }
        output[co] = acc;
    }
}

// ============================================================================
// 3. COMPLETE PIPELINE WITH PING-PONG MEMORY MANAGEMENT
// ============================================================================

struct ExecutionProfile {
    double t_conv1 = 0;
    double t_bitconv1 = 0;
    double t_pool1 = 0;
    double t_bitconv2 = 0;
    double t_bitconv3 = 0;
    double t_pool2 = 0;
    double t_bitconv4 = 0;
    double t_gap = 0;
    double t_dense = 0;
    double t_total = 0;
};

void run_inference(const BitNetModel& model, const float* image_input,
                   float* logits_output, float* buffer_A, float* buffer_B,
                   std::vector<float>& col_buffer,
                   ExecutionProfile* prof = nullptr) {
    auto now = []() { return std::chrono::high_resolution_clock::now(); };
    auto dur = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    auto t0 = now();
    // 1. Conv2D_FP32 (32, 32, 3) -> BufA: (32, 32, 32)
    conv2d_fp32(image_input, 32, 32, 3, model.conv1, buffer_A);
    auto t1 = now();

    // 2. BitConv2D_1 (32, 32, 32) -> BufB: (32, 32, 64)
    bitconv2d(buffer_A, 32, 32, 32, model.bitconv1, buffer_B, col_buffer);
    auto t2 = now();

    // 3. MaxPool2D_1 (32, 32, 64) -> BufA: (16, 16, 64)
    maxpool2d(buffer_B, 32, 32, 64, buffer_A);
    auto t3 = now();

    // 4. BitConv2D_2 (16, 16, 64) -> BufB: (16, 16, 128)
    bitconv2d(buffer_A, 16, 16, 64, model.bitconv2, buffer_B, col_buffer);
    auto t4 = now();

    // 5. BitConv2D_3 (16, 16, 128) -> BufA: (16, 16, 128)
    bitconv2d(buffer_B, 16, 16, 128, model.bitconv3, buffer_A, col_buffer);
    auto t5 = now();

    // 6. MaxPool2D_2 (16, 16, 128) -> BufB: (8, 8, 128)
    maxpool2d(buffer_A, 16, 16, 128, buffer_B);
    auto t6 = now();

    // 7. BitConv2D_4 (8, 8, 128) -> BufA: (8, 8, 256)
    bitconv2d(buffer_B, 8, 8, 128, model.bitconv4, buffer_A, col_buffer);
    auto t7 = now();

    // 8. GlobalAveragePooling2D (8, 8, 256) -> BufB[0..255]
    global_avg_pool2d(buffer_A, 8, 8, 256, buffer_B);
    auto t8 = now();

    // 9. Dense (256) -> logits_output: (10)
    dense_forward(buffer_B, model.dense, logits_output);
    auto t9 = now();

    if (prof) {
        prof->t_conv1    = dur(t0, t1);
        prof->t_bitconv1 = dur(t1, t2);
        prof->t_pool1    = dur(t2, t3);
        prof->t_bitconv2 = dur(t3, t4);
        prof->t_bitconv3 = dur(t4, t5);
        prof->t_pool2    = dur(t5, t6);
        prof->t_bitconv4 = dur(t6, t7);
        prof->t_gap      = dur(t7, t8);
        prof->t_dense    = dur(t8, t9);
        prof->t_total    = dur(t0, t9);
    }
}

// ============================================================================
// 4. SYNTHETIC INITIALIZER & BINARY FILE LOADER
// ============================================================================

void unpack_layer_weights(LayerWeights& lw) {
    if (!lw.is_ternary) return;
    size_t total_weights = lw.kh * lw.kw * lw.cin * lw.cout;
    lw.unpacked_ternary.resize(total_weights);
    for (size_t i = 0; i < total_weights; ++i) {
        uint8_t byte = lw.packed_weights[i / 4];
        // Cast the int8_t directly to float here so the inference loop never has to
        lw.unpacked_ternary[i] = static_cast<float>(decode_2bit(byte, i % 4)); 
    }
}

void init_synthetic_layer(LayerWeights& lw, int kh, int kw, int cin, int cout, bool is_ternary, std::mt19937& rng) {
    lw.is_ternary = is_ternary;
    lw.kh = kh; lw.kw = kw; lw.cin = cin; lw.cout = cout;
    lw.bn_scale.resize(cout, 1.0f);
    lw.bn_bias.resize(cout, 0.0f);
    lw.alpha = 0.05f;

    size_t total = kh * kw * cin * cout;
    if (is_ternary) {
        std::discrete_distribution<int> dist({30, 40, 30}); // 30% -1, 40% 0, 30% +1 (~40% sparsity)
        std::vector<int8_t> temp(total);
        for (size_t i = 0; i < total; ++i) {
            int val = dist(rng);
            temp[i] = (val == 0) ? -1 : ((val == 1) ? 0 : 1);
        }
        lw.packed_weights.resize((total + 3) / 4);
        for (size_t i = 0; i < total; i += 4) {
            int8_t w0 = temp[i];
            int8_t w1 = (i + 1 < total) ? temp[i + 1] : 0;
            int8_t w2 = (i + 2 < total) ? temp[i + 2] : 0;
            int8_t w3 = (i + 3 < total) ? temp[i + 3] : 0;
            lw.packed_weights[i / 4] = encode_2bit(w0, w1, w2, w3);
        }
        unpack_layer_weights(lw);
    } else {
        std::normal_distribution<float> dist(0.0f, 0.05f);
        lw.fp32_weights.resize(total);
        for (size_t i = 0; i < total; ++i) {
            lw.fp32_weights[i] = dist(rng);
        }
    }
}

BitNetModel create_synthetic_model() {
    std::mt19937 rng(42);
    BitNetModel m;
    init_synthetic_layer(m.conv1, 3, 3, 3, 32, false, rng);
    init_synthetic_layer(m.bitconv1, 3, 3, 32, 64, true, rng);
    init_synthetic_layer(m.bitconv2, 3, 3, 64, 128, true, rng);
    init_synthetic_layer(m.bitconv3, 3, 3, 128, 128, true, rng);
    init_synthetic_layer(m.bitconv4, 3, 3, 128, 256, true, rng);
    init_synthetic_layer(m.dense, 1, 1, 256, 10, false, rng);
    return m;
}

bool load_binary_weights(const std::string& filepath, BitNetModel& m) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    auto read_layer = [&](LayerWeights& lw, bool is_ternary, int kh, int kw, int cin, int cout) {
        lw.is_ternary = is_ternary;
        lw.kh = kh; lw.kw = kw; lw.cin = cin; lw.cout = cout;
        lw.bn_scale.resize(cout);
        lw.bn_bias.resize(cout);
        
        file.read(reinterpret_cast<char*>(&lw.alpha), sizeof(float));
        file.read(reinterpret_cast<char*>(lw.bn_scale.data()), cout * sizeof(float));
        file.read(reinterpret_cast<char*>(lw.bn_bias.data()), cout * sizeof(float));

        size_t total = kh * kw * cin * cout;
        if (is_ternary) {
            size_t packed_size = (total + 3) / 4;
            lw.packed_weights.resize(packed_size);
            file.read(reinterpret_cast<char*>(lw.packed_weights.data()), packed_size);
            unpack_layer_weights(lw);
        } else {
            lw.fp32_weights.resize(total);
            file.read(reinterpret_cast<char*>(lw.fp32_weights.data()), total * sizeof(float));
        }
    };

    read_layer(m.conv1, false, 3, 3, 3, 32);
    read_layer(m.bitconv1, true, 3, 3, 32, 64);
    read_layer(m.bitconv2, true, 3, 3, 64, 128);
    read_layer(m.bitconv3, true, 3, 3, 128, 128);
    read_layer(m.bitconv4, true, 3, 3, 128, 256);
    read_layer(m.dense, false, 1, 1, 256, 10);

    return true;
}

// ============================================================================
// 5. MAIN BENCHMARK & METRICS PROFILER
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "===============================================================\n";
    std::cout << "  Ternary CNN Inference Engine & Microbenchmark (CIFAR-10)     \n";
    std::cout << "===============================================================\n";

    BitNetModel model;
    std::string weight_path = (argc > 1) ? argv[1] : "model_weights.bin";

    if (load_binary_weights(weight_path, model)) {
        std::cout << "[+] Loaded trained weights from: " << weight_path << "\n";
    } else {
        std::cout << "[i] Weights file '" << weight_path << "' not found.\n";
        std::cout << "[i] Generating synthetic ternary weights for performance benchmarking.\n";
        model = create_synthetic_model();
    }

    // Measure Footprint
    size_t fp32_equivalent_bytes = (864 + 18432 + 73728 + 147456 + 294912 + 2570 + 1216) * 4;
    size_t packed_ternary_bytes = model.bitconv1.packed_weights.size() +
                                  model.bitconv2.packed_weights.size() +
                                  model.bitconv3.packed_weights.size() +
                                  model.bitconv4.packed_weights.size();
    size_t aux_fp32_bytes = (model.conv1.fp32_weights.size() + model.dense.fp32_weights.size() + 1216) * 4;
    size_t actual_storage_bytes = packed_ternary_bytes + aux_fp32_bytes;

    std::cout << "\n------------------ MEMORY FOOTPRINT AUDIT -------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Unquantized FP32 Weight Footprint : " << (fp32_equivalent_bytes / 1024.0) << " KB\n";
    std::cout << "  Ternary Weights (2-bit packed)    : " << (packed_ternary_bytes / 1024.0) << " KB\n";
    std::cout << "  Auxiliary FP32 Weights & BN Biases: " << (aux_fp32_bytes / 1024.0) << " KB\n";
    std::cout << "  Total Static Model Footprint      : " << (actual_storage_bytes / 1024.0) << " KB\n";
    std::cout << "  Compression Ratio vs FP32 Baseline: " << (double)fp32_equivalent_bytes / actual_storage_bytes << "x\n";

    // Allocate Ping-Pong activation buffers
    // Max buffer size needed is 32*32*64 = 65,536 floats = 256 KB
    const size_t BUFFER_SIZE = 32 * 32 * 64;
    std::vector<float> buffer_A(BUFFER_SIZE, 0.0f);
    std::vector<float> buffer_B(BUFFER_SIZE, 0.0f);
    
    // Allocate an im2col workspace buffer.
    // Max size required: 32x32 spatial * 3x3 kernel * 128 channels = 1,179,648 floats (~4.7 MB)
    std::vector<float> col_buffer(32 * 32 * 3 * 3 * 128, 0.0f);
    
    std::cout << "  Peak Activation Cache Allocation  : " << (2 * BUFFER_SIZE * sizeof(float) / 1024.0) << " KB\n";

    // Synthetic CIFAR-10 image: 32x32x3, scaled to [-1, 1]
    std::vector<float> dummy_image(32 * 32 * 3);
    for (size_t i = 0; i < dummy_image.size(); ++i) {
        dummy_image[i] = ((i % 256) - 127.5f) / 127.5f;
    }
    float logits[10];

    // Warm-up runs
    std::cout << "\n[+] Warming up caches (10 iterations)...\n";
    for (int i = 0; i < 10; ++i) {
        run_inference(model, dummy_image.data(), logits, buffer_A.data(), buffer_B.data(), col_buffer);
    }

    // Benchmark Run
    const int BENCHMARK_ITERATIONS = 200;
    std::cout << "[+] Benchmarking inference across " << BENCHMARK_ITERATIONS << " runs...\n\n";

    ExecutionProfile avg_prof;
    std::vector<double> latencies;
    latencies.reserve(BENCHMARK_ITERATIONS);

    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        ExecutionProfile p;
        run_inference(model, dummy_image.data(), logits, buffer_A.data(), buffer_B.data(), col_buffer, &p);
        
        latencies.push_back(p.t_total);
        avg_prof.t_conv1    += p.t_conv1;
        avg_prof.t_bitconv1 += p.t_bitconv1;
        avg_prof.t_pool1    += p.t_pool1;
        avg_prof.t_bitconv2 += p.t_bitconv2;
        avg_prof.t_bitconv3 += p.t_bitconv3;
        avg_prof.t_pool2    += p.t_pool2;
        avg_prof.t_bitconv4 += p.t_bitconv4;
        avg_prof.t_gap      += p.t_gap;
        avg_prof.t_dense    += p.t_dense;
        avg_prof.t_total    += p.t_total;
    }

    double iters = static_cast<double>(BENCHMARK_ITERATIONS);
    avg_prof.t_conv1 /= iters;
    avg_prof.t_bitconv1 /= iters;
    avg_prof.t_pool1 /= iters;
    avg_prof.t_bitconv2 /= iters;
    avg_prof.t_bitconv3 /= iters;
    avg_prof.t_pool2 /= iters;
    avg_prof.t_bitconv4 /= iters;
    avg_prof.t_gap /= iters;
    avg_prof.t_dense /= iters;
    avg_prof.t_total /= iters;

    std::sort(latencies.begin(), latencies.end());
    double min_lat = latencies.front();
    double max_lat = latencies.back();
    double median_lat = latencies[BENCHMARK_ITERATIONS / 2];

    std::cout << "----------------- LAYER LATENCY BREAKDOWN -------------------\n";
    auto print_layer = [](const char* name, double ms, double total) {
        std::cout << "  " << std::left << std::setw(28) << name 
                  << ": " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << ms << " ms ("
                  << std::setw(5) << std::setprecision(1) << (ms / total * 100.0) << "%)\n";
    };
    print_layer("1. Conv2D FP32 (3->32)", avg_prof.t_conv1, avg_prof.t_total);
    print_layer("2. BitConv2D_1 (32->64)", avg_prof.t_bitconv1, avg_prof.t_total);
    print_layer("   MaxPool2D_1", avg_prof.t_pool1, avg_prof.t_total);
    print_layer("3. BitConv2D_2 (64->128)", avg_prof.t_bitconv2, avg_prof.t_total);
    print_layer("4. BitConv2D_3 (128->128)", avg_prof.t_bitconv3, avg_prof.t_total);
    print_layer("   MaxPool2D_2", avg_prof.t_pool2, avg_prof.t_total);
    print_layer("5. BitConv2D_4 (128->256)", avg_prof.t_bitconv4, avg_prof.t_total);
    print_layer("6. GlobalAvgPool", avg_prof.t_gap, avg_prof.t_total);
    print_layer("7. Dense Output (256->10)", avg_prof.t_dense, avg_prof.t_total);
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  Total Latency (Mean)      : " << avg_prof.t_total << " ms\n";
    std::cout << "  Total Latency (Median)    : " << median_lat << " ms\n";
    std::cout << "  Total Latency (Min..Max)  : " << min_lat << " .. " << max_lat << " ms\n";
    std::cout << "  Throughput                : " << (1000.0 / avg_prof.t_total) << " inferences/sec\n";
    std::cout << "=============================================================\n";

    return 0;
}

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
#include <immintrin.h>
#include <omp.h>

// ============================================================================
// 1. 32-BYTE ALIGNED ALLOCATOR (Fixed with rebind and equality operators)
// ============================================================================

template <typename T, size_t Alignment = 32>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    T* allocate(size_t n) {
        size_t bytes = n * sizeof(T);
        size_t rem = bytes % Alignment;
        if (rem != 0) bytes += (Alignment - rem);
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(bytes, Alignment);
#else
        ptr = std::aligned_alloc(Alignment, bytes);
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
};

template <typename T, typename U, size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) noexcept {
    return true;
}

template <typename T, typename U, size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) noexcept {
    return false;
}

template <typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T, 32>>;

// ============================================================================
// 2. DATA STRUCTURES & PACKED STORAGE
// ============================================================================

inline int8_t decode_2bit(uint8_t byte, int idx) {
    uint8_t code = (byte >> (idx * 2)) & 0x03;
    if (code == 1) return 1;
    if (code == 2) return -1;
    return 0;
}

struct LayerWeights {
    bool is_ternary = false;
    int kh = 3, kw = 3, cin = 0, cout = 0;
    
    aligned_vector<float> fp32_weights;
    std::vector<uint8_t> packed_weights;
    aligned_vector<int8_t> transposed_int8_weights; // Stored in [M][K] layout
    
    aligned_vector<float> bn_scale;
    aligned_vector<float> bn_bias;
    float alpha = 1.0f;
};

struct BitNetModel {
    LayerWeights conv1;      
    LayerWeights bitconv1;   
    LayerWeights bitconv2;   
    LayerWeights bitconv3;   
    LayerWeights bitconv4;   
    LayerWeights dense;      
};

// ============================================================================
// 3. AVX2 INTEGER MICROKERNEL & LAYERS
// ============================================================================

inline int32_t hsum_epi32_avx2(__m256i x) {
    __m128i hi = _mm256_extracti128_si256(x, 1);
    __m128i lo = _mm256_castsi256_si128(x);
    lo = _mm_add_epi32(hi, lo);
    hi = _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2));
    lo = _mm_add_epi32(hi, lo);
    hi = _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1));
    lo = _mm_add_epi32(hi, lo);
    return _mm_cvtsi128_si32(lo);
}

inline int32_t dot_product_int8_avx2(const uint8_t* __restrict__ a, const int8_t* __restrict__ b, int K) {
    __m256i vacc = _mm256_setzero_si256();
    __m256i vones = _mm256_set1_epi16(1);
    
    // Process 32 elements per iteration
    for (int k = 0; k < K; k += 32) {
        __m256i va = _mm256_load_si256((const __m256i*)(a + k));
        __m256i vb = _mm256_load_si256((const __m256i*)(b + k));
        
        __m256i vres16 = _mm256_maddubs_epi16(va, vb);
        __m256i vres32 = _mm256_madd_epi16(vres16, vones);
        
        vacc = _mm256_add_epi32(vacc, vres32);
    }
    return hsum_epi32_avx2(vacc);
}

void conv2d_fp32(const float* input, int H, int W, int Cin, const LayerWeights& layer, float* output) {
    int pad_h = layer.kh / 2, pad_w = layer.kw / 2;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            float* out_pixel = output + (h * W + w) * layer.cout;
            for (int co = 0; co < layer.cout; ++co) {
                float acc = 0.0f;
                for (int ky = 0; ky < layer.kh; ++ky) {
                    int ih = h + ky - pad_h;
                    if (ih < 0 || ih >= H) continue;
                    for (int kx = 0; kx < layer.kw; ++kx) {
                        int iw = w + kx - pad_w;
                        if (iw < 0 || iw >= W) continue;
                        const float* in_ptr = input + (ih * W + iw) * Cin;
                        const float* w_ptr = layer.fp32_weights.data() + (((ky * layer.kw + kx) * Cin) * layer.cout + co);
                        for (int ci = 0; ci < Cin; ++ci) {
                            acc += in_ptr[ci] * w_ptr[ci * layer.cout];
                        }
                    }
                }
                float val = acc * layer.bn_scale[co] + layer.bn_bias[co];
                out_pixel[co] = (val > 0.0f) ? val : 0.0f; // ReLU
            }
        }
    }
}

void bitconv2d_optimized(const float* __restrict__ input, int H, int W, int Cin, const LayerWeights& layer, float* __restrict__ output) {
    const int Cout = layer.cout;
    const int K = layer.kh * layer.kw * Cin;
    const int pad_h = layer.kh / 2, pad_w = layer.kw / 2;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            
            // L1-resident workspace (max K is 1152, using 2048 for safety/padding)
            alignas(32) float patch_fp32[2048]; 
            alignas(32) uint8_t patch_uint8[2048];

            int offset = 0;
            float max_val = 1e-5f;

            // Fused im2col extraction
            for (int ky = 0; ky < layer.kh; ++ky) {
                int ih = h + ky - pad_h;
                for (int kx = 0; kx < layer.kw; ++kx) {
                    int iw = w + kx - pad_w;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        const float* in_ptr = input + (ih * W + iw) * Cin;
                        for (int c = 0; c < Cin; ++c) {
                            float val = in_ptr[c];
                            patch_fp32[offset++] = val;
                            if (val > max_val) max_val = val;
                        }
                    } else {
                        for (int c = 0; c < Cin; ++c) patch_fp32[offset++] = 0.0f;
                    }
                }
            }

            // Dynamic Quantization to uint8_t [0, 127]
            float scale = 127.0f / max_val;
            float inv_scale = max_val / 127.0f;
            for (int k = 0; k < K; ++k) {
                patch_uint8[k] = static_cast<uint8_t>(patch_fp32[k] * scale);
            }

            // Pad remainder of K to 32 for safe AVX2 loads
            int remainder = K % 32;
            int K_padded = (remainder == 0) ? K : K + (32 - remainder);
            for (int k = K; k < K_padded; ++k) patch_uint8[k] = 0;

            // Integer GEMM + Dequantization + Folded BN + ReLU
            float* out_pixel = output + (h * W + w) * Cout;
            for (int m = 0; m < Cout; ++m) {
                const int8_t* b_row = layer.transposed_int8_weights.data() + m * K_padded;
                
                int32_t dot = dot_product_int8_avx2(patch_uint8, b_row, K_padded);
                
                float fp_dot = static_cast<float>(dot) * inv_scale * layer.alpha;
                float activated = fp_dot * layer.bn_scale[m] + layer.bn_bias[m];
                
                out_pixel[m] = activated > 0.0f ? activated : 0.0f;
            }
        }
    }
}

void maxpool2d(const float* input, int H, int W, int C, float* output) {
    for (int h = 0; h < H / 2; ++h) {
        for (int w = 0; w < W / 2; ++w) {
            float* out_pixel = output + (h * (W / 2) + w) * C;
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

void global_avg_pool2d(const float* input, int H, int W, int C, float* output) {
    float norm = 1.0f / (H * W);
    std::fill(output, output + C, 0.0f);
    for (int i = 0; i < H * W; ++i) {
        for (int c = 0; c < C; ++c) output[c] += input[i * C + c];
    }
    for (int c = 0; c < C; ++c) output[c] *= norm;
}

void dense_forward(const float* input, const LayerWeights& layer, float* output) {
    for (int co = 0; co < layer.cout; ++co) {
        float acc = layer.bn_bias[co];
        for (int ci = 0; ci < layer.cin; ++ci) {
            acc += input[ci] * layer.fp32_weights[ci * layer.cout + co];
        }
        output[co] = acc;
    }
}

// ============================================================================
// 4. PIPELINE & INITIALIZATION
// ============================================================================

void run_inference(const BitNetModel& model, const float* image_input, float* logits_output, float* buffer_A, float* buffer_B) {
    conv2d_fp32(image_input, 32, 32, 3, model.conv1, buffer_A);
    bitconv2d_optimized(buffer_A, 32, 32, 32, model.bitconv1, buffer_B);
    maxpool2d(buffer_B, 32, 32, 64, buffer_A);
    bitconv2d_optimized(buffer_A, 16, 16, 64, model.bitconv2, buffer_B);
    bitconv2d_optimized(buffer_B, 16, 16, 128, model.bitconv3, buffer_A);
    maxpool2d(buffer_A, 16, 16, 128, buffer_B);
    bitconv2d_optimized(buffer_B, 8, 8, 128, model.bitconv4, buffer_A);
    global_avg_pool2d(buffer_A, 8, 8, 256, buffer_B);
    dense_forward(buffer_B, model.dense, logits_output);
}

void unpack_layer_weights(LayerWeights& lw) {
    if (!lw.is_ternary) return;
    int K = lw.kh * lw.kw * lw.cin;
    int M = lw.cout;
    
    // Pad K to multiple of 32 for safe AVX2 loads
    int remainder = K % 32;
    int K_padded = (remainder == 0) ? K : K + (32 - remainder);
    
    lw.transposed_int8_weights.resize(K_padded * M, 0);
    
    // Transpose from [K][M] to [M][K_padded]
    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            int i = k * M + m;
            uint8_t byte = lw.packed_weights[i / 4];
            int8_t val = decode_2bit(byte, i % 4);
            lw.transposed_int8_weights[m * K_padded + k] = val;
        }
    }
}

void init_synthetic_layer(LayerWeights& lw, int kh, int kw, int cin, int cout, bool is_ternary) {
    lw.is_ternary = is_ternary;
    lw.kh = kh; lw.kw = kw; lw.cin = cin; lw.cout = cout;
    lw.bn_scale.resize(cout, 1.0f);
    lw.bn_bias.resize(cout, 0.0f);
    lw.alpha = 0.05f;
    size_t total = kh * kw * cin * cout;

    if (is_ternary) {
        lw.packed_weights.resize((total + 3) / 4, 0xAA); // 0xAA generates a mix of 1s and -1s
        unpack_layer_weights(lw);
    } else {
        lw.fp32_weights.resize(total, 0.1f);
    }
}

BitNetModel create_synthetic_model() {
    BitNetModel m;
    init_synthetic_layer(m.conv1, 3, 3, 3, 32, false);
    init_synthetic_layer(m.bitconv1, 3, 3, 32, 64, true);
    init_synthetic_layer(m.bitconv2, 3, 3, 64, 128, true);
    init_synthetic_layer(m.bitconv3, 3, 3, 128, 128, true);
    init_synthetic_layer(m.bitconv4, 3, 3, 128, 256, true);
    init_synthetic_layer(m.dense, 1, 1, 256, 10, false);
    return m;
}

// ============================================================================
// 5. MAIN BENCHMARK
// ============================================================================

int main() {
    std::cout << "===============================================================\n";
    std::cout << "  AVX2 Integer Ternary CNN Inference Engine (CIFAR-10)         \n";
    std::cout << "===============================================================\n";
    
    BitNetModel model = create_synthetic_model();

    // Ping-pong activation caches
    aligned_vector<float> buffer_A(32 * 32 * 64, 0.0f);
    aligned_vector<float> buffer_B(32 * 32 * 64, 0.0f);
    
    // Mock image scaled to [-1, 1]
    aligned_vector<float> dummy_image(32 * 32 * 3, 0.5f);
    float logits[10];

    std::cout << "[+] Warming up caches...\n";
    for (int i = 0; i < 10; ++i) {
        run_inference(model, dummy_image.data(), logits, buffer_A.data(), buffer_B.data());
    }

    const int ITERATIONS = 200;
    std::cout << "[+] Running benchmark (" << ITERATIONS << " iterations)...\n\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < ITERATIONS; ++i) {
        run_inference(model, dummy_image.data(), logits, buffer_A.data(), buffer_B.data());
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_latency = total_ms / ITERATIONS;
    double throughput = 1000.0 / avg_latency;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "---------------------------------------------------------------\n";
    std::cout << "  Mean Latency : " << avg_latency << " ms\n";
    std::cout << "  Throughput   : " << throughput << " Inferences/sec\n";
    std::cout << "===============================================================\n";

    return 0;
}

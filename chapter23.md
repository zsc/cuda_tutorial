# 第23章：量化与低精度计算

量化技术是深度学习模型部署的关键优化手段，通过降低数值精度来换取更高的计算吞吐量和更低的内存占用。本章深入探讨CUDA平台上的量化技术实现，从INT8/INT4基础到自定义量化算子，涵盖训练和推理全流程。我们将重点关注自动驾驶和具身智能场景中的实际应用，帮助你实现4x-8x的推理加速，同时保持可接受的精度损失。

## 23.1 INT8/INT4量化技术

量化是将浮点数值映射到有限整数集合的过程。在深度学习中，我们通常将FP32/FP16权重和激活值量化为INT8甚至INT4，实现存储压缩和计算加速。现代GPU从Turing架构开始提供专门的INT8 Tensor Core，Ampere架构进一步支持INT4运算，使得量化成为生产部署的标配技术。

### 23.1.1 量化基础理论

量化过程可以表示为两个变换：量化（Quantize）和反量化（Dequantize）。

**均匀量化公式：**
```
Q(x) = round(x / scale + zero_point)
DQ(q) = (q - zero_point) * scale
```

其中scale是缩放因子，zero_point是零点偏移。对于INT8量化，q ∈ [-128, 127]（有符号）或 [0, 255]（无符号）。

**量化参数计算：**
```
scale = (x_max - x_min) / (q_max - q_min)
zero_point = round(q_min - x_min / scale)
```

关键在于如何确定x_min和x_max。常见策略包括：
- **MinMax量化**：使用实际最小值和最大值
- **百分位量化**：使用99.9%分位数，裁剪异常值
- **KL散度量化**：最小化量化前后分布的KL散度
- **MSE量化**：最小化均方误差

**INT4量化的特殊考虑：**

INT4仅有16个量化等级，需要更精细的处理：
- 通常采用对称量化（zero_point = 0）
- 结合分组量化（group-wise）提高精度
- 使用查找表（LUT）加速反量化

### 23.1.2 对称与非对称量化

**对称量化**假设数据分布关于零对称：
```
scale = max(|x_max|, |x_min|) / q_max
zero_point = 0
Q(x) = round(x / scale)
```

优点：
- 计算简单，无需存储zero_point
- 零值精确表示（Q(0) = 0）
- 适合权重量化

**非对称量化**允许任意范围映射：
```
scale = (x_max - x_min) / (q_max - q_min)
zero_point = round(q_min - x_min / scale)
```

优点：
- 更好利用量化范围
- 适合激活值（如ReLU后的非负分布）
- 更小的量化误差

**CUDA实现对比：**
```cuda
// 对称量化内核
__global__ void symmetric_quantize_kernel(
    const float* input, int8_t* output, 
    float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] * (127.0f / scale);
        val = fmaxf(-128.0f, fminf(127.0f, val));
        output[idx] = __float2int_rn(val);
    }
}

// 非对称量化内核
__global__ void asymmetric_quantize_kernel(
    const float* input, uint8_t* output,
    float scale, int zero_point, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] / scale + zero_point;
        val = fmaxf(0.0f, fminf(255.0f, val));
        output[idx] = __float2uint_rn(val);
    }
}
```

### 23.1.3 逐层与逐通道量化

**逐层量化（Per-layer）**：
整层共享一组量化参数，实现简单但精度损失较大。

**逐通道量化（Per-channel）**：
每个输出通道独立量化，提高精度但增加存储开销。

**逐组量化（Per-group）**：
将通道分组，组内共享参数，平衡精度和效率。

```cuda
// 逐通道量化的高效实现
__global__ void per_channel_quantize_kernel(
    const float* input,      // [N, C, H, W]
    int8_t* output,         
    const float* scales,     // [C]
    const int* zero_points,  // [C]
    int N, int C, int H, int W) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    
    if (idx < total) {
        int n = idx / (C * H * W);
        int c = (idx / (H * W)) % C;
        
        float scale = scales[c];
        int zp = zero_points[c];
        
        float val = input[idx] / scale + zp;
        val = fmaxf(-128.0f, fminf(127.0f, val));
        output[idx] = __float2int_rn(val);
    }
}
```

**分组量化优化：**
```
     Original Tensor
    [C0][C1][C2][C3]...
         |
         v
    Group Size = 128
    [G0: C0-C127] [G1: C128-C255]...
         |              |
    scale0, zp0    scale1, zp1
```

### 23.1.4 CUDA中的INT8/INT4运算

**INT8 Tensor Core编程：**

Turing架构引入的INT8 Tensor Core提供极高的计算吞吐量：
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// INT8 WMMA示例
__global__ void int8_wmma_gemm(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K) {
    
    const int warpM = 16, warpN = 16, warpK = 16;
    
    // 声明WMMA片段
    fragment<matrix_a, warpM, warpN, warpK, int8_t, row_major> a_frag;
    fragment<matrix_b, warpM, warpN, warpK, int8_t, col_major> b_frag;
    fragment<accumulator, warpM, warpN, warpK, int32_t> c_frag;
    
    // 初始化累加器
    fill_fragment(c_frag, 0);
    
    // 计算warp的全局位置
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    for (int k = 0; k < K; k += warpK) {
        // 加载矩阵片段
        load_matrix_sync(a_frag, A + ..., K);
        load_matrix_sync(b_frag, B + ..., K);
        
        // 执行矩阵乘法
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果
    store_matrix_sync(C + ..., c_frag, N, mem_row_major);
}
```

**INT4运算的位打包：**
```cuda
// INT4打包和解包
__device__ __forceinline__ int8_t pack_int4(int4 a, int4 b) {
    return (a & 0x0F) | ((b & 0x0F) << 4);
}

__device__ __forceinline__ void unpack_int4(
    int8_t packed, int4& a, int4& b) {
    a = (packed & 0x0F);
    if (a & 0x08) a |= 0xFFFFFFF0;  // 符号扩展
    b = (packed >> 4) & 0x0F;
    if (b & 0x08) b |= 0xFFFFFFF0;
}

// INT4 GEMM内核
__global__ void int4_gemm_kernel(
    const int8_t* A_packed,  // 两个INT4打包成一个INT8
    const int8_t* B_packed,
    int8_t* C_packed,
    int M, int N, int K) {
    
    // 使用向量化加载提高带宽利用
    int4 a_vec = *reinterpret_cast<const int4*>(A_packed + ...);
    
    // 解包并计算
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int4 a_low, a_high;
        unpack_int4(a_vec.x, a_low, a_high);
        // 执行INT4运算...
    }
}
```

**DP4A指令优化：**
```cuda
// 使用DP4A指令加速INT8点积
__global__ void dp4a_gemm_kernel(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int32_t sum = 0;
        
        for (int k = 0; k < K; k += 4) {
            // 加载4个INT8值并打包
            int32_t a_packed = *reinterpret_cast<const int32_t*>(&A[row * K + k]);
            int32_t b_packed = *reinterpret_cast<const int32_t*>(&B[col * K + k]);
            
            // DP4A: sum += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
            sum = __dp4a(a_packed, b_packed, sum);
        }
        
        C[row * N + col] = sum;
    }
}
```

### 23.1.5 量化误差分析

量化引入的误差主要包括：
1. **舍入误差**：连续值映射到离散级别
2. **饱和误差**：超出量化范围的值被裁剪
3. **累积误差**：多层量化误差传播

**误差度量：**
```
SQNR (Signal-to-Quantization-Noise Ratio) = 10 * log10(P_signal / P_noise)
其中 P_noise = E[(x - Q(x))²]
```

**误差分析工具：**
```cuda
// 量化误差统计内核
__global__ void quantization_error_kernel(
    const float* original,
    const float* quantized,
    float* mse, float* max_error,
    int n) {
    
    __shared__ float shared_mse[256];
    __shared__ float shared_max[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float local_mse = 0.0f;
    float local_max = 0.0f;
    
    if (idx < n) {
        float error = fabsf(original[idx] - quantized[idx]);
        local_mse = error * error;
        local_max = error;
    }
    
    shared_mse[tid] = local_mse;
    shared_max[tid] = local_max;
    __syncthreads();
    
    // 规约求和与最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mse[tid] += shared_mse[tid + s];
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(mse, shared_mse[0]);
        atomicMax(max_error, shared_max[0]);
    }
}
```

**敏感层识别：**
某些层对量化更敏感，需要特殊处理：
- 网络首尾层通常保持FP16/FP32
- 残差连接的加法操作易累积误差
- 小卷积核（1x1）的量化影响较大

## 23.2 量化感知训练

量化感知训练（Quantization-Aware Training, QAT）在训练过程中模拟量化效应，使模型学习适应量化误差。相比训练后量化（Post-Training Quantization, PTQ），QAT能显著减少精度损失，特别是对于极低比特量化（INT4及以下）更是必不可少。

### 23.2.1 伪量化技术

伪量化（Fake Quantization）在前向传播中模拟量化/反量化过程，但保持浮点运算以支持梯度计算。

**伪量化的数学表示：**
```
y = FakeQuant(x) = DQ(Q(x)) = round(x/s + z) * s - z * s
```

**直通估计器（Straight-Through Estimator, STE）：**
```
前向：y = FakeQuant(x)
反向：∂L/∂x = ∂L/∂y  (梯度直接传递)
```

**CUDA实现：**
```cuda
// 伪量化前向内核
__global__ void fake_quant_forward_kernel(
    const float* input,
    float* output,
    float* scale,
    int* zero_point,
    int n, int num_bits = 8) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = *scale;
        int z = *zero_point;
        
        // 计算量化范围
        int qmin = -(1 << (num_bits - 1));
        int qmax = (1 << (num_bits - 1)) - 1;
        
        // 量化
        float val = input[idx] / s + z;
        val = roundf(val);
        val = fmaxf(qmin, fminf(qmax, val));
        
        // 反量化
        output[idx] = (val - z) * s;
    }
}

// 伪量化反向内核（STE）
__global__ void fake_quant_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    float* grad_scale,
    float* scale,
    int* zero_point,
    int n, int num_bits = 8) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = *scale;
        int z = *zero_point;
        
        // 计算量化范围
        float qmin = -(1 << (num_bits - 1));
        float qmax = (1 << (num_bits - 1)) - 1;
        
        // 检查是否在量化范围内
        float val = input[idx] / s + z;
        bool in_range = (val >= qmin && val <= qmax);
        
        // STE：范围内直接传递梯度，范围外截断
        grad_input[idx] = in_range ? grad_output[idx] : 0.0f;
        
        // 累积scale的梯度
        if (in_range) {
            float q = roundf(val);
            float grad_s = grad_output[idx] * (input[idx] / (s * s) - (q - z) / s);
            atomicAdd(grad_scale, grad_s);
        }
    }
}
```

**可学习量化参数：**
```cuda
// LSQ（Learned Step Size Quantization）实现
class LearnedStepSizeQuantizer {
private:
    float* d_scale;
    float* d_grad_scale;
    float init_scale;
    int num_bits;
    
public:
    __device__ float compute_scale_gradient(
        float x, float grad_out, float scale) {
        
        float Qn = (1 << (num_bits - 1)) - 1;
        float Qp = -Qn;
        
        // 量化值
        float q = roundf(x / scale);
        q = fmaxf(Qp, fminf(Qn, q));
        
        // scale的梯度
        float grad_scale = 0.0f;
        if (q == Qn || q == Qp) {
            // 饱和区域
            grad_scale = grad_out * q;
        } else {
            // 线性区域
            grad_scale = grad_out * (q - x / scale);
        }
        
        return grad_scale;
    }
};
```

### 23.2.2 梯度的量化处理

梯度量化和裁剪对QAT的稳定性至关重要。

**梯度缩放：**
```cuda
// 自适应梯度缩放
__global__ void adaptive_gradient_scaling_kernel(
    float* gradients,
    const float* quantized_weights,
    const float* full_precision_weights,
    float* scaling_factors,
    int n) {
    
    __shared__ float shared_norm[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // 计算量化误差
    float error = 0.0f;
    if (idx < n) {
        error = fabsf(quantized_weights[idx] - full_precision_weights[idx]);
    }
    
    // 规约求平均误差
    shared_norm[tid] = error;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_norm[tid] += shared_norm[tid + s];
        }
        __syncthreads();
    }
    
    // 根据误差调整梯度
    if (idx < n) {
        float avg_error = shared_norm[0] / blockDim.x;
        float scale = 1.0f / (1.0f + avg_error);
        gradients[idx] *= scale;
    }
}
```

**梯度裁剪与正则化：**
```cuda
// 感知量化的梯度裁剪
__global__ void quantization_aware_gradient_clipping(
    float* gradients,
    const float* weights,
    float clip_value,
    int num_bits,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 根据量化比特数调整裁剪阈值
        float bit_scale = powf(2.0f, 8 - num_bits);
        float adjusted_clip = clip_value * bit_scale;
        
        // 应用梯度裁剪
        gradients[idx] = fmaxf(-adjusted_clip, 
                              fminf(adjusted_clip, gradients[idx]));
        
        // 添加量化正则化项
        float quant_reg = 0.01f * (weights[idx] - roundf(weights[idx]));
        gradients[idx] += quant_reg;
    }
}
```

### 23.2.3 批归一化融合

批归一化（BN）与量化的融合能减少推理时的计算量和内存访问。

**BN折叠公式：**
```
BN(x) = γ * (x - μ) / √(σ² + ε) + β
折叠后：W' = W * γ / √(σ² + ε)
        b' = β - μ * γ / √(σ² + ε)
```

**CUDA实现：**
```cuda
// BN参数折叠到卷积权重
__global__ void fold_bn_to_conv_kernel(
    float* conv_weight,      // [OC, IC, KH, KW]
    float* conv_bias,        
    const float* bn_scale,   // γ
    const float* bn_bias,    // β
    const float* bn_mean,    // μ
    const float* bn_var,     // σ²
    float epsilon,
    int OC, int IC, int K) {
    
    int oc = blockIdx.x;
    int idx = threadIdx.x;
    
    if (oc < OC) {
        float scale = bn_scale[oc] / sqrtf(bn_var[oc] + epsilon);
        float bias = bn_bias[oc] - bn_mean[oc] * scale;
        
        // 更新卷积权重
        for (int i = idx; i < IC * K * K; i += blockDim.x) {
            int weight_idx = oc * IC * K * K + i;
            conv_weight[weight_idx] *= scale;
        }
        
        // 更新偏置
        if (idx == 0) {
            conv_bias[oc] = bias;
        }
    }
}

// 量化感知的BN训练
__global__ void quantization_aware_bn_kernel(
    const float* input,
    float* output,
    float* running_mean,
    float* running_var,
    const float* scale,
    const float* bias,
    float momentum,
    int N, int C, int HW,
    bool training) {
    
    int c = blockIdx.x;
    int tid = threadIdx.x;
    
    if (c < C) {
        // 计算当前批次的均值和方差
        float sum = 0.0f, sum_sq = 0.0f;
        
        for (int n = 0; n < N; n++) {
            for (int i = tid; i < HW; i += blockDim.x) {
                int idx = n * C * HW + c * HW + i;
                float val = input[idx];
                sum += val;
                sum_sq += val * val;
            }
        }
        
        // Warp级规约
        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);
        
        if (tid == 0) {
            float mean = sum / (N * HW);
            float var = sum_sq / (N * HW) - mean * mean;
            
            // 更新running statistics
            if (training) {
                running_mean[c] = momentum * running_mean[c] + 
                                  (1 - momentum) * mean;
                running_var[c] = momentum * running_var[c] + 
                                 (1 - momentum) * var;
            }
            
            // 应用BN变换（考虑量化）
            float inv_std = rsqrtf(var + 1e-5f);
            
            // 量化scale和bias
            float q_scale = fake_quantize(scale[c], 8);
            float q_bias = fake_quantize(bias[c], 8);
            
            for (int n = 0; n < N; n++) {
                for (int i = 0; i < HW; i++) {
                    int idx = n * C * HW + c * HW + i;
                    output[idx] = q_scale * (input[idx] - mean) * inv_std + q_bias;
                }
            }
        }
    }
}
```

### 23.2.4 训练策略与技巧

**渐进式量化：**
```cuda
// 动态调整量化比特数
class ProgressiveQuantization {
private:
    int start_bits = 32;
    int target_bits = 4;
    int current_epoch;
    int total_epochs;
    
public:
    __device__ int get_current_bits() {
        float progress = (float)current_epoch / total_epochs;
        int bits = start_bits - (int)(progress * (start_bits - target_bits));
        return max(target_bits, bits);
    }
    
    __global__ void progressive_quantize_kernel(
        const float* input,
        float* output,
        int n) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int bits = get_current_bits();
            
            // 根据当前比特数量化
            float scale = compute_scale(input, n, bits);
            output[idx] = fake_quantize(input[idx], scale, bits);
        }
    }
};
```

**知识蒸馏辅助：**
```cuda
// 使用教师模型指导量化训练
__global__ void distillation_loss_kernel(
    const float* student_logits,  // 量化模型输出
    const float* teacher_logits,  // 全精度模型输出
    float* loss,
    float temperature,
    int batch_size,
    int num_classes) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;
    
    if (idx < total) {
        int b = idx / num_classes;
        int c = idx % num_classes;
        
        // 计算软标签
        float student_soft = expf(student_logits[idx] / temperature);
        float teacher_soft = expf(teacher_logits[idx] / temperature);
        
        // KL散度损失
        float kl_loss = teacher_soft * logf(teacher_soft / student_soft + 1e-8f);
        
        atomicAdd(&loss[b], kl_loss * temperature * temperature);
    }
}
```

### 23.2.5 混合精度训练集成

将QAT与自动混合精度（AMP）结合，加速训练过程。

```cuda
// 混合精度量化感知训练
template<typename T>
__global__ void mixed_precision_qat_kernel(
    const T* input,           // FP16输入
    T* output,               // FP16输出
    float* master_weights,   // FP32主权重
    int8_t* quantized_weights,
    float* scales,
    float learning_rate,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // FP32计算
        float fp32_input = __half2float(input[idx]);
        float fp32_weight = master_weights[idx];
        
        // 伪量化
        float scale = scales[idx / 128];  // 分组量化
        int8_t q_weight = __float2int_rn(fp32_weight / scale);
        q_weight = max(-128, min(127, q_weight));
        quantized_weights[idx] = q_weight;
        
        // 反量化用于前向传播
        float dq_weight = q_weight * scale;
        
        // 计算输出（FP16）
        output[idx] = __float2half(fp32_input * dq_weight);
        
        // 更新FP32主权重
        float gradient = compute_gradient(...);
        master_weights[idx] -= learning_rate * gradient;
    }
}

// 动态损失缩放
__global__ void dynamic_loss_scaling_kernel(
    float* loss,
    float* scale_factor,
    bool overflow_detected) {
    
    if (threadIdx.x == 0) {
        if (overflow_detected) {
            *scale_factor *= 0.5f;  // 减小scale
        } else {
            *scale_factor *= 2.0f;   // 增大scale
            *scale_factor = fminf(*scale_factor, 65536.0f);
        }
        
        *loss *= *scale_factor;
    }
}
```

## 23.3 动态量化策略

动态量化在推理时根据输入数据动态计算量化参数，特别适合激活值分布变化较大的场景。虽然增加了运行时开销，但能显著提高量化精度，是自动驾驶等安全关键应用的首选方案。

### 23.3.1 动态范围校准

**统计信息收集：**
```cuda
// 高效的min/max统计内核
__global__ void collect_minmax_stats_kernel(
    const float* input,
    float* min_vals,
    float* max_vals,
    int n, int num_blocks) {
    
    extern __shared__ float shared_mem[];
    float* s_min = shared_mem;
    float* s_max = &shared_mem[blockDim.x];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // 初始化局部最值
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    
    // 网格步进遍历
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        float val = input[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }
    
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();
    
    // 块内规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    
    // 写回全局内存
    if (tid == 0) {
        min_vals[blockIdx.x] = s_min[0];
        max_vals[blockIdx.x] = s_max[0];
    }
}

// 百分位数计算（用于异常值处理）
__global__ void percentile_calibration_kernel(
    const float* sorted_input,
    float* scale,
    float* zero_point,
    int n,
    float percentile = 0.999f) {
    
    if (threadIdx.x == 0) {
        int lower_idx = (int)(n * (1.0f - percentile));
        int upper_idx = (int)(n * percentile);
        
        float min_val = sorted_input[lower_idx];
        float max_val = sorted_input[upper_idx];
        
        // 计算量化参数
        *scale = (max_val - min_val) / 255.0f;
        *zero_point = roundf(-min_val / *scale);
    }
}
```

**KL散度校准：**
```cuda
// TensorRT风格的KL散度校准
class KLDivergenceCalibrator {
private:
    static constexpr int NUM_BINS = 2048;
    float* d_histogram;
    float* d_reference_dist;
    
public:
    __global__ void build_histogram_kernel(
        const float* input,
        float* histogram,
        float min_val, float max_val,
        int n) {
        
        __shared__ float shared_hist[NUM_BINS];
        
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        
        // 初始化共享内存
        for (int i = tid; i < NUM_BINS; i += blockDim.x) {
            shared_hist[i] = 0.0f;
        }
        __syncthreads();
        
        // 构建直方图
        if (idx < n) {
            float val = input[idx];
            float bin_width = (max_val - min_val) / NUM_BINS;
            int bin = min((int)((val - min_val) / bin_width), NUM_BINS - 1);
            atomicAdd(&shared_hist[bin], 1.0f);
        }
        __syncthreads();
        
        // 写回全局内存
        for (int i = tid; i < NUM_BINS; i += blockDim.x) {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
    
    __device__ float compute_kl_divergence(
        const float* P, const float* Q, int n) {
        
        float kl = 0.0f;
        for (int i = 0; i < n; i++) {
            if (P[i] > 0 && Q[i] > 0) {
                kl += P[i] * logf(P[i] / Q[i]);
            }
        }
        return kl;
    }
};
```

### 23.3.2 激活值量化

**在线激活量化：**
```cuda
// 融合的激活量化内核
template<typename ActivationType>
__global__ void fused_activation_quantize_kernel(
    const float* input,
    int8_t* output,
    float* running_min,
    float* running_max,
    float momentum,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算激活值
    float activated = 0.0f;
    if (idx < n) {
        if constexpr (std::is_same_v<ActivationType, ReLU>) {
            activated = fmaxf(0.0f, input[idx]);
        } else if constexpr (std::is_same_v<ActivationType, SiLU>) {
            activated = input[idx] / (1.0f + expf(-input[idx]));
        }
    }
    
    // 更新统计信息（使用原子操作）
    atomicMin(running_min, activated);
    atomicMax(running_max, activated);
    
    // 动态量化
    __syncthreads();  // 确保统计更新完成
    
    if (idx < n) {
        float scale = (*running_max - *running_min) / 255.0f;
        float zero_point = -*running_min / scale;
        
        int quantized = roundf(activated / scale + zero_point);
        output[idx] = max(-128, min(127, quantized));
    }
}
```

**层级动态量化：**
```cuda
// 逐层动态量化管理器
class LayerWiseDynamicQuantizer {
private:
    struct LayerStats {
        float min_val;
        float max_val;
        float scale;
        int zero_point;
        int calibration_batches;
    };
    
    LayerStats* d_layer_stats;
    int num_layers;
    
public:
    __global__ void update_layer_stats_kernel(
        const float* activation,
        LayerStats* stats,
        int layer_id,
        int n) {
        
        // 使用CUB进行高效规约
        typedef cub::BlockReduce<float, 256> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        
        float val = (idx < n) ? activation[idx] : 0.0f;
        
        float block_min = BlockReduce(temp_storage).Reduce(val, cub::Min());
        float block_max = BlockReduce(temp_storage).Reduce(val, cub::Max());
        
        if (tid == 0) {
            // 更新层统计信息
            atomicMin(&stats[layer_id].min_val, block_min);
            atomicMax(&stats[layer_id].max_val, block_max);
            
            // 重新计算量化参数
            float range = stats[layer_id].max_val - stats[layer_id].min_val;
            stats[layer_id].scale = range / 255.0f;
            stats[layer_id].zero_point = 
                roundf(-stats[layer_id].min_val / stats[layer_id].scale);
        }
    }
};
```

### 23.3.3 自适应量化

**输入感知量化：**
```cuda
// 基于输入特征的自适应量化
__global__ void input_aware_quantization_kernel(
    const float* input,
    int8_t* output,
    float* scales,      // 每个样本的scale
    int* zero_points,   // 每个样本的zero_point
    int batch_size,
    int feature_size) {
    
    int bid = blockIdx.x;  // batch index
    int tid = threadIdx.x;
    
    if (bid < batch_size) {
        // 计算每个样本的统计信息
        float local_min = FLT_MAX;
        float local_max = -FLT_MAX;
        
        for (int i = tid; i < feature_size; i += blockDim.x) {
            int idx = bid * feature_size + i;
            float val = input[idx];
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
        
        // Warp级规约
        local_min = warpReduceMin(local_min);
        local_max = warpReduceMax(local_max);
        
        // 计算自适应量化参数
        if (tid % 32 == 0) {
            float range = local_max - local_min;
            
            // 根据范围选择量化策略
            if (range < 0.1f) {
                // 小范围：使用更高精度
                scales[bid] = range / 127.0f;
                zero_points[bid] = 0;  // 对称量化
            } else {
                // 大范围：标准量化
                scales[bid] = range / 255.0f;
                zero_points[bid] = roundf(-local_min / scales[bid]);
            }
        }
        
        __syncthreads();
        
        // 应用量化
        for (int i = tid; i < feature_size; i += blockDim.x) {
            int idx = bid * feature_size + i;
            float val = input[idx];
            int quantized = roundf(val / scales[bid] + zero_points[bid]);
            output[idx] = max(-128, min(127, quantized));
        }
    }
}
```

### 23.3.4 运行时优化

**量化参数缓存：**
```cuda
// LRU缓存管理量化参数
template<int CACHE_SIZE>
class QuantizationCache {
private:
    struct CacheEntry {
        uint64_t hash;
        float scale;
        int zero_point;
        int timestamp;
    };
    
    CacheEntry cache[CACHE_SIZE];
    int current_time;
    
public:
    __device__ bool lookup(uint64_t hash, float& scale, int& zero_point) {
        for (int i = 0; i < CACHE_SIZE; i++) {
            if (cache[i].hash == hash) {
                cache[i].timestamp = atomicAdd(&current_time, 1);
                scale = cache[i].scale;
                zero_point = cache[i].zero_point;
                return true;
            }
        }
        return false;
    }
    
    __device__ void insert(uint64_t hash, float scale, int zero_point) {
        // 找到最老的条目
        int lru_idx = 0;
        int min_time = cache[0].timestamp;
        
        for (int i = 1; i < CACHE_SIZE; i++) {
            if (cache[i].timestamp < min_time) {
                min_time = cache[i].timestamp;
                lru_idx = i;
            }
        }
        
        // 更新缓存
        cache[lru_idx].hash = hash;
        cache[lru_idx].scale = scale;
        cache[lru_idx].zero_point = zero_point;
        cache[lru_idx].timestamp = atomicAdd(&current_time, 1);
    }
};
```

**流水线量化：**
```cuda
// 异步量化流水线
class AsyncQuantizationPipeline {
private:
    cudaStream_t compute_stream;
    cudaStream_t quantize_stream;
    cudaEvent_t* events;
    
public:
    void process_batch(
        const float* input,
        int8_t* output,
        int batch_size) {
        
        for (int i = 0; i < batch_size; i++) {
            // 在compute_stream中计算统计信息
            collect_stats<<<..., compute_stream>>>(
                input + i * feature_size, ...);
            
            // 记录事件
            cudaEventRecord(events[i], compute_stream);
            
            // 在quantize_stream中执行量化
            cudaStreamWaitEvent(quantize_stream, events[i]);
            apply_quantization<<<..., quantize_stream>>>(
                input + i * feature_size,
                output + i * feature_size, ...);
        }
    }
};
```

### 23.3.5 性能与精度权衡

**混合位宽策略：**
```cuda
// 层级混合精度量化
__global__ void mixed_bitwidth_quantization_kernel(
    const float* input,
    void* output,  // 可能是int4/int8/fp16
    const int* layer_bitwidths,
    int layer_id,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bits = layer_bitwidths[layer_id];
    
    if (idx < n) {
        float val = input[idx];
        
        switch(bits) {
            case 4: {
                // INT4量化
                int8_t* out4 = (int8_t*)output;
                int quantized = roundf(val * 7.0f);  // [-8, 7]
                quantized = max(-8, min(7, quantized));
                
                // 打包两个INT4到一个INT8
                if (idx % 2 == 0) {
                    out4[idx/2] = quantized & 0x0F;
                } else {
                    out4[idx/2] |= (quantized & 0x0F) << 4;
                }
                break;
            }
            case 8: {
                // INT8量化
                int8_t* out8 = (int8_t*)output;
                out8[idx] = __float2int_rn(val * 127.0f);
                break;
            }
            case 16: {
                // FP16
                half* out16 = (half*)output;
                out16[idx] = __float2half(val);
                break;
            }
        }
    }
}

// 敏感度分析
__global__ void sensitivity_analysis_kernel(
    const float* original_output,
    const float* quantized_output,
    float* sensitivity_scores,
    int num_layers,
    int layer_size) {
    
    int layer = blockIdx.x;
    int tid = threadIdx.x;
    
    if (layer < num_layers) {
        float error_sum = 0.0f;
        
        for (int i = tid; i < layer_size; i += blockDim.x) {
            int idx = layer * layer_size + i;
            float diff = original_output[idx] - quantized_output[idx];
            error_sum += diff * diff;
        }
        
        // 规约计算MSE
        error_sum = blockReduceSum(error_sum);
        
        if (tid == 0) {
            sensitivity_scores[layer] = sqrtf(error_sum / layer_size);
        }
    }
}
```

## 23.4 自定义量化算子

### 23.4.1 量化GEMM实现

### 23.4.2 量化卷积优化

### 23.4.3 特殊激活函数处理

### 23.4.4 融合算子设计

### 23.4.5 TensorRT集成

## 23.5 案例：超低比特推理

### 23.5.1 场景需求分析

### 23.5.2 模型量化流程

### 23.5.3 CUDA内核实现

### 23.5.4 性能优化

### 23.5.5 精度恢复技术

## 本章小结

## 练习题

## 常见陷阱与错误 (Gotchas)

## 最佳实践检查清单
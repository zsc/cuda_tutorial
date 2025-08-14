# 第13章：实时语义分割与实例分割

本章深入探讨自动驾驶场景中语义分割和实例分割的GPU加速技术。我们将从高分辨率图像的分块处理开始，逐步深入到深度可分离卷积、NMS并行化、Mask生成等关键技术，最终实现能够满足自动驾驶实时性要求的分割系统。通过本章学习，你将掌握如何在保证精度的前提下，将分割网络的推理速度提升10倍以上。

## 13.1 高分辨率图像的分块处理

自动驾驶系统通常需要处理4K甚至8K分辨率的图像，直接将整张图像送入神经网络会导致显存溢出或推理速度过慢。分块处理（Tiling）是解决这一问题的关键技术，它将大图像分割成多个小块，分别处理后再拼接结果。

### 13.1.1 分块策略设计

分块策略的核心是在显存占用、计算效率和分割精度之间找到平衡点。设输入图像尺寸为 H×W，分块大小为 h×w，重叠区域为 overlap。

**基础分块公式：**
```
tiles_y = ceil((H - overlap) / (h - overlap))
tiles_x = ceil((W - overlap) / (w - overlap))
```

关键考虑因素：

1. **感受野匹配**：分块大小必须大于网络的有效感受野，否则会丢失上下文信息。对于典型的语义分割网络，感受野可能达到数百像素。

2. **内存对齐**：分块尺寸应该是32的倍数，以确保内存访问效率：
```
aligned_h = ((h + 31) / 32) * 32
aligned_w = ((w + 31) / 32) * 32
```

3. **批处理效率**：多个分块可以组成batch同时处理，提高GPU利用率。理想的batch size通常是SM数量的倍数。

**自适应分块算法：**

```
分块决策流程：
1. 计算网络感受野 RF
2. 设置最小分块 min_tile = RF * 1.5
3. 根据可用显存计算最大分块 max_tile
4. 在[min_tile, max_tile]范围内选择32的倍数
5. 计算重叠区域 overlap = RF * 0.25
```

对于动态输入，可以预先建立查找表，根据输入尺寸快速确定最优分块参数。

### 13.1.2 重叠区域处理与边界融合

重叠区域是确保分块边界处分割连续性的关键。处理策略包括：

**1. 重叠区域计算**

重叠宽度应考虑：
- 网络下采样倍数（通常为8或16）
- 空洞卷积的扩张率
- 目标物体的典型尺寸

```
实际重叠计算：
overlap = max(
    downsample_rate * 4,     // 下采样补偿
    dilation_rate * kernel,   // 空洞卷积补偿  
    min_object_size / 4       // 物体尺寸补偿
)
```

**2. 边界融合策略**

常用的融合方法：

a) **线性混合**：在重叠区域使用距离加权
```
weight(x) = (x - left_boundary) / overlap_width
result = (1 - weight) * left_tile + weight * right_tile
```

b) **高斯混合**：使用高斯函数平滑过渡
```
weight(x) = exp(-(x - center)² / (2σ²))
其中 σ = overlap_width / 6
```

c) **最大置信度选择**：选择概率最高的预测
```
result = argmax(softmax(logits_left), softmax(logits_right))
```

**3. GPU实现优化**

边界融合的并行化实现需要考虑内存访问模式：

```cuda
__global__ void fuseBoundaries(
    float* output,      // 输出特征图
    float* tiles,       // 所有分块结果
    int* tile_coords,   // 分块坐标
    float* weights,     // 融合权重
    int H, int W, int C // 尺寸参数
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;
    
    int y = idx / W;
    int x = idx % W;
    
    // 查找覆盖该位置的所有分块
    int tile_count = 0;
    float accumulated[MAX_CHANNELS] = {0};
    float weight_sum = 0;
    
    for (int t = 0; t < num_tiles; t++) {
        if (isInTile(x, y, tile_coords[t])) {
            float w = computeWeight(x, y, tile_coords[t]);
            for (int c = 0; c < C; c++) {
                accumulated[c] += tiles[t][...] * w;
            }
            weight_sum += w;
        }
    }
    
    // 归一化输出
    for (int c = 0; c < C; c++) {
        output[...] = accumulated[c] / weight_sum;
    }
}
```

### 13.1.3 动态分块与负载均衡

静态分块可能导致GPU利用率不均，特别是当图像内容分布不均匀时。动态分块策略可以根据内容复杂度调整分块大小。

**1. 内容复杂度评估**

使用快速预处理评估每个区域的复杂度：

```cuda
__global__ void computeComplexity(
    uint8_t* image,
    float* complexity_map,
    int H, int W
) {
    __shared__ float local_grad[BLOCK_SIZE][BLOCK_SIZE];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 计算局部梯度
    float gx = abs(image[y][x+1] - image[y][x-1]);
    float gy = abs(image[y+1][x] - image[y-1][x]);
    local_grad[threadIdx.y][threadIdx.x] = sqrt(gx*gx + gy*gy);
    
    __syncthreads();
    
    // 归约计算块复杂度
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float sum = 0;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                sum += local_grad[i][j];
            }
        }
        complexity_map[blockIdx.y][blockIdx.x] = sum;
    }
}
```

**2. 自适应分块生成**

根据复杂度图动态调整分块：

```
算法：自适应分块
1. 将图像划分为初始网格
2. 计算每个网格的复杂度
3. 对高复杂度区域使用小分块（提高精度）
4. 对低复杂度区域使用大分块（提高速度）
5. 合并相邻的相似复杂度区域
```

**3. 负载均衡调度**

使用工作窃取（Work Stealing）模式平衡负载：

```cuda
struct TileQueue {
    int* tiles;
    int* head;
    int* tail;
    
    __device__ int steal() {
        int old_head = atomicAdd(head, 1);
        if (old_head < *tail) {
            return tiles[old_head];
        }
        return -1;  // 队列为空
    }
};

__global__ void processTilesDynamic(
    TileQueue queue,
    float* input,
    float* output
) {
    while (true) {
        int tile_id = queue.steal();
        if (tile_id < 0) break;
        
        // 处理分块
        processSingleTile(tile_id, input, output);
    }
}
```

### 13.1.4 流水线并行优化

通过流水线技术可以隐藏数据传输延迟，实现计算与传输的重叠。

**1. 三缓冲流水线**

使用三个缓冲区轮转，实现上传、计算、下载的完全重叠：

```cuda
// 流水线结构
struct Pipeline {
    cudaStream_t upload_stream;
    cudaStream_t compute_stream;
    cudaStream_t download_stream;
    
    float* host_buffers[3];
    float* device_buffers[3];
    float* output_buffers[3];
    
    cudaEvent_t events[3];
};

void pipelineProcess(Pipeline& pipe, TileList& tiles) {
    for (int i = 0; i < tiles.count + 2; i++) {
        int stage = i % 3;
        
        // 上传阶段
        if (i < tiles.count) {
            cudaMemcpyAsync(
                pipe.device_buffers[stage],
                pipe.host_buffers[stage],
                tile_size,
                cudaMemcpyHostToDevice,
                pipe.upload_stream
            );
            cudaEventRecord(pipe.events[stage], pipe.upload_stream);
        }
        
        // 计算阶段
        if (i >= 1 && i < tiles.count + 1) {
            int prev_stage = (stage + 2) % 3;
            cudaStreamWaitEvent(pipe.compute_stream, pipe.events[prev_stage]);
            
            segmentationKernel<<<blocks, threads, 0, pipe.compute_stream>>>(
                pipe.device_buffers[prev_stage],
                pipe.output_buffers[prev_stage]
            );
        }
        
        // 下载阶段
        if (i >= 2) {
            int prev_prev_stage = (stage + 1) % 3;
            cudaMemcpyAsync(
                pipe.host_buffers[prev_prev_stage],
                pipe.output_buffers[prev_prev_stage],
                output_size,
                cudaMemcpyDeviceToHost,
                pipe.download_stream
            );
        }
    }
}
```

**2. 动态批处理优化**

根据分块大小动态组合批次，最大化GPU利用率：

```
批处理策略：
1. 将分块按大小排序
2. 贪心组合相似大小的分块
3. 确保每批的总内存不超过限制
4. 优先组合空间相邻的分块（提高缓存局部性）
```

**3. 预取与缓存优化**

利用L2缓存预取下一个分块的数据：

```cuda
__global__ void segmentWithPrefetch(
    float* current_tile,
    float* next_tile,
    float* output
) {
    // 预取下一个分块到L2
    if (threadIdx.x == 0) {
        for (int i = 0; i < prefetch_size; i += 128) {
            __prefetch_global_L2(next_tile + i);
        }
    }
    
    // 处理当前分块
    processCurrentTile(current_tile, output);
}
```

## 13.2 深度可分离卷积优化

深度可分离卷积（Depthwise Separable Convolution）是轻量级网络的核心组件，它将标准卷积分解为深度卷积（Depthwise）和逐点卷积（Pointwise）两步，大幅减少计算量。在MobileNet、EfficientNet等网络中广泛应用。

### 13.2.1 深度卷积的内存访问模式

深度卷积的特点是每个输入通道独立计算，不存在跨通道的累加操作。这带来了独特的优化机会和挑战。

**1. 内存布局选择**

不同的内存布局对深度卷积性能影响显著：

```
NCHW布局：适合通道数较少的情况
- 优点：每个通道数据连续，利于向量化
- 缺点：跨通道访问需要大步长

NHWC布局：适合通道数较多的情况  
- 优点：空间位置的数据连续，利于合并访问
- 缺点：通道维度不连续，需要重排

NC/32HW32布局：针对Tensor Core优化
- 将通道按32分组，实现对齐访问
- 适合使用WMMA指令的场景
```

**2. 优化的深度卷积实现**

```cuda
template<int KERNEL_SIZE, int CHANNELS_PER_BLOCK>
__global__ void depthwiseConv2d(
    const float* __restrict__ input,   // NHWC layout
    const float* __restrict__ filter,  // [K, K, C]
    float* __restrict__ output,
    int H, int W, int C,
    int stride, int pad
) {
    // 每个线程块处理一组通道
    const int c_start = blockIdx.z * CHANNELS_PER_BLOCK;
    const int c_end = min(c_start + CHANNELS_PER_BLOCK, C);
    
    // 共享内存缓存
    extern __shared__ float smem[];
    float* s_input = smem;
    float* s_filter = smem + (BLOCK_SIZE + KERNEL_SIZE - 1) * 
                             (BLOCK_SIZE + KERNEL_SIZE - 1) * CHANNELS_PER_BLOCK;
    
    // 加载滤波器到共享内存
    if (threadIdx.x < KERNEL_SIZE * KERNEL_SIZE) {
        for (int c = c_start; c < c_end; c++) {
            int local_c = c - c_start;
            s_filter[threadIdx.x * CHANNELS_PER_BLOCK + local_c] = 
                filter[threadIdx.x * C + c];
        }
    }
    
    // 输出位置
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x >= W || out_y >= H) return;
    
    // 协作加载输入到共享内存
    const int tile_x = threadIdx.x;
    const int tile_y = threadIdx.y;
    
    for (int c = c_start; c < c_end; c++) {
        int local_c = c - c_start;
        
        // 加载包含halo的输入块
        for (int dy = -pad; dy < BLOCK_SIZE + pad; dy += blockDim.y) {
            for (int dx = -pad; dx < BLOCK_SIZE + pad; dx += blockDim.x) {
                int in_y = blockIdx.y * BLOCK_SIZE + dy + tile_y;
                int in_x = blockIdx.x * BLOCK_SIZE + dx + tile_x;
                
                float val = 0.0f;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    val = input[(in_y * W + in_x) * C + c];
                }
                
                s_input[((dy + pad) * (BLOCK_SIZE + 2*pad) + (dx + pad)) * 
                       CHANNELS_PER_BLOCK + local_c] = val;
            }
        }
    }
    
    __syncthreads();
    
    // 计算卷积
    float results[CHANNELS_PER_BLOCK];
    #pragma unroll
    for (int c = 0; c < CHANNELS_PER_BLOCK; c++) {
        results[c] = 0.0f;
    }
    
    #pragma unroll
    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
        #pragma unroll
        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
            int in_y = tile_y * stride + ky;
            int in_x = tile_x * stride + kx;
            
            #pragma unroll
            for (int c = 0; c < CHANNELS_PER_BLOCK; c++) {
                float input_val = s_input[(in_y * (BLOCK_SIZE + 2*pad) + in_x) * 
                                         CHANNELS_PER_BLOCK + c];
                float filter_val = s_filter[(ky * KERNEL_SIZE + kx) * 
                                          CHANNELS_PER_BLOCK + c];
                results[c] += input_val * filter_val;
            }
        }
    }
    
    // 写回结果
    for (int c = 0; c < CHANNELS_PER_BLOCK && c_start + c < C; c++) {
        output[(out_y * W + out_x) * C + c_start + c] = results[c];
    }
}
```

**3. 向量化访存优化**

使用向量化指令一次加载多个元素：

```cuda
// 使用float4向量化加载
__global__ void depthwiseConvVectorized(
    const float4* input,  // 假设C是4的倍数
    const float4* filter,
    float4* output
) {
    // 一次处理4个通道
    float4 acc = make_float4(0, 0, 0, 0);
    
    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
            float4 in = input[...];
            float4 flt = filter[...];
            
            // 向量化乘加
            acc.x += in.x * flt.x;
            acc.y += in.y * flt.y;
            acc.z += in.z * flt.z;
            acc.w += in.w * flt.w;
        }
    }
    
    output[...] = acc;
}
```

### 13.2.2 逐点卷积的GEMM优化

逐点卷积（1×1卷积）本质上是矩阵乘法，可以利用高度优化的GEMM实现。

**1. Im2col转换**

将输入特征图转换为矩阵形式：

```cuda
__global__ void im2colPointwise(
    const float* input,  // [N, H, W, C_in]
    float* col_buffer,   // [N*H*W, C_in]
    int N, int H, int W, int C_in
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C_in;
    
    if (idx < total) {
        // 直接复制，因为1x1卷积不需要展开
        col_buffer[idx] = input[idx];
    }
}
```

**2. 使用CUTLASS优化的1×1卷积**

```cuda
#include <cutlass/gemm/device/gemm.h>

using Gemm = cutlass::gemm::device::Gemm<
    float,                          // 元素类型
    cutlass::layout::RowMajor,     // A矩阵布局
    float,                          
    cutlass::layout::ColumnMajor,  // B矩阵布局
    float,
    cutlass::layout::RowMajor,     // C矩阵布局
    float,                          // 累加器类型
    cutlass::arch::OpClassTensorOp, // 使用Tensor Core
    cutlass::arch::Sm80             // 架构
>;

void pointwiseConvCutlass(
    const float* input,   // [N*H*W, C_in]
    const float* weight,  // [C_in, C_out]
    float* output,        // [N*H*W, C_out]
    int M, int N, int K
) {
    Gemm gemm_op;
    
    cutlass::Status status = gemm_op({
        {M, N, K},          // 问题尺寸
        {input, K},         // A矩阵
        {weight, N},        // B矩阵
        {output, N},        // C矩阵
        {output, N},        // D矩阵（输出）
        {1.0f, 0.0f}        // alpha, beta
    });
}
```

**3. 寄存器级优化**

```cuda
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void pointwiseConvTiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 寄存器缓存
    float reg_A[TILE_M];
    float reg_B[TILE_N];
    float reg_C[TILE_M][TILE_N] = {0};
    
    // 共享内存双缓冲
    __shared__ float smem_A[2][TILE_K][TILE_M];
    __shared__ float smem_B[2][TILE_K][TILE_N];
    
    int tid = threadIdx.x;
    int buffer = 0;
    
    // 主循环
    for (int k = 0; k < K; k += TILE_K) {
        // 协作加载到共享内存
        loadTileToShared(A, smem_A[buffer], ...);
        loadTileToShared(B, smem_B[buffer], ...);
        
        __syncthreads();
        
        // 计算
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            // 加载到寄存器
            #pragma unroll
            for (int m = 0; m < TILE_M; m++) {
                reg_A[m] = smem_A[buffer][kk][m];
            }
            
            #pragma unroll
            for (int n = 0; n < TILE_N; n++) {
                reg_B[n] = smem_B[buffer][kk][n];
            }
            
            // 外积累加
            #pragma unroll
            for (int m = 0; m < TILE_M; m++) {
                #pragma unroll
                for (int n = 0; n < TILE_N; n++) {
                    reg_C[m][n] += reg_A[m] * reg_B[n];
                }
            }
        }
        
        buffer ^= 1;  // 切换缓冲区
    }
    
    // 写回结果
    storeTileFromRegisters(reg_C, C, ...);
}
```

### 13.2.3 融合算子设计

将深度卷积、逐点卷积和激活函数融合，减少内存访问。

**1. 完整的融合实现**

```cuda
template<typename ActivationOp>
__global__ void fusedDepthwiseSeparable(
    const float* input,
    const float* dw_filter,
    const float* pw_weight,
    const float* bias,
    float* output,
    ActivationOp activation,
    int H, int W, int C_in, int C_out
) {
    // 分配寄存器
    float dw_result[MAX_CHANNELS_PER_THREAD];
    float pw_result = 0.0f;
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idx = out_idx / C_out;
    int c_out = out_idx % C_out;
    
    if (pixel_idx >= H * W) return;
    
    int y = pixel_idx / W;
    int x = pixel_idx % W;
    
    // 步骤1：深度卷积（在寄存器中）
    #pragma unroll
    for (int c_in = 0; c_in < C_in; c_in++) {
        dw_result[c_in] = 0.0f;
        
        #pragma unroll
        for (int ky = 0; ky < 3; ky++) {
            #pragma unroll
            for (int kx = 0; kx < 3; kx++) {
                int in_y = y + ky - 1;
                int in_x = x + kx - 1;
                
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    float in_val = input[(in_y * W + in_x) * C_in + c_in];
                    float flt_val = dw_filter[(ky * 3 + kx) * C_in + c_in];
                    dw_result[c_in] += in_val * flt_val;
                }
            }
        }
    }
    
    // 步骤2：逐点卷积（直接累加）
    #pragma unroll
    for (int c_in = 0; c_in < C_in; c_in++) {
        pw_result += dw_result[c_in] * pw_weight[c_out * C_in + c_in];
    }
    
    // 步骤3：偏置和激活
    pw_result += bias[c_out];
    pw_result = activation(pw_result);
    
    // 写回结果
    output[(y * W + x) * C_out + c_out] = pw_result;
}

// ReLU6激活函数
struct ReLU6 {
    __device__ float operator()(float x) {
        return fminf(fmaxf(x, 0.0f), 6.0f);
    }
};
```

**2. 使用Tensor Core的融合实现**

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void fusedDepthwiseSeparableTensorCore(
    const half* input,
    const half* filters,
    half* output
) {
    // 声明fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 深度卷积部分（简化示例）
    // ... 深度卷积计算 ...
    
    // 逐点卷积使用Tensor Core
    wmma::load_matrix_sync(a_frag, input, 16);
    wmma::load_matrix_sync(b_frag, filters, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // 存储结果
    wmma::store_matrix_sync(output, c_frag, 16, wmma::mem_row_major);
}
```

### 13.2.4 Winograd算法应用

Winograd算法通过减少乘法次数来加速小卷积核的计算，特别适合3×3卷积。

**1. Winograd F(2,3)实现**

对于3×3卷积，输出2×2块：

```cuda
__device__ void winograd_f2x3_transform_input(
    const float input[4][4],
    float transformed[4][4]
) {
    // BT * input * B
    // BT = [1   0  -1   0]
    //      [0   1   1   0]
    //      [0  -1   1   0]
    //      [0   1   0  -1]
    
    float temp[4][4];
    
    // 行变换
    for (int i = 0; i < 4; i++) {
        temp[i][0] = input[i][0] - input[i][2];
        temp[i][1] = input[i][1] + input[i][2];
        temp[i][2] = -input[i][1] + input[i][2];
        temp[i][3] = input[i][1] - input[i][3];
    }
    
    // 列变换
    for (int j = 0; j < 4; j++) {
        transformed[0][j] = temp[0][j] - temp[2][j];
        transformed[1][j] = temp[1][j] + temp[2][j];
        transformed[2][j] = -temp[1][j] + temp[2][j];
        transformed[3][j] = temp[1][j] - temp[3][j];
    }
}

__device__ void winograd_f2x3_transform_filter(
    const float filter[3][3],
    float transformed[4][4]
) {
    // G * filter * GT
    // G = [1    0    0]
    //     [0.5  0.5  0.5]
    //     [0.5 -0.5  0.5]
    //     [0    0    1]
    
    float temp[4][3];
    
    // 行变换
    temp[0][0] = filter[0][0];
    temp[0][1] = filter[0][1];
    temp[0][2] = filter[0][2];
    
    for (int j = 0; j < 3; j++) {
        temp[1][j] = 0.5f * (filter[0][j] + filter[1][j] + filter[2][j]);
        temp[2][j] = 0.5f * (filter[0][j] - filter[1][j] + filter[2][j]);
    }
    
    temp[3][0] = filter[2][0];
    temp[3][1] = filter[2][1];
    temp[3][2] = filter[2][2];
    
    // 列变换
    for (int i = 0; i < 4; i++) {
        transformed[i][0] = temp[i][0];
        transformed[i][1] = 0.5f * (temp[i][0] + temp[i][1] + temp[i][2]);
        transformed[i][2] = 0.5f * (temp[i][0] - temp[i][1] + temp[i][2]);
        transformed[i][3] = temp[i][2];
    }
}

__device__ void winograd_f2x3_output_transform(
    const float product[4][4],
    float output[2][2]
) {
    // AT * product * A
    // AT = [1  1  1  0]
    //      [0  1 -1 -1]
    
    float temp[2][4];
    
    // 行变换
    for (int j = 0; j < 4; j++) {
        temp[0][j] = product[0][j] + product[1][j] + product[2][j];
        temp[1][j] = product[1][j] - product[2][j] - product[3][j];
    }
    
    // 列变换
    for (int i = 0; i < 2; i++) {
        output[i][0] = temp[i][0] + temp[i][1] + temp[i][2];
        output[i][1] = temp[i][1] - temp[i][2] - temp[i][3];
    }
}
```

**2. 批量Winograd实现**

```cuda
template<int BATCH_SIZE>
__global__ void winogradBatchedConv(
    const float* input,
    const float* filter,
    float* output,
    int H, int W, int C
) {
    __shared__ float s_transformed_input[BATCH_SIZE][4][4];
    __shared__ float s_transformed_filter[4][4];
    
    int batch_idx = threadIdx.x;
    int tile_idx = blockIdx.x;
    
    // 变换滤波器（只需一次）
    if (batch_idx == 0) {
        float filter_tile[3][3];
        loadFilter(filter, filter_tile);
        winograd_f2x3_transform_filter(filter_tile, s_transformed_filter);
    }
    
    // 每个线程处理一个批次元素
    if (batch_idx < BATCH_SIZE) {
        float input_tile[4][4];
        loadInputTile(input, input_tile, tile_idx, batch_idx);
        winograd_f2x3_transform_input(input_tile, s_transformed_input[batch_idx]);
    }
    
    __syncthreads();
    
    // 逐元素乘法
    float product[4][4] = {0};
    if (batch_idx < BATCH_SIZE) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                product[i][j] = s_transformed_input[batch_idx][i][j] * 
                               s_transformed_filter[i][j];
            }
        }
        
        // 输出变换
        float output_tile[2][2];
        winograd_f2x3_output_transform(product, output_tile);
        
        // 写回结果
        storeOutputTile(output, output_tile, tile_idx, batch_idx);
    }
}
```

## 13.3 NMS（非极大值抑制）并行化

非极大值抑制是目标检测中的关键后处理步骤，用于去除重叠的检测框。传统NMS算法的串行特性使其成为GPU加速的瓶颈。

### 13.3.1 传统NMS的并行化瓶颈

传统NMS算法的伪代码：
```
1. 按置信度排序所有检测框
2. 选择置信度最高的框
3. 删除与该框IoU > threshold的所有框
4. 重复步骤2-3直到没有框剩余
```

**并行化挑战：**

1. **数据依赖性**：每个框的保留/删除依赖于之前的决策
2. **动态工作负载**：每轮迭代的计算量不同
3. **内存访问模式**：频繁的随机访问和删除操作

**IoU计算的并行化**

首先优化IoU计算，这是NMS中最频繁的操作：

```cuda
__device__ float computeIoU(
    const float4 box1,  // x1, y1, x2, y2
    const float4 box2
) {
    float x1 = fmaxf(box1.x, box2.x);
    float y1 = fmaxf(box1.y, box2.y);
    float x2 = fminf(box1.z, box2.z);
    float y2 = fminf(box1.w, box2.w);
    
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    
    float area1 = (box1.z - box1.x) * (box1.w - box1.y);
    float area2 = (box2.z - box2.x) * (box2.w - box2.y);
    
    return intersection / (area1 + area2 - intersection + 1e-6f);
}

// 批量IoU计算
__global__ void computeIoUMatrix(
    const float4* boxes,
    float* iou_matrix,
    int num_boxes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < num_boxes && idy < num_boxes) {
        float iou = (idx == idy) ? 1.0f : computeIoU(boxes[idx], boxes[idy]);
        iou_matrix[idy * num_boxes + idx] = iou;
    }
}
```

### 13.3.2 分块NMS算法

分块NMS通过将检测框分组来实现并行处理：

**1. 空间分块策略**

```cuda
struct SpatialBlock {
    int x_bin, y_bin;
    int start_idx, end_idx;
    float4* boxes;
    float* scores;
};

__global__ void spatialBlockNMS(
    SpatialBlock* blocks,
    int* keep_mask,
    float iou_threshold,
    int num_blocks
) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    SpatialBlock block = blocks[block_id];
    
    // 块内NMS
    for (int i = block.start_idx; i < block.end_idx; i++) {
        if (keep_mask[i] == 0) continue;
        
        // 并行检查当前框与其他框
        for (int j = i + 1; j < block.end_idx; j++) {
            if (keep_mask[j] == 0) continue;
            
            float iou = computeIoU(block.boxes[i], block.boxes[j]);
            if (iou > iou_threshold) {
                // 保留分数更高的框
                if (block.scores[i] > block.scores[j]) {
                    atomicExch(&keep_mask[j], 0);
                } else {
                    atomicExch(&keep_mask[i], 0);
                    break;
                }
            }
        }
    }
}
```

**2. 类别并行NMS**

对不同类别的检测框并行处理：

```cuda
__global__ void perClassNMS(
    float4* boxes,          // [N, 4]
    float* scores,          // [N, num_classes]
    int* class_ids,         // [N]
    int* keep_mask,         // [N]
    float iou_threshold,
    int num_boxes,
    int num_classes
) {
    int class_id = blockIdx.x;
    if (class_id >= num_classes) return;
    
    // 使用共享内存存储该类别的框索引
    extern __shared__ int class_indices[];
    int class_count = 0;
    
    // 收集该类别的框
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_boxes; i++) {
            if (class_ids[i] == class_id) {
                class_indices[class_count++] = i;
            }
        }
    }
    __syncthreads();
    
    // 并行处理类内NMS
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    for (int i = tid; i < class_count; i += stride) {
        int idx_i = class_indices[i];
        if (keep_mask[idx_i] == 0) continue;
        
        for (int j = i + 1; j < class_count; j++) {
            int idx_j = class_indices[j];
            if (keep_mask[idx_j] == 0) continue;
            
            float iou = computeIoU(boxes[idx_i], boxes[idx_j]);
            if (iou > iou_threshold) {
                float score_i = scores[idx_i * num_classes + class_id];
                float score_j = scores[idx_j * num_classes + class_id];
                
                if (score_i > score_j) {
                    atomicExch(&keep_mask[idx_j], 0);
                }
            }
        }
    }
}
```

### 13.3.3 GPU友好的NMS变体

**1. Batched NMS**

批量处理多个图像的NMS：

```cuda
__global__ void batchedNMS(
    float4* boxes,          // [batch_size, max_boxes, 4]
    float* scores,          // [batch_size, max_boxes]
    int* valid_counts,      // [batch_size]
    int* output_indices,    // [batch_size, max_output]
    float iou_threshold,
    int batch_size,
    int max_boxes,
    int max_output
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int offset = batch_idx * max_boxes;
    int valid_count = valid_counts[batch_idx];
    
    // 使用位掩码跟踪保留的框
    __shared__ unsigned int keep_mask[MAX_BOXES / 32];
    
    // 初始化掩码
    if (threadIdx.x < (valid_count + 31) / 32) {
        keep_mask[threadIdx.x] = 0xFFFFFFFF;
    }
    __syncthreads();
    
    // 按分数排序后的索引（预先计算）
    for (int i = 0; i < valid_count; i++) {
        int idx_i = offset + i;
        
        // 检查是否已被抑制
        int mask_idx = i / 32;
        int bit_idx = i % 32;
        if ((keep_mask[mask_idx] & (1 << bit_idx)) == 0) continue;
        
        // 并行检查后续框
        int tid = threadIdx.x;
        for (int j = i + 1 + tid; j < valid_count; j += blockDim.x) {
            int mask_j = j / 32;
            int bit_j = j % 32;
            if ((keep_mask[mask_j] & (1 << bit_j)) == 0) continue;
            
            float iou = computeIoU(boxes[idx_i], boxes[offset + j]);
            if (iou > iou_threshold) {
                // 原子清除位
                atomicAnd(&keep_mask[mask_j], ~(1 << bit_j));
            }
        }
        __syncthreads();
    }
    
    // 收集保留的框索引
    if (threadIdx.x == 0) {
        int out_idx = 0;
        for (int i = 0; i < valid_count && out_idx < max_output; i++) {
            int mask_idx = i / 32;
            int bit_idx = i % 32;
            if (keep_mask[mask_idx] & (1 << bit_idx)) {
                output_indices[batch_idx * max_output + out_idx++] = i;
            }
        }
    }
}
```

**2. DIoU-NMS（Distance IoU NMS）**

考虑中心点距离的NMS变体：

```cuda
__device__ float computeDIoU(
    const float4 box1,
    const float4 box2
) {
    float iou = computeIoU(box1, box2);
    
    // 计算中心点距离
    float cx1 = (box1.x + box1.z) * 0.5f;
    float cy1 = (box1.y + box1.w) * 0.5f;
    float cx2 = (box2.x + box2.z) * 0.5f;
    float cy2 = (box2.y + box2.w) * 0.5f;
    
    float dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
    
    // 计算包围框对角线长度
    float ex1 = fminf(box1.x, box2.x);
    float ey1 = fminf(box1.y, box2.y);
    float ex2 = fmaxf(box1.z, box2.z);
    float ey2 = fmaxf(box1.w, box2.w);
    
    float diag_sq = (ex2 - ex1) * (ex2 - ex1) + (ey2 - ey1) * (ey2 - ey1);
    
    return iou - dist_sq / (diag_sq + 1e-6f);
}
```

### 13.3.4 Soft-NMS的高效实现

Soft-NMS通过衰减而非删除重叠框的分数，保留更多信息：

```cuda
__global__ void softNMS(
    float4* boxes,
    float* scores,
    int* indices,
    float sigma,
    float score_threshold,
    int num_boxes
) {
    extern __shared__ float s_scores[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // 加载分数到共享内存
    if (gid < num_boxes) {
        s_scores[tid] = scores[gid];
    }
    __syncthreads();
    
    // 对每个框进行Soft-NMS
    for (int i = 0; i < num_boxes; i++) {
        // 找到当前最高分数的框
        __shared__ int max_idx;
        __shared__ float max_score;
        
        if (tid == 0) {
            max_idx = -1;
            max_score = score_threshold;
            for (int j = 0; j < num_boxes; j++) {
                if (s_scores[j] > max_score) {
                    max_score = s_scores[j];
                    max_idx = j;
                }
            }
        }
        __syncthreads();
        
        if (max_idx == -1) break;
        
        // 保存当前框
        if (tid == 0) {
            indices[i] = max_idx;
        }
        
        // 并行更新其他框的分数
        float4 max_box = boxes[max_idx];
        
        for (int j = tid; j < num_boxes; j += blockDim.x) {
            if (j != max_idx && s_scores[j] > 0) {
                float iou = computeIoU(max_box, boxes[j]);
                
                // Gaussian衰减
                float weight = expf(-(iou * iou) / sigma);
                s_scores[j] *= weight;
                
                // 线性衰减（可选）
                // s_scores[j] *= (iou < iou_threshold) ? 1.0f : (1.0f - iou);
            }
        }
        
        // 将已选择框的分数设为0
        if (tid == max_idx % blockDim.x) {
            s_scores[max_idx % blockDim.x] = 0;
        }
        __syncthreads();
    }
}
```

**优化的Soft-NMS实现**

```cuda
template<int TILE_SIZE>
__global__ void optimizedSoftNMS(
    float4* boxes,
    float* scores,
    int* keep_flags,
    float sigma,
    float iou_threshold,
    int num_boxes
) {
    // 使用寄存器缓存
    float4 reg_boxes[TILE_SIZE];
    float reg_scores[TILE_SIZE];
    
    int tile_start = blockIdx.x * TILE_SIZE;
    int tile_end = min(tile_start + TILE_SIZE, num_boxes);
    
    // 加载tile到寄存器
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++) {
        int idx = tile_start + i;
        if (idx < num_boxes) {
            reg_boxes[i] = boxes[idx];
            reg_scores[i] = scores[idx];
        }
    }
    
    // 处理每个框
    for (int i = 0; i < num_boxes; i++) {
        if (keep_flags[i] == 0) continue;
        
        float4 current_box = boxes[i];
        float current_score = scores[i];
        
        // 与tile中的框计算IoU
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; j++) {
            int idx = tile_start + j;
            if (idx != i && idx < num_boxes) {
                float iou = computeIoU(current_box, reg_boxes[j]);
                
                if (iou > iou_threshold) {
                    // 应用Soft-NMS衰减
                    float decay = expf(-(iou * iou) / (2.0f * sigma * sigma));
                    reg_scores[j] *= decay;
                    
                    // 如果分数过低，标记为删除
                    if (reg_scores[j] < 0.01f) {
                        atomicExch(&keep_flags[idx], 0);
                    }
                }
            }
        }
    }
    
    // 写回更新的分数
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++) {
        int idx = tile_start + i;
        if (idx < num_boxes) {
            scores[idx] = reg_scores[i];
        }
    }
}
```

## 13.4 Mask生成与后处理

实例分割需要为每个检测到的目标生成精确的掩码，这涉及RoIAlign、上采样和边缘细化等技术。

### 13.4.1 RoIAlign的优化实现

RoIAlign通过双线性插值避免了RoIPool的量化误差：

```cuda
__global__ void roiAlign(
    const float* features,    // [C, H, W]
    const float* rois,        // [N, 5] (batch_idx, x1, y1, x2, y2)
    float* output,            // [N, C, pool_h, pool_w]
    int channels,
    int height,
    int width,
    int num_rois,
    int pool_h,
    int pool_w,
    float spatial_scale,
    int sampling_ratio
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int pw = idx % pool_w;
    int ph = (idx / pool_w) % pool_h;
    int c = (idx / pool_w / pool_h) % channels;
    int n = idx / pool_w / pool_h / channels;
    
    if (n >= num_rois) return;
    
    // RoI坐标
    const float* roi = rois + n * 5;
    int batch_idx = roi[0];
    float x1 = roi[1] * spatial_scale;
    float y1 = roi[2] * spatial_scale;
    float x2 = roi[3] * spatial_scale;
    float y2 = roi[4] * spatial_scale;
    
    float roi_w = x2 - x1;
    float roi_h = y2 - y1;
    
    float bin_w = roi_w / pool_w;
    float bin_h = roi_h / pool_h;
    
    // 采样点数
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceilf(bin_h);
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceilf(bin_w);
    
    float sum = 0.0f;
    int count = roi_bin_grid_h * roi_bin_grid_w;
    
    // 双线性插值采样
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        float y = y1 + ph * bin_h + (iy + 0.5f) * bin_h / roi_bin_grid_h;
        
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            float x = x1 + pw * bin_w + (ix + 0.5f) * bin_w / roi_bin_grid_w;
            
            // 双线性插值
            int x_low = floorf(x);
            int y_low = floorf(y);
            int x_high = x_low + 1;
            int y_high = y_low + 1;
            
            float lx = x - x_low;
            float ly = y - y_low;
            float hx = 1.0f - lx;
            float hy = 1.0f - ly;
            
            // 边界检查
            x_low = max(0, min(x_low, width - 1));
            x_high = max(0, min(x_high, width - 1));
            y_low = max(0, min(y_low, height - 1));
            y_high = max(0, min(y_high, height - 1));
            
            // 插值计算
            float v1 = features[(c * height + y_low) * width + x_low];
            float v2 = features[(c * height + y_low) * width + x_high];
            float v3 = features[(c * height + y_high) * width + x_low];
            float v4 = features[(c * height + y_high) * width + x_high];
            
            sum += hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
        }
    }
    
    output[idx] = sum / count;
}
```

### 13.4.2 Mask上采样策略

**1. 双线性上采样**

```cuda
__global__ void bilinearUpsample(
    const float* input,   // [N, C, H, W]
    float* output,        // [N, C, H*scale, W*scale]
    int N, int C, int H, int W,
    int scale
) {
    int out_w = W * scale;
    int out_h = H * scale;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_h * out_w;
    
    if (idx >= total) return;
    
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / out_w / out_h) % C;
    int n = idx / out_w / out_h / C;
    
    // 计算在原图中的坐标
    float h_ratio = (float)H / out_h;
    float w_ratio = (float)W / out_w;
    
    float src_h = oh * h_ratio;
    float src_w = ow * w_ratio;
    
    int h0 = floorf(src_h);
    int w0 = floorf(src_w);
    int h1 = min(h0 + 1, H - 1);
    int w1 = min(w0 + 1, W - 1);
    
    float dh = src_h - h0;
    float dw = src_w - w0;
    
    // 双线性插值
    float v00 = input[((n * C + c) * H + h0) * W + w0];
    float v01 = input[((n * C + c) * H + h0) * W + w1];
    float v10 = input[((n * C + c) * H + h1) * W + w0];
    float v11 = input[((n * C + c) * H + h1) * W + w1];
    
    float value = (1 - dh) * (1 - dw) * v00 +
                  (1 - dh) * dw * v01 +
                  dh * (1 - dw) * v10 +
                  dh * dw * v11;
    
    output[idx] = value;
}
```

**2. 转置卷积上采样**

```cuda
__global__ void transposeConvUpsample(
    const float* input,
    const float* weights,  // [C_out, C_in, K, K]
    float* output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int kernel_size,
    int stride,
    int padding
) {
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int oc = (idx / W_out / H_out) % C_out;
    int n = idx / W_out / H_out / C_out;
    
    if (n >= N) return;
    
    float sum = 0.0f;
    
    // 计算哪些输入位置会贡献到当前输出位置
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int ih = (oh + padding - kh);
            int iw = (ow + padding - kw);
            
            if (ih % stride == 0 && iw % stride == 0) {
                ih /= stride;
                iw /= stride;
                
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                    for (int ic = 0; ic < C_in; ic++) {
                        float in_val = input[((n * C_in + ic) * H_in + ih) * W_in + iw];
                        float w_val = weights[((oc * C_in + ic) * kernel_size + kh) * kernel_size + kw];
                        sum += in_val * w_val;
                    }
                }
            }
        }
    }
    
    output[idx] = sum;
}
```

### 13.4.3 边缘细化算法

**1. 条件随机场（CRF）后处理**

```cuda
__global__ void denseCRF(
    float* masks,        // [N, H, W]
    float* refined,      // [N, H, W]
    float* features,     // [N, C, H, W] 图像特征
    int N, int H, int W,
    float theta_alpha,   // 位置权重
    float theta_beta,    // 颜色权重
    float theta_gamma,   // 平滑权重
    int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel = idx % (H * W);
    int n = idx / (H * W);
    
    if (n >= N) return;
    
    int y = pixel / W;
    int x = pixel % W;
    
    // 迭代优化
    for (int iter = 0; iter < iterations; iter++) {
        float sum = 0.0f;
        float norm = 0.0f;
        
        // 计算邻域势能
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                
                int ny = y + dy;
                int nx = x + dx;
                
                if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                    // 位置项
                    float dist_sq = dx * dx + dy * dy;
                    float pos_term = expf(-dist_sq / (2 * theta_alpha * theta_alpha));
                    
                    // 颜色相似性
                    float color_diff = 0.0f;
                    for (int c = 0; c < 3; c++) {
                        float diff = features[((n * 3 + c) * H + y) * W + x] -
                                   features[((n * 3 + c) * H + ny) * W + nx];
                        color_diff += diff * diff;
                    }
                    float color_term = expf(-color_diff / (2 * theta_beta * theta_beta));
                    
                    float weight = pos_term * color_term;
                    sum += weight * masks[(n * H + ny) * W + nx];
                    norm += weight;
                }
            }
        }
        
        refined[(n * H + y) * W + x] = theta_gamma * masks[(n * H + y) * W + x] +
                                       (1 - theta_gamma) * sum / (norm + 1e-6f);
    }
}
```

**2. 形态学操作优化**

```cuda
__global__ void morphologicalRefinement(
    float* masks,
    float* refined,
    int H, int W,
    int kernel_size,
    bool is_erosion
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= W || y >= H) return;
    
    float result = is_erosion ? 1.0f : 0.0f;
    
    for (int ky = -kernel_size/2; ky <= kernel_size/2; ky++) {
        for (int kx = -kernel_size/2; kx <= kernel_size/2; kx++) {
            int ny = y + ky;
            int nx = x + kx;
            
            if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                float val = masks[ny * W + nx];
                if (is_erosion) {
                    result = fminf(result, val);
                } else {
                    result = fmaxf(result, val);
                }
            }
        }
    }
    
    refined[y * W + x] = result;
}
```

### 13.4.4 实例融合与去重

```cuda
__global__ void instanceMerging(
    float* masks,         // [N, H, W]
    int* instance_ids,    // [N]
    float* merged_masks,  // [M, H, W]
    int* merge_map,       // [N] -> M
    float overlap_threshold,
    int N, int H, int W
) {
    int instance_i = blockIdx.x;
    int instance_j = blockIdx.y;
    
    if (instance_i >= N || instance_j >= N || instance_i >= instance_j) return;
    
    // 计算masks重叠度
    __shared__ float intersection;
    __shared__ float union_area;
    
    if (threadIdx.x == 0) {
        intersection = 0.0f;
        union_area = 0.0f;
    }
    __syncthreads();
    
    // 并行计算IoU
    for (int idx = threadIdx.x; idx < H * W; idx += blockDim.x) {
        float mask_i = masks[instance_i * H * W + idx];
        float mask_j = masks[instance_j * H * W + idx];
        
        atomicAdd(&intersection, mask_i * mask_j);
        atomicAdd(&union_area, fmaxf(mask_i, mask_j));
    }
    __syncthreads();
    
    // 判断是否需要合并
    if (threadIdx.x == 0) {
        float iou = intersection / (union_area + 1e-6f);
        if (iou > overlap_threshold) {
            // 合并到ID较小的实例
            atomicMin(&merge_map[instance_j], merge_map[instance_i]);
        }
    }
}
```

## 13.5 多尺度特征融合策略

多尺度特征融合是提升分割精度的关键技术，通过结合不同尺度的特征信息来改善小目标检测和边界定位。

### 13.5.1 FPN的内存优化

特征金字塔网络（FPN）的内存优化实现：

```cuda
template<int NUM_LEVELS>
__global__ void fpnFusion(
    float** pyramid_features,  // 各层特征
    float* fused_output,
    int* feature_dims,         // 各层尺寸
    float* level_weights,       // 层权重
    int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = idx % C;
    int spatial_idx = idx / C;
    
    if (spatial_idx >= H * W) return;
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    #pragma unroll
    for (int level = 0; level < NUM_LEVELS; level++) {
        int level_h = feature_dims[level * 2];
        int level_w = feature_dims[level * 2 + 1];
        
        // 计算对应位置
        float scale_h = (float)level_h / H;
        float scale_w = (float)level_w / W;
        
        int y = spatial_idx / W;
        int x = spatial_idx % W;
        
        float src_y = y * scale_h;
        float src_x = x * scale_w;
        
        // 双线性插值采样
        int y0 = floorf(src_y);
        int x0 = floorf(src_x);
        int y1 = min(y0 + 1, level_h - 1);
        int x1 = min(x0 + 1, level_w - 1);
        
        float dy = src_y - y0;
        float dx = src_x - x0;
        
        float* level_feat = pyramid_features[level];
        
        float v00 = level_feat[(c * level_h + y0) * level_w + x0];
        float v01 = level_feat[(c * level_h + y0) * level_w + x1];
        float v10 = level_feat[(c * level_h + y1) * level_w + x0];
        float v11 = level_feat[(c * level_h + y1) * level_w + x1];
        
        float interpolated = (1-dy) * (1-dx) * v00 +
                            (1-dy) * dx * v01 +
                            dy * (1-dx) * v10 +
                            dy * dx * v11;
        
        sum += interpolated * level_weights[level];
        weight_sum += level_weights[level];
    }
    
    fused_output[idx] = sum / (weight_sum + 1e-6f);
}
```

### 13.5.2 跨尺度特征对齐

```cuda
__global__ void crossScaleAlignment(
    float* low_res_features,   // 低分辨率特征
    float* high_res_features,  // 高分辨率特征
    float* aligned_features,
    float* attention_weights,
    int H_low, int W_low,
    int H_high, int W_high,
    int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int c = idx % C;
    int pixel = idx / C;
    int y = pixel / W_high;
    int x = pixel % W_high;
    
    if (y >= H_high || x >= W_high) return;
    
    // 计算低分辨率对应位置
    float y_low = (float)y * H_low / H_high;
    float x_low = (float)x * W_low / W_high;
    
    // 获取低分辨率特征（插值）
    int y0 = floorf(y_low);
    int x0 = floorf(x_low);
    
    float low_feat = low_res_features[(c * H_low + y0) * W_low + x0];
    float high_feat = high_res_features[(c * H_high + y) * W_high + x];
    
    // 计算注意力权重
    float attention = attention_weights[(y * W_high + x) * C + c];
    
    // 加权融合
    aligned_features[idx] = attention * high_feat + (1 - attention) * low_feat;
}
```

### 13.5.3 自适应特征聚合

```cuda
__global__ void adaptiveFeatureAggregation(
    float** multi_scale_features,
    float* aggregated,
    float* scale_weights,  // 可学习的尺度权重
    int num_scales,
    int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W * C) return;
    
    int c = idx % C;
    int spatial = idx / C;
    
    // Softmax归一化权重
    float weight_sum = 0.0f;
    float normalized_weights[8];  // 假设最多8个尺度
    
    for (int s = 0; s < num_scales; s++) {
        float w = expf(scale_weights[s * C + c]);
        normalized_weights[s] = w;
        weight_sum += w;
    }
    
    // 加权聚合
    float result = 0.0f;
    for (int s = 0; s < num_scales; s++) {
        normalized_weights[s] /= weight_sum;
        result += multi_scale_features[s][idx] * normalized_weights[s];
    }
    
    aggregated[idx] = result;
}
```

### 13.5.4 轻量级融合模块设计

```cuda
__global__ void lightweightFusion(
    float* feat1,
    float* feat2,
    float* fused,
    float* gate_weights,  // 1x1卷积参数
    int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;
    
    // 共享内存缓存
    extern __shared__ float s_gate[];
    
    // 计算门控权重
    if (threadIdx.x < C) {
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int i = threadIdx.x; i < H * W; i += blockDim.x) {
            sum1 += feat1[i * C + threadIdx.x];
            sum2 += feat2[i * C + threadIdx.x];
        }
        
        // 全局平均池化
        s_gate[threadIdx.x] = gate_weights[threadIdx.x] * sum1 / (H * W);
        s_gate[C + threadIdx.x] = gate_weights[C + threadIdx.x] * sum2 / (H * W);
    }
    __syncthreads();
    
    // 应用门控融合
    for (int c = 0; c < C; c++) {
        float g1 = 1.0f / (1.0f + expf(-s_gate[c]));  // sigmoid
        float g2 = 1.0f - g1;
        
        fused[idx * C + c] = g1 * feat1[idx * C + c] + g2 * feat2[idx * C + c];
    }
}
```

## 本章小结

本章深入探讨了实时语义分割与实例分割的GPU加速技术。我们从高分辨率图像的分块处理策略开始，详细分析了如何通过合理的分块、重叠区域处理和流水线优化来处理超大分辨率图像。在深度可分离卷积优化部分，我们展示了如何通过内存布局优化、向量化访存、算子融合和Winograd算法将轻量级网络的推理速度提升数倍。

NMS并行化是目标检测的关键瓶颈，我们介绍了分块NMS、Soft-NMS等GPU友好的变体实现。在Mask生成部分，详细讲解了RoIAlign、上采样和边缘细化的优化技术。最后，多尺度特征融合策略展示了如何高效地结合不同尺度的特征信息。

**关键优化点总结：**
- 分块处理中的重叠区域融合：weight(x) = exp(-(x-center)²/(2σ²))
- 深度卷积的通道并行：每个通道独立计算，适合SIMT架构
- Winograd F(2,3)变换：将9次乘法减少到4次
- Soft-NMS衰减公式：score *= exp(-(IoU²)/σ)
- FPN多尺度融合：自适应权重聚合

## 练习题

### 基础题

1. **分块策略设计**
   实现一个自适应分块算法，根据GPU显存大小和网络感受野自动确定最优分块参数。
   
   *提示：考虑显存占用 = batch_size × channels × tile_height × tile_width × 4 bytes*
   
   <details>
   <summary>参考答案</summary>
   
   最优分块大小应满足：(1)大于1.5倍感受野；(2)是32的倍数；(3)总显存占用小于可用显存的80%。重叠区域应为感受野的25%。可以预先建立查找表避免运行时计算。
   </details>

2. **IoU批量计算优化**
   设计一个高效的批量IoU计算核函数，处理N×N的IoU矩阵，要求利用共享内存和向量化访问。
   
   *提示：使用float4一次加载4个坐标值*
   
   <details>
   <summary>参考答案</summary>
   
   将boxes加载到共享内存，使用tiling技术分块计算IoU矩阵。每个线程块处理32×32的子矩阵，利用float4向量化加载坐标。对角线元素直接设为1.0避免计算。
   </details>

3. **深度卷积内存布局**
   比较NCHW和NHWC布局对深度卷积性能的影响，实现两种布局的kernel并测试。
   
   *提示：考虑内存合并访问和缓存局部性*
   
   <details>
   <summary>参考答案</summary>
   
   NHWC布局在通道数较多时性能更好，因为相邻线程访问连续内存。NCHW在通道数较少时更优，因为每个通道的空间数据连续。建议根据C/HW比值动态选择布局。
   </details>

### 挑战题

4. **混合精度Winograd实现**
   实现支持FP16输入和FP32累加的Winograd F(4,3)算法，要求使用Tensor Core加速矩阵乘法部分。
   
   *提示：变换矩阵可以预计算并存储为常量*
   
   <details>
   <summary>参考答案</summary>
   
   F(4,3)需要6×6的变换tile，36个元素的逐点乘法可以用WMMA指令加速。输入/输出变换使用FP32保证精度，中间计算使用FP16。注意处理边界情况的padding。
   </details>

5. **并行化的Mask R-CNN后处理**
   设计完整的Mask R-CNN后处理流水线，包括NMS、RoIAlign和mask上采样，要求三个阶段并行执行。
   
   *提示：使用CUDA Stream和Event同步*
   
   <details>
   <summary>参考答案</summary>
   
   创建3个stream分别处理NMS、RoIAlign和上采样。NMS完成后通过event通知RoIAlign开始，同时继续处理下一批。使用环形缓冲区管理中间结果。关键是平衡各阶段的计算量。
   </details>

6. **自适应特征融合网络**
   实现一个可学习的多尺度特征融合模块，支持任意数量的输入尺度，权重通过反向传播更新。
   
   *提示：使用atomicAdd实现梯度累加*
   
   <details>
   <summary>参考答案</summary>
   
   前向传播计算softmax归一化的尺度权重，反向传播需要计算权重梯度和特征梯度。使用共享内存缓存中间结果，原子操作累加梯度。注意数值稳定性，softmax计算时减去最大值。
   </details>

7. **实时全景分割系统**
   设计一个完整的全景分割推理系统，结合语义分割和实例分割，要求在2080Ti上达到30FPS@1080p。
   
   *提示：考虑模型量化和算子融合*
   
   <details>
   <summary>参考答案</summary>
   
   使用INT8量化backbone，FP16计算分割头。将多个卷积层融合为一个kernel减少内存访问。使用TensorRT优化推理图。关键路径上避免CPU-GPU同步。预分配所有缓冲区避免动态分配。
   </details>

8. **增量式实例分割**
   实现支持视频流的增量式实例分割，利用时序信息加速处理，要求支持目标跟踪和ID保持。
   
   *提示：使用光流估计和特征传播*
   
   <details>
   <summary>参考答案</summary>
   
   关键帧执行完整分割，其他帧通过光流传播mask。使用匈牙利算法匹配实例ID。特征图可以在时序上重用，只更新变化区域。维护目标特征库用于重识别。考虑遮挡和新目标出现的处理。
   </details>

## 常见陷阱与错误

1. **分块边界处理不当**
   - 错误：直接丢弃重叠区域
   - 正确：使用加权融合或置信度选择
   
2. **NMS中的竞态条件**
   - 错误：多个线程同时修改keep_mask
   - 正确：使用原子操作或分阶段处理

3. **Winograd数值不稳定**
   - 错误：直接使用大的变换矩阵
   - 正确：使用改进的变换矩阵减少数值误差

4. **RoIAlign边界越界**
   - 错误：不检查采样点是否在图像内
   - 正确：clamp坐标到有效范围

5. **深度卷积的bank conflict**
   - 错误：多个线程访问同一bank的共享内存
   - 正确：使用padding或重排访问模式

6. **特征融合的内存爆炸**
   - 错误：同时保存所有尺度的特征图
   - 正确：流式处理，及时释放不需要的特征

## 最佳实践检查清单

### 设计阶段
- [ ] 分析输入图像分辨率分布，设计分块策略
- [ ] 评估网络感受野，确定重叠区域大小
- [ ] 选择合适的内存布局（NCHW vs NHWC）
- [ ] 确定精度要求（FP32/FP16/INT8）
- [ ] 设计批处理策略最大化GPU利用率

### 实现阶段
- [ ] 使用向量化指令（float4）加速内存访问
- [ ] 实现算子融合减少kernel启动开销
- [ ] 使用共享内存缓存频繁访问的数据
- [ ] 避免warp divergence，特别是在边界处理
- [ ] 使用流水线隐藏内存延迟

### 优化阶段
- [ ] Profile确定性能瓶颈（计算/内存/同步）
- [ ] 调整block和grid配置优化占用率
- [ ] 实现多流并发隐藏延迟
- [ ] 考虑使用Tensor Core加速矩阵运算
- [ ] 优化内存访问模式避免bank conflict

### 验证阶段
- [ ] 测试不同分辨率输入的正确性
- [ ] 验证边界情况（空图、单目标、密集目标）
- [ ] 检查数值稳定性（梯度爆炸/消失）
- [ ] 对比不同优化版本的精度损失
- [ ] 压力测试检查内存泄漏
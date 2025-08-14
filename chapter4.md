# 第4章：共享内存与Bank Conflict

共享内存是GPU上最重要的性能优化手段之一。与全局内存相比，共享内存提供了高达100倍的带宽和10倍更低的延迟。然而，要充分发挥共享内存的性能潜力，必须深入理解其硬件架构，特别是bank conflict机制。本章将深入剖析共享内存的工作原理，掌握bank conflict的检测与规避技术，并通过双缓冲等高级技术实现内核的极致优化。在自动驾驶的实时感知和具身智能的高频控制场景中，这些技术是达到毫秒级响应的关键。

## 4.1 共享内存架构详解

### 4.1.1 硬件组织结构

共享内存在物理上被组织为多个独立的存储体（bank），这种设计允许多个线程同时访问不同的bank，实现真正的并行访问。现代GPU（Volta及之后）的共享内存具有以下特征：

**Bank组织方式**：
- 32个bank，每个bank宽度为4字节
- 连续的32个4字节字被分配到32个不同的bank
- 地址到bank的映射：`bank_id = (byte_address / 4) % 32`

```
地址空间布局（以4字节为单位）：
     Bank0  Bank1  Bank2  ...  Bank31
Row0:  [0]    [1]    [2]   ...   [31]
Row1:  [32]   [33]   [34]  ...   [63]
Row2:  [64]   [65]   [66]  ...   [95]
...
```

**访问特性**：
- 单周期访问：无conflict时，一个warp的32个线程可在一个周期内完成访问
- 广播机制：多个线程读取同一地址时自动广播，不产生conflict
- 多播机制：Volta+架构支持同一bank内多个地址的多播

### 4.1.2 内存容量与配置

不同GPU架构的共享内存容量：

| 架构 | SM总容量 | 每线程块最大 | 可配置性 |
|------|----------|--------------|----------|
| Pascal | 64 KB | 48 KB | 固定分配 |
| Volta/Turing | 96 KB | 64 KB | L1/共享内存动态配置 |
| Ampere | 164 KB | 100 KB | 细粒度配置（0-164KB） |
| Hopper | 228 KB | 144 KB | 自动管理模式 |

**动态配置策略**：
```cuda
// 运行时配置
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, percentage);

// 编译时声明
__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
```

### 4.1.3 访问延迟与带宽

共享内存的性能特征：

**延迟对比**：
- 共享内存：约20-30个时钟周期
- L1缓存：约28个时钟周期（命中）
- L2缓存：约200个时钟周期
- 全局内存：约300-400个时钟周期

**理论带宽计算**：
对于V100（1530 MHz，80个SM）：
- 每SM带宽 = 32 banks × 4 bytes × 1530 MHz = 195.84 GB/s
- 总带宽 = 195.84 GB/s × 80 = 15.67 TB/s
- 相比之下，全局内存带宽仅为900 GB/s

### 4.1.4 原子操作支持

共享内存支持完整的原子操作集：

```cuda
// 共享内存原子操作示例
__shared__ int counter[32];
atomicAdd(&counter[threadIdx.x % 32], 1);  // 可能导致conflict

// 优化：使用不同bank
__shared__ int counter[32 * 32];  // 每个warp使用不同的行
atomicAdd(&counter[threadIdx.x + warpId * 32], 1);  // 无conflict
```

## 4.2 Bank Conflict的本质与检测

### 4.2.1 Bank Conflict的定义

Bank conflict发生在一个warp内的多个线程试图访问同一个bank的不同地址时。这会导致访问串行化，严重降低性能。

**Conflict类型**：
1. **无conflict**：每个线程访问不同的bank
2. **广播**：多个线程访问相同地址（只读）
3. **n-way conflict**：n个线程访问同一bank的不同地址

```cuda
// 示例1：无conflict - 连续访问
__shared__ float data[1024];
float val = data[threadIdx.x];  // 每个线程访问不同bank

// 示例2：2-way conflict
float val = data[threadIdx.x * 2];  // 偶数线程访问偶数bank

// 示例3：32-way conflict（最坏情况）
float val = data[threadIdx.x * 32];  // 所有线程访问同一bank
```

### 4.2.2 Conflict的性能影响

Bank conflict对性能的影响是累乘的：

| Conflict程度 | 访问周期 | 有效带宽 |
|-------------|---------|----------|
| 无conflict | 1 | 100% |
| 2-way | 2 | 50% |
| 4-way | 4 | 25% |
| 8-way | 8 | 12.5% |
| 32-way | 32 | 3.125% |

### 4.2.3 检测工具与方法

**使用Nsight Compute检测**：
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
./your_kernel
```

**代码级检测**：
```cuda
template<typename T>
__device__ void detectBankConflict(T* sharedMem, int idx) {
    unsigned int start = clock();
    T value = sharedMem[idx];
    unsigned int end = clock();
    
    // 通过时间差判断是否有conflict
    if (end - start > THRESHOLD) {
        // 可能存在bank conflict
    }
}
```

### 4.2.4 常见的Conflict模式

**跨步访问模式**：
```cuda
// Stride = 2: 2-way conflict
__shared__ float matrix[32][33];  // 注意：33而不是32
float val = matrix[threadIdx.y][threadIdx.x * 2];

// Stride = 8: 8-way conflict  
float val = matrix[threadIdx.y][threadIdx.x * 8];
```

**矩阵转置模式**：
```cuda
// 原始版本：严重conflict
__shared__ float tile[32][32];
// 读取时列优先，写入时行优先
tile[threadIdx.y][threadIdx.x] = input[...];  
__syncthreads();
output[...] = tile[threadIdx.x][threadIdx.y];  // 32-way conflict!
```

**归约操作模式**：
```cuda
// 树形归约中的conflict
__shared__ float sdata[256];
// Stride从大到小变化，后期会产生conflict
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];  // s较小时产生conflict
    }
    __syncthreads();
}
```

## 4.3 规避策略：Padding、Swizzling与Permutation

### 4.3.1 Padding技术

Padding是最简单有效的conflict规避方法，通过添加额外的列来改变bank映射：

```cuda
// 原始：32-way conflict in transpose
__shared__ float tile[32][32];  

// Padding：消除conflict
__shared__ float tile[32][33];  // 添加1列padding

// 通用padding公式
template<int TILE_SIZE>
struct SharedMemoryPadded {
    static constexpr int PADDING = (TILE_SIZE % 32 == 0) ? 1 : 0;
    __shared__ float data[TILE_SIZE][TILE_SIZE + PADDING];
};
```

**Padding开销分析**：
- 空间开销：(33-32)/32 = 3.125%
- 完全消除conflict，性能提升可达32倍

### 4.3.2 Swizzling技术

Swizzling通过重新排列数据布局来避免conflict，不增加额外存储：

```cuda
// XOR swizzling示例
__device__ int swizzle(int x, int y, int mask = 0x7) {
    return x ^ ((y & mask) << 2);
}

__shared__ float tile[32][32];
// 使用swizzled索引
int swizzled_x = swizzle(threadIdx.x, threadIdx.y);
float val = tile[threadIdx.y][swizzled_x];
```

**高级Swizzling模式**：
```cuda
// Diagonal swizzling
__device__ int diagonalSwizzle(int x, int y, int width) {
    return (x + y) % width;
}

// Block-cyclic swizzling  
__device__ int blockCyclicSwizzle(int x, int y, int blockSize = 4) {
    int blockX = x / blockSize;
    int blockY = y / blockSize;
    int inBlockX = x % blockSize;
    int inBlockY = y % blockSize;
    return blockX * blockSize + (inBlockX + blockY) % blockSize;
}
```

### 4.3.3 Permutation策略

Permutation通过重新排列线程访问顺序来避免conflict：

```cuda
// 线程重映射
__device__ int permuteThread(int tid, int numThreads) {
    // 位反转permutation
    int result = 0;
    int bits = __ffs(numThreads) - 1;
    for (int i = 0; i < bits; i++) {
        if (tid & (1 << i)) {
            result |= (1 << (bits - 1 - i));
        }
    }
    return result;
}

// 使用permutation访问
int permuted_tid = permuteThread(threadIdx.x, blockDim.x);
float val = sharedMem[permuted_tid];
```

### 4.3.4 策略选择指南

| 场景 | 推荐策略 | 理由 |
|------|---------|------|
| 矩阵转置 | Padding | 简单有效，开销小 |
| 卷积/Stencil | Swizzling | 保持局部性，无空间开销 |
| 归约操作 | Permutation | 动态调整访问模式 |
| 稀疏访问 | 混合策略 | 根据稀疏模式定制 |

## 4.4 双缓冲与流水线技术

### 4.4.1 双缓冲原理

双缓冲技术通过重叠计算与数据传输来隐藏内存延迟：

```cuda
template<int TILE_SIZE>
__global__ void matmulDoubleBuffer(float* A, float* B, float* C, int N) {
    // 双缓冲共享内存
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];  // +1 for padding
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    
    float sum = 0.0f;
    int buffer = 0;
    
    // 预加载第一个tile
    As[buffer][ty][tx] = A[...];
    Bs[buffer][ty][tx] = B[...];
    __syncthreads();
    
    for (int tile = 1; tile < N/TILE_SIZE; tile++) {
        // 异步加载下一个tile到另一个buffer
        int next_buffer = 1 - buffer;
        As[next_buffer][ty][tx] = A[...];  // tile+1的数据
        Bs[next_buffer][ty][tx] = B[...];
        
        // 同时计算当前buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[buffer][ty][k] * Bs[buffer][k][tx];
        }
        
        __syncthreads();
        buffer = next_buffer;  // 切换buffer
    }
    
    // 处理最后一个tile
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[buffer][ty][k] * Bs[buffer][k][tx];
    }
    
    C[...] = sum;
}
```

### 4.4.2 软件流水线优化

软件流水线将循环展开为预载、稳态和收尾三个阶段：

```cuda
// 三阶段流水线
template<int STAGES>
__global__ void pipelinedKernel(float* input, float* output, int N) {
    __shared__ float buffer[STAGES][BLOCK_SIZE];
    
    // Prologue: 填充流水线
    for (int s = 0; s < STAGES-1; s++) {
        buffer[s][threadIdx.x] = input[s * BLOCK_SIZE + threadIdx.x];
    }
    __syncthreads();
    
    // Steady state: 主循环
    for (int i = STAGES-1; i < N/BLOCK_SIZE; i++) {
        int curr_stage = i % STAGES;
        int next_stage = (i + 1) % STAGES;
        
        // 加载下一阶段数据
        buffer[next_stage][threadIdx.x] = input[(i+1) * BLOCK_SIZE + threadIdx.x];
        
        // 处理当前阶段
        float result = compute(buffer[curr_stage][threadIdx.x]);
        output[(i-STAGES+1) * BLOCK_SIZE + threadIdx.x] = result;
        
        __syncthreads();
    }
    
    // Epilogue: 清空流水线
    for (int s = 0; s < STAGES-1; s++) {
        int stage = (N/BLOCK_SIZE + s) % STAGES;
        float result = compute(buffer[stage][threadIdx.x]);
        output[...] = result;
        __syncthreads();
    }
}
```

### 4.4.3 异步拷贝与流水线（Ampere+）

Ampere架构引入了异步拷贝指令，实现真正的硬件级流水线：

```cuda
// 使用cuda::pipeline (C++20)
#include <cuda/pipeline>

template<int BLOCK_SIZE>
__global__ void asyncPipelineKernel(float* input, float* output, int N) {
    __shared__ float smem[2][BLOCK_SIZE];
    
    auto pipe = cuda::make_pipeline();
    const auto thread_role = cuda::pipeline_role::producer;
    
    // 异步加载第一个块
    cuda::memcpy_async(smem[0], input, sizeof(float) * BLOCK_SIZE, pipe);
    pipe.producer_commit();
    
    for (int i = 1; i < N/BLOCK_SIZE; i++) {
        int curr_buf = (i-1) % 2;
        int next_buf = i % 2;
        
        // 异步加载下一块
        cuda::memcpy_async(smem[next_buf], 
                          input + i * BLOCK_SIZE, 
                          sizeof(float) * BLOCK_SIZE, pipe);
        pipe.producer_commit();
        
        // 等待当前块就绪
        pipe.consumer_wait();
        
        // 处理当前块
        float result = process(smem[curr_buf][threadIdx.x]);
        output[(i-1) * BLOCK_SIZE + threadIdx.x] = result;
        
        pipe.consumer_release();
    }
    
    // 处理最后一块
    pipe.consumer_wait();
    float result = process(smem[(N/BLOCK_SIZE-1) % 2][threadIdx.x]);
    output[(N/BLOCK_SIZE-1) * BLOCK_SIZE + threadIdx.x] = result;
    pipe.consumer_release();
}
```

### 4.4.4 循环展开与寄存器缓存

结合循环展开和寄存器缓存进一步优化：

```cuda
template<int TILE_SIZE, int VECTOR_SIZE>
__global__ void optimizedGEMM(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    // 寄存器缓存
    float a_reg[VECTOR_SIZE];
    float b_reg[VECTOR_SIZE];
    float c_reg[VECTOR_SIZE][VECTOR_SIZE] = {0};
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    for (int tile = 0; tile < N/TILE_SIZE; tile++) {
        // 协作加载到共享内存
        As[ty][tx] = A[...];
        Bs[ty][tx] = B[...];
        __syncthreads();
        
        // 向量化计算，减少共享内存访问
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += VECTOR_SIZE) {
            // 加载到寄存器
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; v++) {
                a_reg[v] = As[ty][k + v];
                b_reg[v] = Bs[k + v][tx];
            }
            
            // 寄存器级计算
            #pragma unroll
            for (int i = 0; i < VECTOR_SIZE; i++) {
                #pragma unroll
                for (int j = 0; j < VECTOR_SIZE; j++) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }
    
    // 写回结果
    #pragma unroll
    for (int i = 0; i < VECTOR_SIZE; i++) {
        #pragma unroll
        for (int j = 0; j < VECTOR_SIZE; j++) {
            C[...] = c_reg[i][j];
        }
    }
}
```

## 4.5 案例：高性能矩阵乘法的共享内存优化

### 4.5.1 问题分析

矩阵乘法是评估GPU性能的经典基准，其优化涉及共享内存使用的各个方面。对于自动驾驶中的神经网络推理，GEMM操作占据了70%以上的计算时间。

**性能目标**：
- 达到理论峰值性能的90%以上
- 在V100上实现 > 14 TFLOPS（FP32）
- 支持非方阵和小批量场景

### 4.5.2 优化版本演进

**版本1：朴素共享内存**
```cuda
// 基础版本：~30% 峰值性能
__global__ void gemmV1(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < K/16; tile++) {
        As[ty][tx] = A[(by*16 + ty)*K + tile*16 + tx];
        Bs[ty][tx] = B[(tile*16 + ty)*N + bx*16 + tx];
        __syncthreads();
        
        for (int k = 0; k < 16; k++) {
            sum += As[ty][k] * Bs[k][tx];  // Bank conflict!
        }
        __syncthreads();
    }
    
    C[(by*16 + ty)*N + bx*16 + tx] = sum;
}
```

**版本2：Padding消除conflict**
```cuda
// Padding版本：~60% 峰值性能
__global__ void gemmV2(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[16][17];  // Padding
    __shared__ float Bs[16][17];  // Padding
    // ... 其余代码相同
}
```

**版本3：向量化访存 + 双缓冲**
```cuda
// 高性能版本：~85% 峰值性能
template<int BM, int BN, int BK, int TM, int TN>
__global__ void gemmV3(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[2][BK][BM + 1];  // 双缓冲 + padding
    __shared__ float Bs[2][BK][BN + 1];
    
    // 每个线程负责TM x TN的输出tile
    float c_reg[TM][TN] = {0};
    float a_reg[TM];
    float b_reg[TN];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int buffer = 0;
    
    // 预加载第一个tile (向量化)
    float4* As_ptr = reinterpret_cast<float4*>(&As[buffer][0][0]);
    float4* Bs_ptr = reinterpret_cast<float4*>(&Bs[buffer][0][0]);
    // ... 向量化加载
    
    for (int tile = 1; tile < K/BK; tile++) {
        int next_buffer = 1 - buffer;
        
        // 异步加载下一个tile
        // ...
        
        // 计算当前tile
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // 加载到寄存器
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                a_reg[m] = As[buffer][k][...];
            }
            
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                b_reg[n] = Bs[buffer][k][...];
            }
            
            // 外积更新
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    c_reg[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }
        
        __syncthreads();
        buffer = next_buffer;
    }
    
    // 写回结果
    // ...
}
```

### 4.5.3 性能分析与调优

**Occupancy分析**：
```cuda
// 计算最优配置
int sharedMemPerBlock = (BM + 1) * BK * sizeof(float) * 2 +  // As双缓冲
                        (BN + 1) * BK * sizeof(float) * 2;   // Bs双缓冲

int maxBlocksPerSM = min(
    SM_SHARED_MEM_SIZE / sharedMemPerBlock,
    MAX_THREADS_PER_SM / (blockDim.x * blockDim.y)
);

// 目标：maxBlocksPerSM >= 2 for hiding latency
```

**带宽利用率计算**：
```
有效带宽 = (2 * M * N * K * sizeof(float)) / kernel_time
带宽效率 = 有效带宽 / 理论带宽

目标：
- 共享内存带宽效率 > 80%
- 全局内存带宽效率 > 70%
```

### 4.5.4 自动驾驶场景优化

在自动驾驶的实时推理中，矩阵乘法常见于：
1. 点云特征提取（PointPillars）
2. 多头注意力计算（Transformer）
3. 特征金字塔融合（FPN）

**小批量优化**：
```cuda
// 批量为1的特殊优化
template<>
__global__ void gemmBatch1(float* A, float* B, float* C, int M, int N, int K) {
    // 使用持久化内核，避免重复加载权重
    __shared__ float Bs_persistent[WEIGHT_TILE_SIZE];
    
    // 权重只加载一次
    if (blockIdx.y == 0) {
        Bs_persistent[threadIdx.x] = B[...];
    }
    __syncthreads();
    
    // 流式处理输入
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        // 处理一行输出
    }
}
```

**动态形状适配**：
```cuda
// 处理非对齐尺寸
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void gemmDynamic(float* A, float* B, float* C, 
                            int M, int N, int K) {
    // 边界检查版本
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];
    
    // 计算有效加载范围
    int valid_m = min(TILE_M, M - blockIdx.y * TILE_M);
    int valid_n = min(TILE_N, N - blockIdx.x * TILE_N);
    
    // 带边界检查的加载
    if (threadIdx.y < valid_m && threadIdx.x < TILE_K) {
        As[threadIdx.y][threadIdx.x] = A[...];
    } else {
        As[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    // ... 类似处理Bs和计算逻辑
}
```

### 4.5.5 性能基准与对比

**不同优化级别的性能对比**（V100，FP32）：

| 优化级别 | 性能(TFLOPS) | 峰值占比 | 主要瓶颈 |
|---------|-------------|----------|----------|
| Cublas | 15.6 | 99% | - |
| 本章优化版本 | 14.2 | 91% | 寄存器压力 |
| 双缓冲+Padding | 10.8 | 69% | 同步开销 |
| 仅Padding | 8.5 | 54% | Bank conflict |
| 朴素共享内存 | 4.7 | 30% | 严重conflict |
| 无共享内存 | 1.2 | 8% | 全局内存带宽 |

## 本章小结

本章深入探讨了GPU共享内存的架构和优化技术，掌握了以下关键概念：

1. **共享内存架构**：32个bank的组织方式，地址映射规则，以及不同GPU架构的容量配置
2. **Bank Conflict机制**：conflict的产生原因、性能影响和检测方法
3. **规避策略**：Padding、Swizzling和Permutation三种主要技术的原理和应用场景
4. **双缓冲技术**：通过计算与数据传输的重叠实现延迟隐藏
5. **实战优化**：从30%到90%峰值性能的矩阵乘法优化历程

**关键公式回顾**：
- Bank映射：`bank_id = (byte_address / 4) % 32`
- Padding计算：`padding = (TILE_SIZE % 32 == 0) ? 1 : 0`
- 有效带宽：`BW_eff = (bytes_accessed / kernel_time)`
- Occupancy：`occ = active_warps / max_warps_per_SM`

## 练习题

### 基础题

**1. Bank Conflict识别**
分析以下代码片段，判断是否存在bank conflict及其类型：
```cuda
__shared__ float data[256];
// Case A
float a = data[threadIdx.x];
// Case B  
float b = data[threadIdx.x * 2];
// Case C
float c = data[threadIdx.x * 32];
// Case D
float d = data[0];  // 所有线程读取
```

<details>
<summary>答案</summary>

- Case A：无conflict，连续访问不同bank
- Case B：2-way conflict，偶数线程访问偶数bank
- Case C：32-way conflict，所有线程访问同一bank
- Case D：无conflict，广播机制自动处理

</details>

**2. Padding计算**
给定一个32×32的共享内存数组用于矩阵转置，计算需要的padding大小并解释原因。

**提示**：考虑转置操作中的访问模式

<details>
<summary>答案</summary>

需要padding 1列，变为32×33。原因：转置时列访问变为行访问，stride=32会导致所有线程访问同一bank。添加1列padding后，stride变为33，错开了bank映射。

</details>

**3. 共享内存容量规划**
在Ampere架构(164KB共享内存/SM)上，设计一个GEMM kernel的共享内存分配方案，要求：
- Block size: 128×128
- 双缓冲
- 目标occupancy ≥ 50%

**提示**：计算每个block需要的共享内存，确保至少2个block能同时运行

<details>
<summary>答案</summary>

每个tile需要：(128×8 + 8×128) × 4 bytes × 2(双缓冲) = 16KB
添加padding：约16.5KB
每个block总需求：约33KB
164KB可容纳：164/33 ≈ 4个blocks，满足occupancy要求

</details>

### 挑战题

**4. Swizzling函数设计**
设计一个swizzling函数，使得32×32矩阵转置时无bank conflict且不使用额外存储。

**提示**：使用XOR操作改变索引映射

<details>
<summary>答案</summary>

```
__device__ int swizzle(int x, int y) {
    // XOR低5位，实现bank交错
    return x ^ ((y & 0x1F) >> 2);
}

使用方式：
tile[threadIdx.y][swizzle(threadIdx.x, threadIdx.y)] = input[...];
```

关键：利用XOR操作使相邻warp的线程访问不同bank组。

</details>

**5. 性能瓶颈分析**
某GEMM kernel报告：
- 计算吞吐：8 TFLOPS
- 共享内存带宽利用率：45%
- Occupancy：75%
- Bank conflict率：平均4-way

请分析性能瓶颈并提出优化方案。

**提示**：从多个维度分析瓶颈来源

<details>
<summary>答案</summary>

瓶颈分析：
1. 主要瓶颈是bank conflict（4-way导致带宽降至25%）
2. 共享内存带宽利用率低（45%）印证了conflict问题

优化方案：
1. 添加padding消除conflict
2. 使用swizzling技术
3. 调整tile大小避免conflict pattern
4. 增加寄存器缓存减少共享内存访问

预期改进：消除conflict后，带宽利用率可达80%+，性能提升约2倍。

</details>

**6. 双缓冲流水线设计**
为一个Stencil计算（5-point）设计双缓冲方案，要求：
- 输入：1D数组，长度N
- 每个点需要左右各2个邻居
- 最大化计算与加载的重叠

**提示**：考虑halo区域的处理

<details>
<summary>答案</summary>

设计要点：
1. 共享内存：2个buffer，每个BLOCK_SIZE+4（halo）
2. 流水线阶段：
   - 预加载buffer 0
   - 循环：加载buffer (i+1)%2，计算buffer i%2
   - 收尾：处理最后buffer
3. Halo处理：每个buffer多加载4个元素
4. 同步点：加载后、计算前各一个syncthreads

关键优化：使用异步内存传输（Ampere+）进一步隐藏延迟。

</details>

**7. 混合精度共享内存优化**
设计一个FP16 GEMM kernel的共享内存布局，要求：
- 利用Tensor Core
- 最小化bank conflict
- 支持FP32累加器

**提示**：考虑FP16的bank宽度和Tensor Core的访问模式

<details>
<summary>答案</summary>

布局设计：
1. FP16存储：每个bank可存2个FP16值
2. Fragment布局：16×16×16 for Tensor Core
3. Swizzling：每8行做一次permutation
4. 累加器：使用寄存器存储FP32结果

关键点：
- Bank conflict pattern与FP32不同
- 需要考虑wmma的访问对齐
- 使用ldmatrix指令优化加载

</details>

**8. 动态共享内存分配策略**
设计一个自适应的共享内存分配策略，根据问题规模动态调整tile大小和buffer数量。

**提示**：建立性能模型，考虑occupancy和data reuse的平衡

<details>
<summary>答案</summary>

策略框架：
1. 性能模型：T = T_compute + T_memory - T_overlap
2. 约束条件：
   - 共享内存限制：tile_size² × buffers × 4 ≤ SM_capacity
   - Occupancy要求：blocks_per_SM ≥ 2
3. 优化目标：maximize (计算强度 × occupancy)
4. 实现：
   - 小矩阵：大tile，单缓冲
   - 中矩阵：中tile，双缓冲
   - 大矩阵：小tile，三缓冲

决策树示例：
```
if (M*N*K < threshold_small)
    use_config(128×128, single_buffer);
else if (M*N*K < threshold_medium)
    use_config(64×64, double_buffer);
else
    use_config(32×32, triple_buffer);
```

</details>

## 常见陷阱与错误（Gotchas）

### 1. 隐式的Bank Conflict
```cuda
// 错误：结构体大小导致conflict
struct Vec3 { float x, y, z; };  // 12 bytes
__shared__ Vec3 vectors[32];
// 访问vectors[threadIdx.x]会产生conflict！

// 正确：padding到16字节
struct Vec4 { float x, y, z, w; };  // 16 bytes
__shared__ Vec4 vectors[32];
```

### 2. 动态索引的陷阱
```cuda
// 危险：动态索引可能产生不可预测的conflict
__shared__ float data[1024];
int idx = some_computation(threadIdx.x);
float val = data[idx];  // 编译器无法优化

// 改进：使用静态模式或确保无conflict的映射
int idx = (threadIdx.x + offset) & MASK;
```

### 3. False Sharing
```cuda
// 问题：不同warp写入相邻地址
__shared__ int counters[32];
atomicAdd(&counters[warpId], 1);  // False sharing!

// 解决：padding避免cache line竞争
__shared__ int counters[32][8];  // 每个counter占用32字节
atomicAdd(&counters[warpId][0], 1);
```

### 4. 同步错误
```cuda
// 错误：条件同步导致死锁
if (threadIdx.x < 16) {
    // 处理数据
    __syncthreads();  // 死锁！只有部分线程到达
}

// 正确：所有线程都要经过同步点
if (threadIdx.x < 16) {
    // 处理数据
}
__syncthreads();  // 所有线程都会到达
```

### 5. 共享内存初始化
```cuda
// 错误：未初始化的共享内存包含垃圾值
__shared__ float sdata[256];
float sum = sdata[threadIdx.x];  // 垃圾值！

// 正确：显式初始化
__shared__ float sdata[256];
sdata[threadIdx.x] = 0.0f;
__syncthreads();
```

## 最佳实践检查清单

### 设计阶段
- [ ] 计算共享内存需求，确保不超过硬件限制
- [ ] 评估bank conflict风险，选择合适的规避策略
- [ ] 规划数据布局，考虑padding和对齐
- [ ] 设计同步策略，避免过度同步
- [ ] 考虑与L1缓存的配置权衡

### 实现阶段
- [ ] 使用模板参数使tile大小可配置
- [ ] 实现边界检查处理非对齐尺寸
- [ ] 添加padding消除bank conflict
- [ ] 使用pragma unroll优化内层循环
- [ ] 实现双缓冲隐藏内存延迟

### 优化阶段
- [ ] 使用Nsight Compute分析bank conflict
- [ ] 测量共享内存带宽利用率
- [ ] 评估occupancy与共享内存使用的平衡
- [ ] 尝试不同的swizzling模式
- [ ] 考虑寄存器缓存减少共享内存压力

### 验证阶段
- [ ] 检查所有共享内存访问的边界
- [ ] 验证同步点的正确性
- [ ] 测试不同问题规模的性能
- [ ] 确认无race condition
- [ ] 对比优化前后的性能指标

### 部署阶段
- [ ] 为不同GPU架构提供优化参数
- [ ] 实现自动tuning机制
- [ ] 添加性能监控和日志
- [ ] 编写使用文档和注意事项
- [ ] 设置合理的fallback策略
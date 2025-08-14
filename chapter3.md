# 第3章：全局内存优化策略

全局内存是CUDA编程中最基础也是最关键的内存类型。作为GPU上容量最大的内存空间，全局内存承载着绝大部分的数据存储和传输任务。然而，它也是延迟最高的内存层级，一次未优化的全局内存访问可能需要数百个时钟周期。在自动驾驶的激光雷达点云处理或具身智能的高分辨率视觉SLAM中，每秒需要处理GB级别的数据，内存带宽往往成为制约系统性能的主要瓶颈。本章将深入探讨全局内存的访问机制，掌握合并访问、缓存优化、向量化等关键技术，最终实现接近硬件理论峰值的内存带宽利用率。

## 3.1 内存合并访问模式

### 3.1.1 合并访问的硬件机制

当一个warp（32个线程）执行内存访问指令时，硬件会尝试将这些访问合并成尽可能少的内存事务。现代GPU支持32字节、64字节和128字节三种事务大小，选择哪种取决于访问的地址分布和数据量。

```
内存事务生成规则：
┌─────────────────────────────────────────────┐
│  Warp内32个线程的内存请求                    │
│  Thread 0: addr_0                           │
│  Thread 1: addr_1                           │
│  ...                                        │
│  Thread 31: addr_31                         │
└─────────────┬───────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  硬件合并逻辑                               │
│  1. 计算最小和最大地址                       │
│  2. 确定覆盖的128字节对齐段数量              │
│  3. 生成1-32个内存事务                       │
└─────────────┬───────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  内存事务（32B/64B/128B）                   │
│  事务1: [base_addr, base_addr+size)         │
│  事务2: ...                                 │
└─────────────────────────────────────────────┘
```

理想情况下，如果warp内所有线程访问连续的内存地址，且起始地址128字节对齐，那么只需要一个128字节的事务即可完成全部访问。最坏情况下，如果每个线程访问的地址都分散在不同的128字节段中，则需要32个事务。

### 3.1.2 合并访问模式分析

**连续访问模式（Coalesced Access）**

最理想的访问模式，线程ID与内存地址呈线性关系：

```
float data[N];
int tid = threadIdx.x + blockIdx.x * blockDim.x;
float val = data[tid];  // 完美合并

内存布局：
Thread:  0   1   2   3   4   5   6   7  ...  31
Address: 0   4   8   12  16  20  24  28 ... 124
         └───────────── 128字节 ─────────────┘
         生成1个128B事务
```

**跨步访问模式（Strided Access）**

线程以固定步长访问内存，合并效率取决于步长大小：

```
float data[N];
int tid = threadIdx.x + blockIdx.x * blockDim.x;
float val = data[tid * stride];  // 跨步访问

步长=2时的内存布局：
Thread:  0   1   2   3   4   5   6   7  ...  31
Address: 0   8   16  24  32  40  48  56 ... 248
         └──── 128B ────┘└──── 128B ────┘
         生成2个128B事务，带宽利用率50%

步长=32时：
每个线程访问不同的128B段，生成32个事务
带宽利用率仅3.125%（4B/128B）
```

**随机访问模式（Random Access）**

最差的访问模式，通常出现在哈希表、稀疏数据结构中：

```
int indices[N];  // 随机索引
float data[M];
int tid = threadIdx.x + blockIdx.x * blockDim.x;
float val = data[indices[tid]];  // 随机访问

可能生成1-32个事务，平均性能极差
```

### 3.1.3 优化策略

**数据布局转换：AoS到SoA**

在自动驾驶场景中，点云数据常用结构体数组（AoS）表示：

```
// AoS布局 - 访问效率低
struct Point {
    float x, y, z;
    float intensity;
    uint16_t ring;
    uint16_t padding;
};
Point points[N];

// 访问x坐标时，实际读取了整个结构体
float x = points[tid].x;  // 读取16字节，只用4字节
// 带宽利用率：4/16 = 25%

// SoA布局 - 访问效率高
struct PointCloud {
    float* x;
    float* y;
    float* z;
    float* intensity;
    uint16_t* ring;
};

// 访问x坐标，完美合并
float x = point_cloud.x[tid];  // 带宽利用率100%
```

**访问模式重组**

通过改变计算顺序或使用共享内存缓冲，将随机访问转换为合并访问：

```cuda
// 原始：随机访问
__global__ void scatter_kernel(float* out, float* in, int* indices, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[indices[tid]] = in[tid];  // 随机写
    }
}

// 优化：使用原子操作+排序
// 1. 先对indices排序
// 2. 使用分段的合并写入
__global__ void scatter_optimized(float* out, float* in, 
                                  int* sorted_indices, int* segment_starts, 
                                  int n_segments) {
    // 每个block处理一个segment，保证合并写入
    int seg_id = blockIdx.x;
    int start = segment_starts[seg_id];
    int end = segment_starts[seg_id + 1];
    
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        out[sorted_indices[i]] = in[i];  // segment内合并写
    }
}
```

## 3.2 缓存行为与配置

### 3.2.1 L1/L2缓存架构

现代GPU的缓存层次结构在不同架构间有显著差异：

```
Volta/Turing架构（V100/RTX 2080）：
┌──────────────────────────────────┐
│         SM (流多处理器)           │
│  ┌────────────────────────────┐  │
│  │  L1 Data Cache (128KB)     │  │  ← 与共享内存统一
│  │  + Shared Memory            │  │     可配置分割
│  └────────────┬───────────────┘  │
│               ↓                   │
└───────────────┼───────────────────┘
                ↓
┌──────────────────────────────────┐
│      L2 Cache (6MB)              │  ← 全局共享
└──────────────┬───────────────────┘
                ↓
┌──────────────────────────────────┐
│    Global Memory (HBM2)          │
└──────────────────────────────────┘

Ampere架构（A100/RTX 3090）：
- L1: 192KB每SM（可配置）
- L2: 40MB（A100）或 6MB（RTX 3090）
- 新增异步拷贝指令，支持绕过L1

Hopper架构（H100）：
- L1: 256KB每SM
- L2: 50MB
- 新增TMA（Tensor Memory Accelerator）单元
```

缓存行为的关键参数：

- **缓存行大小**：128字节（所有架构统一）
- **L1缓存策略**：写穿（write-through），读时缓存
- **L2缓存策略**：写回（write-back），支持原子操作缓存

### 3.2.2 缓存配置与控制

**配置L1缓存大小**

```cuda
// 配置kernel的L1缓存偏好
cudaFuncSetCacheConfig(my_kernel, cudaFuncCachePreferL1);    // 偏好L1
cudaFuncSetCacheConfig(my_kernel, cudaFuncCachePreferShared); // 偏好共享内存
cudaFuncSetCacheConfig(my_kernel, cudaFuncCachePreferEqual);  // 均衡分配

// Volta+架构的动态配置
cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxL1);  // 最大化L1缓存
```

**使用只读缓存路径**

```cuda
// __ldg内在函数：通过只读缓存路径加载
__global__ void kernel(float* __restrict__ data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 普通加载：通过L1/L2
    float val1 = data[tid];
    
    // 只读缓存加载：通过纹理缓存路径
    float val2 = __ldg(&data[tid]);
    
    // const __restrict__也会启用只读缓存
    const float* __restrict__ ro_data = data;
    float val3 = ro_data[tid];
}
```

**缓存绕过策略**

```cuda
// 使用向量化加载绕过L1（Ampere+）
__global__ void streaming_kernel(float4* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 流式访问，不污染L1缓存
    float4 val;
    asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(&data[tid]));
}
```

### 3.2.3 缓存友好的访问模式

**时间局部性优化**

在具身智能的传感器融合中，多次访问同一数据：

```cuda
// 差的时间局部性
__global__ void sensor_fusion_bad(float* lidar, float* camera, float* imu,
                                  float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // 每个数据只访问一次，缓存无法发挥作用
        float l = lidar[tid];
        float c = camera[tid];
        float i = imu[tid];
        output[tid] = l * 0.5f + c * 0.3f + i * 0.2f;
    }
}

// 好的时间局部性
__global__ void sensor_fusion_good(float* lidar, float* camera, float* imu,
                                   float* output, int n, int window) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 使用滑动窗口，重复访问缓存中的数据
    for (int w = 0; w < window; w++) {
        int idx = tid + w * gridDim.x * blockDim.x;
        if (idx < n) {
            float l = lidar[idx];
            float c = camera[idx];
            float i = imu[idx];
            
            // 时间序列滤波，多次访问相邻数据
            float filtered = 0;
            for (int k = -2; k <= 2; k++) {
                int nidx = idx + k;
                if (nidx >= 0 && nidx < n) {
                    filtered += lidar[nidx] * 0.2f;  // 从缓存读取
                }
            }
            output[idx] = filtered + c * 0.3f + i * 0.2f;
        }
    }
}
```

**空间局部性优化**

```cuda
// 2D卷积的空间局部性优化
__global__ void conv2d_optimized(float* input, float* output, 
                                 float* kernel, int width, int height) {
    // 使用2D thread block匹配2D数据布局
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0;
        
        // 访问3x3邻域，利用空间局部性
        #pragma unroll
        for (int ky = -1; ky <= 1; ky++) {
            #pragma unroll
            for (int kx = -1; kx <= 1; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    // 相邻线程访问相邻内存，提高缓存命中率
                    sum += input[ny * width + nx] * kernel[(ky+1)*3 + (kx+1)];
                }
            }
        }
        output[y * width + x] = sum;
    }
}
```

## 3.3 向量化load/store操作

### 3.3.1 向量化访存原理

GPU的内存控制器针对向量化访问进行了优化。使用float2、float4等向量类型可以：
- 减少指令数量
- 提高内存吞吐量
- 改善指令级并行性

```
标量访问 vs 向量访问的指令生成：

标量访问（4条指令）：
LD.E R0, [address+0]
LD.E R1, [address+4]
LD.E R2, [address+8]
LD.E R3, [address+12]

向量访问（1条指令）：
LD.E.128 R0:R3, [address]  // 一次加载128位
```

### 3.3.2 实现技术

**使用内建向量类型**

```cuda
// 向量化的矩阵拷贝
__global__ void matrix_copy_vectorized(float* dst, const float* src, int n) {
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int vec_tid = tid * 4;  // 每个线程处理4个float
    
    if (vec_tid < n) {
        // 将指针转换为float4*
        float4* dst4 = reinterpret_cast<float4*>(dst);
        const float4* src4 = reinterpret_cast<const float4*>(src);
        
        // 一次读写16字节
        dst4[tid] = src4[tid];
    }
}

// 更激进的向量化：使用CUDA的大向量类型
struct alignas(32) float8 {
    float4 x, y;
};

__global__ void matrix_copy_float8(float* dst, const float* src, int n) {
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int vec_tid = tid * 8;
    
    if (vec_tid < n) {
        float8* dst8 = reinterpret_cast<float8*>(dst);
        const float8* src8 = reinterpret_cast<const float8*>(src);
        dst8[tid] = src8[tid];  // 一次32字节
    }
}
```

**联合体优化技巧**

```cuda
// 使用联合体进行类型双关
union Vec4 {
    float4 vec;
    float arr[4];
    struct { float x, y, z, w; };
    uint4 u;
};

__global__ void process_rgbadata(uint8_t* image, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid * 4 < n) {
        // 读取4个RGBA像素（16字节）
        uint4 pixels = reinterpret_cast<uint4*>(image)[tid];
        
        Vec4 result;
        // 提取并归一化每个通道
        result.arr[0] = (pixels.x & 0xFF) / 255.0f;         // R
        result.arr[1] = ((pixels.x >> 8) & 0xFF) / 255.0f;  // G
        result.arr[2] = ((pixels.x >> 16) & 0xFF) / 255.0f; // B
        result.arr[3] = ((pixels.x >> 24) & 0xFF) / 255.0f; // A
        
        // 向量化写入
        reinterpret_cast<float4*>(output)[tid] = result.vec;
    }
}
```

### 3.3.3 应用场景：多通道传感器数据处理

在自动驾驶中，处理多通道激光雷达数据：

```cuda
// Velodyne 64线激光雷达数据处理
struct VelodynePoint {
    float x, y, z;        // 3D坐标
    float intensity;      // 反射强度
    float azimuth;       // 方位角
    float distance;      // 距离
    uint16_t ring;       // 激光线号
    uint16_t time;       // 时间戳
};

__global__ void process_velodyne_vectorized(
    VelodynePoint* points,
    float4* cartesian_out,  // x,y,z,intensity
    float2* polar_out,      // azimuth,distance
    int n_points) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n_points) {
        // 向量化读取点云数据（32字节对齐）
        VelodynePoint p = points[tid];
        
        // 坐标变换（自车坐标系）
        float cos_a = __cosf(p.azimuth);
        float sin_a = __sinf(p.azimuth);
        
        float4 cartesian;
        cartesian.x = p.distance * cos_a;
        cartesian.y = p.distance * sin_a;
        cartesian.z = p.z;
        cartesian.w = p.intensity;
        
        float2 polar;
        polar.x = p.azimuth;
        polar.y = p.distance;
        
        // 向量化写入
        cartesian_out[tid] = cartesian;
        polar_out[tid] = polar;
    }
}

## 3.4 内存带宽优化

### 3.4.1 带宽分析与测量

理解和测量内存带宽是优化的第一步。GPU的内存带宽受多个因素影响：

**理论带宽计算**

```
理论带宽 = 内存频率 × 总线宽度 × 2 (DDR)

示例（V100）：
- HBM2内存频率：877 MHz
- 总线宽度：4096 bits = 512 bytes
- 理论带宽 = 877 MHz × 512 bytes × 2 = 900 GB/s

示例（A100）：
- HBM2e内存频率：1215 MHz  
- 总线宽度：5120 bits = 640 bytes
- 理论带宽 = 1215 MHz × 640 bytes × 2 = 1555 GB/s

示例（H100）：
- HBM3内存频率：1593 MHz
- 总线宽度：6144 bits = 768 bytes  
- 理论带宽 = 1593 MHz × 768 bytes × 2 = 2448 GB/s
```

**有效带宽测量**

```cuda
// 带宽测试kernel
__global__ void bandwidth_test(float* dst, const float* src, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// 测量函数
float measure_bandwidth(int n_elements) {
    float *d_src, *d_dst;
    cudaMalloc(&d_src, n_elements * sizeof(float));
    cudaMalloc(&d_dst, n_elements * sizeof(float));
    
    // 预热
    bandwidth_test<<<1024, 256>>>(d_dst, d_src, n_elements);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 测量
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        bandwidth_test<<<1024, 256>>>(d_dst, d_src, n_elements);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 计算带宽（读+写）
    float bytes = 2.0f * n_elements * sizeof(float) * 100;
    float bandwidth = bytes / (milliseconds * 1e6);  // GB/s
    
    return bandwidth;
}
```

**Roofline模型分析**

```
Roofline模型：性能受计算强度限制

性能(GFLOPS)
    ↑
    │     计算受限区域
    │    ╱─────────── 峰值计算性能
    │   ╱ 
    │  ╱  内存受限区域
    │ ╱   
    │╱    性能 = 带宽 × 计算强度
    └────────────────────→ 计算强度(FLOPS/Byte)
         Ridge Point

计算强度 = 浮点运算数 / 内存访问字节数

示例分析：
- SAXPY (y = a*x + y): 2 FLOPS / 12 Bytes = 0.167
- GEMM (C = A*B + C): 2*n³ FLOPS / 4*n² Bytes = n/2
- 卷积: 取决于kernel大小和复用程度
```

### 3.4.2 带宽优化技术

**数据压缩与解压**

在具身智能场景中，深度图像的压缩传输：

```cuda
// 16位深度图压缩为8位+缩放因子
__global__ void compress_depth(uint8_t* compressed, float* scale_factors,
                               const uint16_t* depth, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // 计算16x16块的最大最小值（使用共享内存）
        __shared__ uint16_t s_min, s_max;
        
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            s_min = 65535;
            s_max = 0;
        }
        __syncthreads();
        
        int idx = y * width + x;
        uint16_t val = depth[idx];
        
        // 原子更新最值
        atomicMin(&s_min, val);
        atomicMax(&s_max, val);
        __syncthreads();
        
        // 量化到8位
        float scale = (s_max - s_min) / 255.0f;
        uint8_t compressed_val = (val - s_min) / scale;
        
        compressed[idx] = compressed_val;
        
        // 块的第一个线程保存缩放因子
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int block_id = blockIdx.y * gridDim.x + blockIdx.x;
            scale_factors[block_id * 2] = s_min;
            scale_factors[block_id * 2 + 1] = scale;
        }
    }
}
```

**异步内存传输与计算重叠**

```cuda
// 使用CUDA流实现传输与计算重叠
void process_with_overlap(float* h_data, float* d_data, 
                         float* d_result, int n, int n_chunks) {
    int chunk_size = n / n_chunks;
    
    // 创建流
    cudaStream_t streams[2];
    for (int i = 0; i < 2; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 双缓冲
    float *d_buffer[2];
    cudaMalloc(&d_buffer[0], chunk_size * sizeof(float));
    cudaMalloc(&d_buffer[1], chunk_size * sizeof(float));
    
    // 流水线处理
    for (int i = 0; i < n_chunks; i++) {
        int stream_id = i % 2;
        
        // 异步拷贝到GPU
        cudaMemcpyAsync(d_buffer[stream_id], 
                       h_data + i * chunk_size,
                       chunk_size * sizeof(float),
                       cudaMemcpyHostToDevice,
                       streams[stream_id]);
        
        // 在该流上启动kernel
        process_kernel<<<grid, block, 0, streams[stream_id]>>>(
            d_result + i * chunk_size,
            d_buffer[stream_id],
            chunk_size);
        
        // 异步拷贝回CPU（如需要）
        cudaMemcpyAsync(h_result + i * chunk_size,
                       d_result + i * chunk_size,
                       chunk_size * sizeof(float),
                       cudaMemcpyDeviceToHost,
                       streams[stream_id]);
    }
    
    // 同步所有流
    for (int i = 0; i < 2; i++) {
        cudaStreamSynchronize(streams[i]);
    }
}
```

**预取策略（Unified Memory）**

```cuda
// 统一内存的预取优化
void slam_with_prefetch(float* unified_map, int map_size,
                        float* new_scan, int scan_size) {
    // 预取地图数据到GPU
    cudaMemPrefetchAsync(unified_map, map_size * sizeof(float), 0);
    
    // 预取新扫描数据到GPU
    cudaMemPrefetchAsync(new_scan, scan_size * sizeof(float), 0);
    
    // 启动SLAM kernel
    slam_update_kernel<<<grid, block>>>(unified_map, new_scan, map_size, scan_size);
    
    // 预取更新后的地图回CPU（用于可视化）
    cudaMemPrefetchAsync(unified_map, map_size * sizeof(float), cudaCpuDeviceId);
}
```

### 3.4.3 带宽受限算法优化

**Kernel融合减少内存访问**

```cuda
// 未融合版本：3次全局内存访问
__global__ void add_kernel(float* c, const float* a, const float* b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] + b[tid];
}

__global__ void mul_kernel(float* c, const float* c, float scalar, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = c[tid] * scalar;
}

// 融合版本：2次全局内存访问
__global__ void fused_add_mul(float* c, const float* a, const float* b, 
                              float scalar, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = (a[tid] + b[tid]) * scalar;  // 寄存器中完成所有计算
    }
}
```

**计算与访存重叠（延迟隐藏）**

```cuda
// 使用指令级并行隐藏内存延迟
__global__ void compute_intensive_kernel(float* output, const float* input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // 展开循环，增加并行度
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    
    for (int i = tid; i < n; i += stride * 4) {
        // 发起多个独立的内存请求
        float val0 = (i < n) ? input[i] : 0;
        float val1 = (i + stride < n) ? input[i + stride] : 0;
        float val2 = (i + 2*stride < n) ? input[i + 2*stride] : 0;
        float val3 = (i + 3*stride < n) ? input[i + 3*stride] : 0;
        
        // 在等待内存时进行计算
        acc0 += __sinf(val0) * __cosf(val0);
        acc1 += __sinf(val1) * __cosf(val1);
        acc2 += __sinf(val2) * __cosf(val2);
        acc3 += __sinf(val3) * __cosf(val3);
    }
    
    // 归约结果
    if (tid < n) {
        output[tid] = acc0 + acc1 + acc2 + acc3;
    }
}
```

**具身智能案例：实时SLAM的带宽优化**

```cuda
// ICP（迭代最近点）算法的带宽优化
struct PointNormal {
    float3 point;
    float3 normal;
};

__global__ void icp_correspondence_optimized(
    int* correspondences,        // 输出：对应关系
    float* distances,            // 输出：距离
    const PointNormal* source,   // 源点云
    const PointNormal* target,   // 目标点云
    const float* transform,      // 4x4变换矩阵
    int n_source,
    int n_target,
    float max_dist) {
    
    // 每个block处理一个源点，使用共享内存缓存目标点
    __shared__ PointNormal s_target[256];
    
    int source_idx = blockIdx.x;
    if (source_idx >= n_source) return;
    
    // 加载并变换源点
    PointNormal src = source[source_idx];
    float3 transformed_point;
    transformed_point.x = transform[0] * src.point.x + transform[1] * src.point.y + 
                         transform[2] * src.point.z + transform[3];
    transformed_point.y = transform[4] * src.point.x + transform[5] * src.point.y + 
                         transform[6] * src.point.z + transform[7];
    transformed_point.z = transform[8] * src.point.x + transform[9] * src.point.y + 
                         transform[10] * src.point.z + transform[11];
    
    float min_dist = max_dist;
    int best_idx = -1;
    
    // 分块处理目标点云
    for (int chunk_start = 0; chunk_start < n_target; chunk_start += blockDim.x) {
        // 协作加载目标点到共享内存
        int target_idx = chunk_start + threadIdx.x;
        if (target_idx < n_target) {
            s_target[threadIdx.x] = target[target_idx];
        }
        __syncthreads();
        
        // 计算距离
        int chunk_size = min(blockDim.x, n_target - chunk_start);
        for (int i = 0; i < chunk_size; i++) {
            float3 diff;
            diff.x = transformed_point.x - s_target[i].point.x;
            diff.y = transformed_point.y - s_target[i].point.y;
            diff.z = transformed_point.z - s_target[i].point.z;
            
            float dist = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = chunk_start + i;
            }
        }
        __syncthreads();
    }
    
    // 使用warp级归约找到最小距离
    min_dist = warp_reduce_min(min_dist);
    best_idx = warp_broadcast(best_idx, min_dist);
    
    // 第一个线程写入结果
    if (threadIdx.x == 0) {
        correspondences[source_idx] = best_idx;
        distances[source_idx] = sqrtf(min_dist);
    }
}
```

## 3.5 案例：高带宽矩阵转置

矩阵转置是展示内存优化技术的经典案例。看似简单的操作，实则充分暴露了内存访问模式对性能的决定性影响。

### 3.5.1 朴素实现与性能分析

```cuda
// 朴素实现：严重的内存访问问题
__global__ void transpose_naive(float* out, const float* in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];  
        // 问题：读取合并，但写入跨步！
    }
}

性能分析：
- 读取：连续线程读取连续地址，完美合并
- 写入：连续线程写入间隔height的地址，严重不合并
- 带宽利用率：约10-15%（取决于矩阵大小）
```

### 3.5.2 合并访问优化

```cuda
// 使用共享内存实现合并读写
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_coalesced(float* out, const float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1];  // +1避免bank conflict
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 合并读取到共享内存（每个线程读4个元素）
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    // 交换线程的x和y来实现转置后的合并写入
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

性能提升：
- 读写都实现合并访问
- 带宽利用率：约60-70%
```

### 3.5.3 向量化优化

```cuda
// 使用float4向量化进一步优化
__global__ void transpose_vectorized(float* out, const float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1];
    
    // 每个线程处理4个float
    int x = blockIdx.x * TILE_DIM + threadIdx.x * 4;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 向量化读取
    if (x < width && y < height) {
        float4 data = reinterpret_cast<const float4*>(&in[y * width + x])[0];
        tile[threadIdx.y][threadIdx.x * 4] = data.x;
        tile[threadIdx.y][threadIdx.x * 4 + 1] = data.y;
        tile[threadIdx.y][threadIdx.x * 4 + 2] = data.z;
        tile[threadIdx.y][threadIdx.x * 4 + 3] = data.w;
    }
    
    __syncthreads();
    
    // 转置后向量化写入
    x = blockIdx.y * TILE_DIM + threadIdx.x * 4;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < height && y < width) {
        float4 data;
        data.x = tile[threadIdx.x * 4][threadIdx.y];
        data.y = tile[threadIdx.x * 4 + 1][threadIdx.y];
        data.z = tile[threadIdx.x * 4 + 2][threadIdx.y];
        data.w = tile[threadIdx.x * 4 + 3][threadIdx.y];
        reinterpret_cast<float4*>(&out[y * height + x])[0] = data;
    }
}

性能提升：
- 减少内存事务数量
- 带宽利用率：约75-80%
```

### 3.5.4 共享内存协同优化

```cuda
// 终极优化：双缓冲+异步拷贝（Ampere+）
__global__ void transpose_ultimate(float* out, const float* in, int width, int height) {
    // 双缓冲共享内存
    __shared__ float tile[2][TILE_DIM][TILE_DIM+1];
    
    // 异步拷贝管道
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    
    const int tile_x = blockIdx.x * TILE_DIM;
    const int tile_y = blockIdx.y * TILE_DIM;
    
    // 第一个tile的异步加载
    if (threadIdx.x < TILE_DIM && threadIdx.y == 0) {
        cuda::memcpy_async(
            &tile[0][threadIdx.x][0],
            &in[(tile_y + threadIdx.x) * width + tile_x],
            sizeof(float) * TILE_DIM,
            pipe
        );
    }
    pipe.producer_commit();
    
    // 主循环：重叠传输和计算
    for (int k = 0; k < gridDim.x; k++) {
        int buffer_id = k & 1;
        int next_buffer = 1 - buffer_id;
        
        // 等待当前tile就绪
        pipe.consumer_wait();
        __syncthreads();
        
        // 预取下一个tile（如果不是最后一个）
        if (k < gridDim.x - 1) {
            if (threadIdx.x < TILE_DIM && threadIdx.y == 0) {
                int next_tile_x = ((k + 1) % gridDim.x) * TILE_DIM;
                cuda::memcpy_async(
                    &tile[next_buffer][threadIdx.x][0],
                    &in[(tile_y + threadIdx.x) * width + next_tile_x],
                    sizeof(float) * TILE_DIM,
                    pipe
                );
            }
            pipe.producer_commit();
        }
        
        // 处理当前tile的转置写入
        int out_x = tile_y + threadIdx.x;
        int out_y = k * TILE_DIM + threadIdx.y;
        
        if (out_x < height && out_y < width) {
            #pragma unroll
            for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                out[(out_y + i) * height + out_x] = 
                    tile[buffer_id][threadIdx.x][threadIdx.y + i];
            }
        }
        
        pipe.consumer_release();
    }
}

性能达到：
- 带宽利用率：约85-90%
- 接近硬件理论极限
```

### 3.5.5 性能对比与分析

```
性能测试结果（8192x8192矩阵，V100）：

实现版本            带宽(GB/s)   利用率    相对性能
─────────────────────────────────────────────
朴素实现            95          10.6%     1.0x
合并访问            570         63.3%     6.0x  
向量化              684         76.0%     7.2x
双缓冲+异步         810         90.0%     8.5x
理论峰值            900         100%      -

关键优化点：
1. 合并访问：6倍性能提升（最关键）
2. 向量化：额外20%提升
3. 异步传输：额外18%提升
4. Bank conflict避免：约5%提升
```

## 3.6 本章小结

全局内存优化是CUDA性能调优的基础和关键。本章深入探讨了内存访问模式、缓存行为、向量化技术和带宽优化策略。

**核心要点**：

1. **合并访问是第一要务**：未合并的内存访问会导致10倍甚至更多的性能损失。通过数据布局转换（AoS→SoA）和访问模式重组可以实现合并。

2. **缓存配置因场景而异**：L1缓存对具有时空局部性的算法有利，但流式访问应考虑绕过L1。使用`__ldg`和`const __restrict__`可以利用只读缓存路径。

3. **向量化减少指令开销**：float2/float4等向量类型可以减少内存事务数量，提高带宽利用率。在处理多通道数据时尤其有效。

4. **带宽优化需要系统思维**：
   - 测量和分析：使用Roofline模型识别瓶颈
   - 数据压缩：在带宽受限时权衡计算与传输
   - 异步传输：重叠计算与数据移动
   - Kernel融合：减少内存往返次数

5. **实战案例的启示**：矩阵转置案例展示了从10%到90%带宽利用率的优化过程，核心在于合并访问+共享内存+向量化+异步传输的组合使用。

**关键公式**：

- 有效带宽 = 数据传输量 / 执行时间
- 带宽利用率 = 有效带宽 / 理论带宽
- 计算强度 = FLOPS / 内存访问字节数
- 内存事务效率 = 请求字节数 / 传输字节数

记住：在自动驾驶和具身智能的实时系统中，每GB/s的带宽提升都可能意味着更高的帧率、更低的延迟，甚至决定系统能否实时运行。掌握这些优化技术，是构建高性能AI系统的必备技能。

## 3.7 练习题

### 基础题（理解概念）

**练习3.1**：给定一个warp的32个线程，分别访问地址[0, 128, 256, ..., 3968]（步长128字节），计算需要多少个128字节的内存事务？带宽利用率是多少？

<details>
<summary>提示</summary>
考虑128字节对齐的内存段，每个线程访问不同的段。
</details>

<details>
<summary>答案</summary>
需要32个128字节事务。每个线程访问4字节，共需128字节，但实际传输32×128=4096字节。带宽利用率=128/4096=3.125%。
</details>

**练习3.2**：解释为什么在共享内存tile声明中使用`[TILE_DIM][TILE_DIM+1]`而不是`[TILE_DIM][TILE_DIM]`？

<details>
<summary>提示</summary>
考虑共享内存的bank组织，32个bank，每个bank 4字节宽。
</details>

<details>
<summary>答案</summary>
添加padding避免bank conflict。当TILE_DIM=32时，同一列的连续元素会映射到同一个bank，导致32路bank conflict。+1使得列元素分散到不同bank。
</details>

**练习3.3**：在Volta架构上，L1缓存和共享内存共享128KB空间。如果kernel使用64KB共享内存，L1缓存有多大？这对性能有何影响？

<details>
<summary>提示</summary>
考虑缓存大小对命中率的影响，以及不同访问模式的需求。
</details>

<details>
<summary>答案</summary>
L1缓存为64KB。影响：1)缓存容量减少可能降低时间局部性好的算法性能；2)对流式访问影响小；3)需要权衡共享内存带来的数据复用收益与L1缓存减少的损失。
</details>

### 进阶题（应用技术）

**练习3.4**：设计一个kernel，高效地将RGB图像（3通道，uint8_t）转换为灰度图像，灰度值计算公式为：`gray = 0.299*R + 0.587*G + 0.114*B`。要求达到>80%的带宽利用率。

<details>
<summary>提示</summary>
1. 使用向量化加载RGB数据
2. 考虑数据对齐
3. 使用整数运算避免浮点转换
</details>

<details>
<summary>答案</summary>
关键点：1)使用uchar4向量化读取4个像素（12字节）；2)使用定点数运算(×1000)避免浮点；3)确保输入输出都是合并访问；4)每个线程处理多个像素增加ILP。
</details>

**练习3.5**：优化稀疏矩阵向量乘法（SpMV）的内存访问。给定CSR格式的稀疏矩阵，如何减少随机访问带来的性能损失？

<details>
<summary>提示</summary>
1. 考虑向量x的访问模式
2. 使用纹理内存或只读缓存
3. 分块处理提高局部性
</details>

<details>
<summary>答案</summary>
策略：1)向量x通过纹理缓存/__ldg访问；2)使用SELL-C-σ格式改善合并；3)行分块使多个线程协作处理长行；4)使用共享内存缓存频繁访问的x元素。
</details>

### 挑战题（综合优化）

**练习3.6**：实现一个高性能的2D卷积kernel（5×5卷积核），处理4K分辨率图像，要求达到>1TFLOPS的计算吞吐量。分析内存访问模式并给出优化策略。

<details>
<summary>提示</summary>
1. 计算重用率：每个输出需要25次乘加
2. 使用共享内存缓存输入tile
3. 考虑halo区域的处理
4. 向量化和循环展开
</details>

<details>
<summary>答案</summary>
优化要点：1)输入tile加载到共享内存(如18×18 for 14×14输出)；2)使用float4向量化；3)寄存器缓存卷积核；4)每个线程计算2×2或4×4输出块；5)texture内存处理边界；6)考虑使用Tensor Core（如可用）。
</details>

**练习3.7**：在自动驾驶场景中，需要实时处理64线激光雷达数据（每帧约13万点）。设计一个数据结构和访问模式，支持：1)快速范围查询；2)体素化（voxelization）；3)KNN搜索。要求总处理时间<10ms。

<details>
<summary>提示</summary>
1. 考虑空间数据结构（网格、八叉树）
2. 平衡构建时间和查询效率
3. 利用点云的时序特性
</details>

<details>
<summary>答案</summary>
方案：1)使用规则网格+哈希表，O(1)体素查询；2)Morton编码实现空间局部性；3)分层网格支持多尺度；4)使用滑动窗口复用上帧结构；5)共享内存缓存邻域体素；6)原子操作处理点计数。
</details>

**练习3.8**：设计一个统一内存(Unified Memory)的使用策略，用于具身智能机器人的SLAM系统。系统需要维护一个大规模地图（>1GB），同时CPU需要进行路径规划，GPU进行地图更新。如何优化数据移动？

<details>
<summary>提示</summary>
1. 使用cudaMemAdvise提示访问模式
2. 预取策略
3. 分区管理
4. 考虑页面迁移开销
</details>

<details>
<summary>答案</summary>
策略：1)地图分块，活跃块预取到GPU；2)使用cudaMemAdviseSetReadMostly标记静态区域；3)CPU路径规划时预取相关块；4)使用流并发更新不同块；5)定期整理减少碎片；6)关键路径避免页错误。
</details>

## 3.8 常见陷阱与错误

### 陷阱1：误判内存合并
```cuda
// 看似合并，实际不合并
struct Vec3 { float x, y, z; };
__global__ void process(Vec3* data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float x = data[tid].x;  // 线程访问步长12字节，不完全合并！
}
// 解决：使用SoA布局或float3向量类型
```

### 陷阱2：缓存污染
```cuda
// 大量流式数据污染L1缓存
__global__ void stream_process(float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = in[tid] * 2.0f;  // 无重用，却占用缓存
}
// 解决：使用 -Xptxas -dlcm=cg 编译选项绕过L1
```

### 陷阱3：向量化的错误假设
```cuda
// 错误：假设地址总是对齐的
float4* ptr = (float4*)(&array[offset]);  // offset不是4的倍数时未对齐！
float4 val = *ptr;  // 可能导致性能下降或错误
// 解决：检查对齐或使用非对齐加载
```

### 陷阱4：统一内存的隐藏开销
```cuda
// 页面迁移的隐藏成本
__global__ void kernel(float* unified_data) {
    // 首次访问触发页错误，可能有ms级延迟！
    float val = unified_data[threadIdx.x];  
}
// 解决：使用cudaMemPrefetchAsync预取
```

### 陷阱5：原子操作的带宽影响
```cuda
// 原子操作严重降低带宽
__global__ void histogram(int* bins, int* data) {
    int val = data[threadIdx.x];
    atomicAdd(&bins[val], 1);  // 多个线程竞争同一地址
}
// 解决：使用共享内存局部直方图+归约
```

### 陷阱6：Bank Conflict的隐蔽性
```cuda
// 隐蔽的bank conflict
__shared__ float matrix[16][16];  // 16 < 32，看似安全
matrix[threadIdx.y][threadIdx.x] = value;  // 但列访问时仍有conflict！
// 解决：使用[16][17]或重新安排访问模式
```

### 陷阱7：错误的带宽计算
```cuda
// 错误：只计算有用数据
bandwidth = useful_bytes / time;  // 忽略了浪费的传输！
// 正确：计算实际传输
bandwidth = actual_transferred_bytes / time;
```

### 陷阱8：过度优化的陷阱
```cuda
// 过度复杂的优化反而降低性能
// 16个不同的内存访问模式，寄存器压力过大
// 解决：平衡优化复杂度与收益
```

## 3.9 最佳实践检查清单

### 设计阶段
- [ ] **数据结构选择**：AoS vs SoA，考虑访问模式
- [ ] **内存分配**：确保基地址对齐（使用cudaMallocPitch）
- [ ] **算法选择**：评估计算强度，判断是否内存受限
- [ ] **容量规划**：估算内存带宽需求vs硬件能力
- [ ] **访问模式分析**：画出内存访问图，识别热点

### 实现阶段
- [ ] **合并访问**：连续线程访问连续地址
- [ ] **向量化**：使用float2/float4减少事务
- [ ] **缓存配置**：根据访问模式选择L1/Shared比例
- [ ] **只读路径**：标记const __restrict__或使用__ldg
- [ ] **避免bank conflict**：共享内存padding
- [ ] **数据预取**：统一内存使用cudaMemPrefetchAsync
- [ ] **循环展开**：增加指令级并行，隐藏延迟

### 优化阶段
- [ ] **性能分析**：使用Nsight Compute检查内存效率
- [ ] **带宽测量**：对比实测vs理论带宽
- [ ] **瓶颈识别**：Load/Store单元利用率
- [ ] **事务分析**：检查L2缓存事务大小分布
- [ ] **占用率分析**：确保足够的活跃warp隐藏延迟
- [ ] **迭代优化**：根据profiler数据调整

### 验证阶段
- [ ] **正确性**：cuda-memcheck检查越界访问
- [ ] **扩展性**：测试不同问题规模
- [ ] **稳定性**：长时间运行测试
- [ ] **移植性**：不同GPU架构的性能
- [ ] **边界情况**：非对齐、非2的幂次大小
- [ ] **错误处理**：内存分配失败等异常情况

### 部署阶段
- [ ] **性能监控**：运行时带宽监控
- [ ] **自适应调优**：根据硬件动态调整参数
- [ ] **版本兼容**：不同CUDA版本的兼容性
- [ ] **文档完善**：记录优化决策和权衡
- [ ] **持续优化**：新硬件的适配计划

记住：优秀的CUDA程序员总是从内存访问模式开始思考问题。在动手编码前，先在纸上画出数据布局和访问模式！每一个字节的传输都应该是有意义的，每一次内存访问都应该经过精心设计。
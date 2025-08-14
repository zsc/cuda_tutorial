# 第25章：性能分析与调优方法论

本章深入探讨CUDA程序性能分析与优化的系统化方法论。我们将学习如何识别性能瓶颈、使用Roofline模型进行理论分析、优化内存访问模式、平衡占用率与资源使用，以及实现指令级并行。通过一个真实的优化案例，展示如何将性能提升10倍到100倍的完整过程。

## 25.1 性能瓶颈识别流程

### 25.1.1 性能分析的层次化方法

性能优化应遵循自顶向下的层次化方法：

```
应用层面 (Application Level)
    ↓
算法层面 (Algorithm Level)  
    ↓
内核层面 (Kernel Level)
    ↓
指令层面 (Instruction Level)
```

每个层次都有其特定的分析工具和优化策略。应用层面关注整体吞吐量和延迟，算法层面关注复杂度和并行度，内核层面关注资源利用率，指令层面关注流水线效率。

### 25.1.2 性能指标体系

关键性能指标（KPI）包括：

1. **吞吐量指标**
   - 有效带宽利用率 (Effective Bandwidth Utilization)
   - 计算吞吐量 (FLOPS/IOPS)
   - 内存吞吐量 (GB/s)

2. **延迟指标**
   - 内核执行时间
   - 内存访问延迟
   - 同步开销

3. **效率指标**
   - SM占用率 (Occupancy)
   - 指令吞吐量 (Instructions Per Cycle)
   - 分支效率 (Branch Efficiency)

### 25.1.3 瓶颈识别工具链

**Nsight Compute工作流程：**

```
1. 初始分析 (Speed of Light)
   ├── SM吞吐量
   ├── 内存吞吐量
   └── 达到的峰值百分比

2. 详细分析 (Detailed Metrics)
   ├── 内存工作负载分析
   ├── 计算工作负载分析
   ├── 占用率分析
   └── 指令混合分析

3. 源码级分析 (Source Level)
   ├── 热点代码定位
   ├── 内存访问模式
   └── 分支分歧分析
```

### 25.1.4 瓶颈分类与识别

**计算瓶颈特征：**
- SM利用率接近100%
- 内存带宽利用率较低
- 指令吞吐量饱和

**内存瓶颈特征：**
- 内存带宽利用率高
- SM空闲时间多（stall）
- L2缓存命中率低

**延迟瓶颈特征：**
- 占用率低
- 寄存器/共享内存使用过多
- 同步频繁

### 25.1.5 性能剖析实践

使用nvprof/ncu的典型命令：

```bash
# 基础性能剖析
ncu --target-processes all ./application

# 详细内存分析
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
    ./application

# 占用率分析
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --metrics sm__maximum_warps_per_active_cycle_pct \
    ./application
```

## 25.2 Roofline模型分析

### 25.2.1 Roofline模型基础

Roofline模型将程序性能表示为算术强度（Arithmetic Intensity）的函数：

```
性能上界 = min(峰值计算性能, 算术强度 × 峰值内存带宽)

其中：
算术强度 = 计算操作数 / 内存访问字节数 (FLOP/Byte)
```

模型可视化：

```
性能 (GFLOPS)
    ↑
    │     计算瓶颈区域
    │    ╱━━━━━━━━━━━━━━━ 峰值计算性能
    │   ╱ 
    │  ╱  内存瓶颈区域
    │ ╱   
    │╱    
    └────────────────────→ 算术强度 (FLOP/Byte)
         Ridge Point
```

### 25.2.2 硬件参数获取

不同GPU架构的关键参数：

```
GPU型号        峰值计算(TFLOPS)  峰值带宽(GB/s)  Ridge Point(FLOP/Byte)
─────────────────────────────────────────────────────────────────────
V100          15.7 (FP32)       900            17.4
A100          19.5 (FP32)       1555           12.5  
H100          67.0 (FP32)       3350           20.0
RTX 4090      82.6 (FP32)       1008           81.9
```

### 25.2.3 算术强度计算

实际算术强度的精确计算需要考虑：

1. **缓存效应**
   ```
   有效算术强度 = 计算操作数 / (DRAM读取 + DRAM写入)
   ```

2. **数据重用**
   ```
   重用因子 = 总访问字节数 / 唯一访问字节数
   ```

3. **混合精度**
   ```
   等效FLOPS = FP32_OPS + 2×FP16_OPS + 4×INT8_OPS
   ```

### 25.2.4 性能优化路径

基于Roofline模型的优化策略：

```
当前位置：内存瓶颈区域
优化路径1：提高算术强度
  ├── 增加数据重用（tiling）
  ├── 算法融合（kernel fusion）
  └── 降低精度（quantization）

优化路径2：提高内存效率
  ├── 改善访存模式（coalescing）
  ├── 使用共享内存（caching）
  └── 压缩数据（compression）
```

### 25.2.5 多级Roofline分析

考虑内存层次的扩展模型：

```
性能上界 = min(
    峰值计算性能,
    AI × L1带宽,
    AI × L2带宽,
    AI × DRAM带宽
)
```

这允许我们识别具体的内存层次瓶颈。

## 25.3 内存访问模式优化

### 25.3.1 合并访问优化

**理想的合并访问模式：**

```
线程 0: 访问地址 0x1000
线程 1: 访问地址 0x1004
线程 2: 访问地址 0x1008
线程 3: 访问地址 0x100C
...
线程31: 访问地址 0x107C
→ 一次128字节事务
```

**非合并访问的代价：**

```
情况              事务数  带宽效率
──────────────────────────────────
完全合并          1      100%
跨步访问(stride=2) 2      50%
跨步访问(stride=4) 4      25%
随机访问          32     3.125%
```

### 25.3.2 内存访问模式分析

使用Nsight Compute检测访问模式：

```
关键指标：
- gld_efficiency: 全局加载效率
- gst_efficiency: 全局存储效率
- gld_transactions_per_request: 每请求事务数
- l2_cache_hit_rate: L2缓存命中率
```

**优化前后对比示例：**

```
优化前（AoS布局）：
struct Point { float x, y, z; };
Point points[N];
// 访问pattern: x0,y0,z0,x1,y1,z1,...

优化后（SoA布局）：
float x[N], y[N], z[N];
// 访问pattern: x0,x1,x2,... y0,y1,y2,... z0,z1,z2,...

性能提升：3-4x
```

### 25.3.3 缓存优化策略

**L1/L2缓存配置：**

```cuda
// 配置L1缓存大小
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);

// 缓存配置选项：
cudaFuncCachePreferNone   // 无偏好
cudaFuncCachePreferShared  // 偏好共享内存
cudaFuncCachePreferL1      // 偏好L1缓存
cudaFuncCachePreferEqual   // 均衡分配
```

**缓存行为优化：**

1. **时间局部性优化**
   - 数据重用窗口最小化
   - 循环分块（tiling）

2. **空间局部性优化**
   - 连续内存访问
   - 数据预取

### 25.3.4 共享内存优化

**Bank Conflict避免策略：**

```
策略1：Padding
shared_memory[threadIdx.y][threadIdx.x + 1]  // 添加1个元素偏移

策略2：Permutation
index = (threadIdx.x + threadIdx.y) % 32

策略3：Swizzling  
index = threadIdx.x ^ (threadIdx.y & 0x1F)
```

**双缓冲技术：**

```cuda
__shared__ float buffer[2][TILE_SIZE];
int current = 0;

// 主循环
for(int i = 0; i < iterations; i++) {
    // 异步加载到buffer[current]
    // 计算使用buffer[1-current]
    __syncthreads();
    current = 1 - current;
}
```

### 25.3.5 向量化访存

**使用向量化load/store：**

```cuda
// 标量访问 (32位)
float val = data[idx];

// 向量化访问 (128位)
float4 val = reinterpret_cast<float4*>(data)[idx/4];

// 性能提升：最高4x
```

**内存对齐要求：**

```
数据类型    对齐要求    建议事务大小
────────────────────────────────────
float       4字节      32字节
float2      8字节      64字节
float4      16字节     128字节
```

## 25.4 占用率与寄存器平衡

### 25.4.1 占用率计算

占用率定义：
```
占用率 = 活跃warp数 / 最大warp数
```

限制因素：
1. 寄存器使用
2. 共享内存使用
3. 线程块大小

**占用率计算示例（A100）：**

```
SM资源限制：
- 最大线程数：2048
- 最大warp数：64
- 寄存器数：65536
- 共享内存：164KB

内核配置：
- 线程块大小：256
- 每线程寄存器：64
- 共享内存：48KB

计算：
- 寄存器限制的块数 = 65536/(256*64) = 4
- 共享内存限制的块数 = 164/48 = 3
- 线程限制的块数 = 2048/256 = 8

实际块数 = min(4,3,8) = 3
占用率 = (3*256/32)/64 = 37.5%
```

### 25.4.2 寄存器压力管理

**寄存器溢出的影响：**

```
寄存器使用  位置      访问延迟   带宽
─────────────────────────────────────
≤32        寄存器    1 cycle    极高
33-255     寄存器    1 cycle    高
>255       本地内存  200 cycles 低
```

**寄存器优化技术：**

```cuda
// 1. 寄存器重用
float tmp = a[i];
b[i] = tmp * 2;
c[i] = tmp * 3;  // 重用tmp

// 2. 编译器提示
__launch_bounds__(256, 4)  // 限制寄存器使用

// 3. 寄存器分配控制
#pragma unroll 4  // 控制展开程度
```

### 25.4.3 共享内存与占用率权衡

**动态共享内存分配：**

```cuda
extern __shared__ float shared[];

// 内核启动时指定大小
kernel<<<blocks, threads, shared_size>>>();
```

**占用率敏感度分析：**

```
占用率    相对性能    适用场景
─────────────────────────────────
25%      60-70%     高寄存器压力
50%      80-90%     平衡型负载
75%      95-100%    内存密集型
100%     90-95%     计算密集型
```

### 25.4.4 线程块配置优化

**最优线程块大小选择：**

```
原则：
1. 能被32整除（warp大小）
2. 至少192-256线程（隐藏延迟）
3. 不超过512线程（调度开销）

常用配置：
- 1D: 256, 512
- 2D: (16,16), (32,8)
- 3D: (8,8,4), (8,4,8)
```

### 25.4.5 动态并行与占用率

**动态并行的占用率影响：**

```cuda
__global__ void parent() {
    if(threadIdx.x == 0) {
        child<<<1, 32>>>();  // 动态启动
        cudaDeviceSynchronize();
    }
}
```

动态并行会预留资源，降低父内核占用率。建议：
- 批量启动子内核
- 使用流实现异步执行
- 考虑使用持久化内核

## 25.5 指令级并行优化

### 25.5.1 指令流水线

GPU指令流水线阶段：

```
取指(IF) → 译码(ID) → 执行(EX) → 访存(MEM) → 写回(WB)

延迟隐藏需求：
- 算术指令：4-6个warp
- 内存指令：20-40个warp
- 特殊函数：8-16个warp
```

### 25.5.2 指令级并行度(ILP)

**提高ILP的技术：**

```cuda
// 低ILP（串行依赖）
float sum = 0;
for(int i = 0; i < N; i++) {
    sum += data[i];  // 每次迭代依赖前一次
}

// 高ILP（并行累加）
float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
for(int i = 0; i < N; i += 4) {
    sum0 += data[i];
    sum1 += data[i+1];
    sum2 += data[i+2];
    sum3 += data[i+3];
}
float sum = sum0 + sum1 + sum2 + sum3;
```

### 25.5.3 循环展开优化

**手动循环展开：**

```cuda
#pragma unroll 8
for(int i = 0; i < 64; i++) {
    result[i] = a[i] * b[i] + c[i];
}
```

**展开因子选择：**

```
展开因子  寄存器压力  指令缓存  性能提升
────────────────────────────────────────
2         低         低        10-20%
4         中         中        20-40%
8         高         高        30-50%
16        很高       溢出      20-30%
```

### 25.5.4 指令调度优化

**避免指令相关性：**

```cuda
// 差的调度（连续依赖）
a = b * c;
d = a + e;  // 等待a
f = d * g;  // 等待d

// 好的调度（交错独立指令）
a = b * c;
h = i * j;  // 独立计算
d = a + e;
k = h + l;  // 独立计算
f = d * g;
m = k * n;
```

### 25.5.5 混合精度与指令选择

**利用Tensor Core：**

```cuda
// FP16累加到FP32
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**指令吞吐量对比：**

```
指令类型        V100    A100    H100
───────────────────────────────────
FP32 FMA       64      64      128
FP16 FMA       128     128     256
INT8 DP4A      256     512     512
Tensor Core    8x      16x     32x
```

## 25.6 案例：从10x到100x的优化历程

### 25.6.1 问题描述

优化目标：自动驾驶场景中的3D点云体素化处理

输入：
- 100万个3D点（激光雷达数据）
- 空间范围：[-50m, 50m] × [-50m, 50m] × [-3m, 3m]
- 体素大小：0.1m × 0.1m × 0.2m

输出：
- 1000×1000×30的体素网格
- 每个体素的点数和特征统计

### 25.6.2 基准实现（Baseline）

```cuda
__global__ void voxelize_baseline(
    float3* points, int n_points,
    int* voxel_indices, float* voxel_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_points) return;
    
    float3 point = points[idx];
    
    // 计算体素索引
    int vx = (point.x + 50.0f) / 0.1f;
    int vy = (point.y + 50.0f) / 0.1f;
    int vz = (point.z + 3.0f) / 0.2f;
    
    if(vx >= 0 && vx < 1000 && 
       vy >= 0 && vy < 1000 && 
       vz >= 0 && vz < 30) {
        
        int voxel_idx = vx + vy * 1000 + vz * 1000000;
        
        // 原子操作更新体素
        atomicAdd(&voxel_features[voxel_idx * 4 + 0], point.x);
        atomicAdd(&voxel_features[voxel_idx * 4 + 1], point.y);
        atomicAdd(&voxel_features[voxel_idx * 4 + 2], point.z);
        atomicAdd(&voxel_features[voxel_idx * 4 + 3], 1.0f);
    }
}

// 性能：100ms
```

### 25.6.3 优化迭代1：内存访问优化（2x）

```cuda
// 使用float4向量化访问
__global__ void voxelize_v1(
    float4* points, int n_points,  // 改为float4
    int* voxel_indices, float4* voxel_features) {  // 改为float4
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_points) return;
    
    float4 point = points[idx];  // 一次加载4个float
    
    // ... 体素索引计算 ...
    
    // 向量化原子操作
    atomicAdd(&voxel_features[voxel_idx].x, point.x);
    atomicAdd(&voxel_features[voxel_idx].y, point.y);
    atomicAdd(&voxel_features[voxel_idx].z, point.z);
    atomicAdd(&voxel_features[voxel_idx].w, 1.0f);
}

// 性能：50ms (2x加速)
```

### 25.6.4 优化迭代2：减少原子操作冲突（5x）

```cuda
// 使用共享内存做局部聚合
__global__ void voxelize_v2(
    float4* points, int n_points,
    int* voxel_indices, float4* voxel_features) {
    
    __shared__ float4 local_voxels[LOCAL_VOXEL_SIZE];
    __shared__ int local_indices[LOCAL_VOXEL_SIZE];
    __shared__ int local_count;
    
    if(threadIdx.x == 0) local_count = 0;
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_points) {
        float4 point = points[idx];
        
        // 计算体素索引
        int voxel_idx = compute_voxel_index(point);
        
        // 先在共享内存中聚合
        int local_idx = atomicAdd(&local_count, 1);
        if(local_idx < LOCAL_VOXEL_SIZE) {
            local_indices[local_idx] = voxel_idx;
            local_voxels[local_idx] = point;
        }
    }
    __syncthreads();
    
    // 批量写回全局内存
    // ...
}

// 性能：20ms (5x加速)
```

### 25.6.5 优化迭代3：空间哈希优化（10x）

```cuda
// 使用空间哈希减少内存占用
__global__ void voxelize_v3(
    float4* points, int n_points,
    HashTable* hash_table) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_points) return;
    
    float4 point = points[idx];
    
    // 使用Morton编码作为哈希键
    uint32_t morton = morton3D(
        (point.x + 50.0f) / 0.1f,
        (point.y + 50.0f) / 0.1f,
        (point.z + 3.0f) / 0.2f
    );
    
    // 哈希表插入（使用开放寻址）
    hash_table.insert(morton, point);
}

// 性能：10ms (10x加速)
```

### 25.6.6 优化迭代4：多流并行（20x）

```cuda
// 使用多流处理不同空间区域
void voxelize_multi_stream(
    float4* points, int n_points,
    VoxelGrid* grid) {
    
    const int n_streams = 8;
    cudaStream_t streams[n_streams];
    
    // 空间划分
    for(int s = 0; s < n_streams; s++) {
        cudaStreamCreate(&streams[s]);
        
        int start = (n_points * s) / n_streams;
        int end = (n_points * (s+1)) / n_streams;
        
        voxelize_kernel<<<blocks, threads, 0, streams[s]>>>(
            points + start, end - start, grid
        );
    }
    
    // 同步所有流
    for(int s = 0; s < n_streams; s++) {
        cudaStreamSynchronize(streams[s]);
    }
}

// 性能：5ms (20x加速)
```

### 25.6.7 优化迭代5：专用硬件特性（50x）

```cuda
// 利用Tensor Core加速特征计算
__global__ void voxelize_tensor_core(
    float4* points, int n_points,
    VoxelGrid* grid) {
    
    // 使用协作组
    cooperative_groups::grid_group g = 
        cooperative_groups::this_grid();
    
    // 使用Tensor Core做批量矩阵运算
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // 批量处理16×16的点
    // ...
    
    // 网格级同步
    g.sync();
}

// 性能：2ms (50x加速)
```

### 25.6.8 最终优化：算法级改进（100x）

```cuda
// 层次化体素结构 + 稀疏表示
class HierarchicalVoxelGrid {
    // Level 0: 粗粒度体素 (1m × 1m × 1m)
    // Level 1: 中粒度体素 (0.2m × 0.2m × 0.4m)  
    // Level 2: 细粒度体素 (0.1m × 0.1m × 0.2m)
    
    __device__ void insert(float4 point) {
        // 只在非空的粗粒度体素中创建细粒度体素
        int coarse_idx = compute_coarse_index(point);
        if(atomicCAS(&coarse_occupied[coarse_idx], 0, 1) == 0) {
            // 第一个点，分配细粒度结构
            allocate_fine_voxels(coarse_idx);
        }
        
        // 插入到细粒度体素
        int fine_idx = compute_fine_index(point, coarse_idx);
        insert_to_fine_voxel(fine_idx, point);
    }
};

// 性能：1ms (100x加速)
```

### 25.6.9 性能优化总结

```
优化阶段                性能    加速比   关键技术
────────────────────────────────────────────────────
Baseline               100ms    1x      原始实现
内存向量化             50ms     2x      float4访问
共享内存聚合           20ms     5x      减少原子冲突
空间哈希               10ms     10x     稀疏表示
多流并行               5ms      20x     并发执行
Tensor Core            2ms      50x     专用硬件
算法优化               1ms      100x    层次化结构
```

### 25.6.10 优化经验总结

1. **性能分析驱动**：每次优化前先profile找瓶颈
2. **逐步优化**：不要一次改动太多
3. **算法与实现并重**：算法优化往往带来最大提升
4. **硬件特性利用**：充分利用新硬件特性
5. **权衡取舍**：精度vs性能，内存vs计算

## 本章小结

本章系统介绍了CUDA性能分析与优化的完整方法论：

1. **性能瓶颈识别**：通过层次化分析方法和专业工具定位性能瓶颈
2. **Roofline模型**：理论分析性能上界，指导优化方向
3. **内存优化**：合并访问、缓存优化、向量化等技术
4. **占用率平衡**：权衡资源使用，最大化硬件利用率
5. **指令级优化**：提高ILP、循环展开、指令调度
6. **实战案例**：展示100倍性能提升的完整优化过程

关键公式：
- Roofline性能上界：`P = min(P_peak, AI × BW_peak)`
- 占用率：`Occupancy = Active_Warps / Max_Warps`
- 有效带宽：`BW_eff = BW_peak × Efficiency`

## 练习题

### 基础题

1. **Roofline模型计算**
   给定GPU峰值性能500 GFLOPS，内存带宽100 GB/s，某内核执行1000次浮点运算，访问250字节内存。计算该内核的性能上界。
   
   <details>
   <summary>提示</summary>
   计算算术强度AI = FLOP/Byte，然后应用Roofline公式
   </details>
   
   <details>
   <summary>答案</summary>
   AI = 1000/250 = 4 FLOP/Byte
   性能上界 = min(500, 4×100) = 400 GFLOPS
   该内核受内存带宽限制
   </details>

2. **占用率分析**
   某GPU的SM有32768个寄存器，最大64个warp。内核使用每线程40个寄存器，线程块大小256。计算最大占用率。
   
   <details>
   <summary>提示</summary>
   计算寄存器限制的最大块数，然后计算对应的warp数
   </details>
   
   <details>
   <summary>答案</summary>
   每块寄存器需求 = 256 × 40 = 10240
   最大块数 = 32768 / 10240 = 3
   活跃warp数 = 3 × 256 / 32 = 24
   占用率 = 24 / 64 = 37.5%
   </details>

3. **内存合并分析**
   32个线程访问数组A[tid * stride]，stride分别为1、2、4时，需要多少个128字节事务？（假设float类型）
   
   <details>
   <summary>提示</summary>
   考虑128字节能覆盖32个float的情况
   </details>
   
   <details>
   <summary>答案</summary>
   stride=1: 1个事务（完全合并）
   stride=2: 2个事务（间隔访问）
   stride=4: 4个事务（更稀疏访问）
   </details>

### 挑战题

4. **性能瓶颈诊断**
   某内核的profile显示：SM利用率30%，内存带宽利用率85%，L2缓存命中率20%，占用率75%。分析性能瓶颈并提出优化建议。
   
   <details>
   <summary>提示</summary>
   高内存带宽利用率+低缓存命中率通常意味着什么？
   </details>
   
   <details>
   <summary>答案</summary>
   瓶颈：内存访问模式差，大量cache miss
   优化建议：
   1. 改善数据局部性（tiling）
   2. 使用共享内存缓存
   3. 优化数据布局（AoS→SoA）
   4. 考虑数据压缩减少带宽需求
   </details>

5. **优化策略选择**
   矩阵乘法C=A×B，A为M×K，B为K×N。当M=N=4096，K=128时，如何优化？当M=N=128，K=4096时呢？
   
   <details>
   <summary>提示</summary>
   计算两种情况的算术强度，分析是计算瓶颈还是内存瓶颈
   </details>
   
   <details>
   <summary>答案</summary>
   情况1 (K小)：AI = O(K) = 128，计算瓶颈
   - 使用Tensor Core
   - 最大化占用率
   - 循环展开
   
   情况2 (K大)：AI = O(MN/K) = 4，内存瓶颈  
   - 优化内存访问
   - 使用共享内存tiling
   - 考虑分块算法
   </details>

6. **多版本内核设计**
   设计一个自适应的归约求和内核，根据数据规模N自动选择优化策略。
   
   <details>
   <summary>提示</summary>
   不同规模需要不同的并行策略和内存访问模式
   </details>
   
   <details>
   <summary>答案</summary>
   N < 1024: 单块，使用共享内存
   N < 1M: 多块两阶段归约
   N < 100M: 使用原子操作+多块
   N > 100M: 多级归约+持久化内核
   根据N/SM数量决定块配置
   </details>

7. **性能模型预测**
   建立一个简单的性能模型，预测矩阵转置的执行时间。考虑内存带宽、缓存效应和占用率。
   
   <details>
   <summary>提示</summary>
   T = max(T_compute, T_memory)，考虑有效带宽
   </details>
   
   <details>
   <summary>答案</summary>
   T = max(
       N²/(SM_count × Throughput × Occupancy),
       2N² × sizeof(float) / (BW_peak × Efficiency)
   )
   其中Efficiency取决于访问模式：
   - 朴素转置：~25%
   - 分块转置：~60%
   - 优化转置：~85%
   </details>

8. **端到端优化方案**
   为点云配准（ICP算法）设计完整的GPU优化方案，包括数据结构、内核设计和性能目标。
   
   <details>
   <summary>提示</summary>
   ICP包括最近邻搜索、对应点匹配、变换矩阵计算
   </details>
   
   <details>
   <summary>答案</summary>
   1. 数据结构：KD-Tree用BFS布局优化缓存
   2. 最近邻：使用纹理内存+共享内存缓存
   3. 矩阵计算：利用Tensor Core
   4. 多GPU：空间划分并行
   5. 目标：100Hz处理100k点云
   优化重点：减少树遍历的内存访问
   </details>

## 常见陷阱与错误

1. **过度优化占用率**
   - 错误：盲目追求100%占用率
   - 正确：根据内核特性选择合适占用率

2. **忽视内存访问模式**
   - 错误：只关注计算优化
   - 正确：内存优化往往带来更大提升

3. **不当的性能度量**
   - 错误：只看内核时间
   - 正确：考虑端到端时间和数据传输

4. **过早优化**
   - 错误：一开始就做底层优化
   - 正确：先优化算法，再优化实现

5. **忽略硬件差异**
   - 错误：一套代码跑所有GPU
   - 正确：针对不同架构调优

## 最佳实践检查清单

### 性能分析阶段
- [ ] 使用Nsight Compute进行详细profile
- [ ] 识别主要性能瓶颈（计算/内存/延迟）
- [ ] 计算Roofline模型理论性能
- [ ] 建立性能基准和优化目标

### 内存优化阶段
- [ ] 确保内存访问合并
- [ ] 优化数据布局（AoS vs SoA）
- [ ] 使用适当的缓存策略
- [ ] 实现共享内存tiling
- [ ] 应用向量化load/store

### 计算优化阶段
- [ ] 平衡占用率与资源使用
- [ ] 提高指令级并行度
- [ ] 使用循环展开
- [ ] 利用专用硬件（Tensor Core）
- [ ] 考虑混合精度计算

### 系统级优化阶段
- [ ] 实现流并行
- [ ] 优化CPU-GPU通信
- [ ] 使用CUDA Graph减少启动开销
- [ ] 实现内核融合
- [ ] 考虑多GPU扩展

### 验证与部署阶段
- [ ] 验证数值精度
- [ ] 测试不同输入规模
- [ ] 评估不同GPU架构性能
- [ ] 记录优化过程和结果
- [ ] 建立持续优化流程
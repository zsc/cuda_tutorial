# 第11章：激光雷达点云处理加速

## 章节大纲

### 11.1 点云数据结构与GPU存储优化
- 点云数据的特性与挑战
- AoS vs SoA布局选择
- 动态点云的内存管理
- 空间局部性优化

### 11.2 空间索引结构的并行构建
- KD-Tree的GPU并行构建算法
- Octree的自底向上构建
- Morton编码与空间填充曲线
- 平衡树的负载均衡策略

### 11.3 最近邻搜索的CUDA优化
- 暴力搜索的向量化实现
- KD-Tree搜索的并行化
- 近似最近邻算法
- 批量查询优化

### 11.4 点云配准算法加速
- ICP算法的GPU实现
- NDT配准的并行化策略
- 对应点匹配优化
- 变换矩阵的并行求解

### 11.5 实时3D目标检测
- PointPillars的CUDA实现
- CenterPoint的关键优化
- 体素化与特征提取
- NMS的GPU加速

### 本章小结
### 练习题
### 常见陷阱与错误
### 最佳实践检查清单

---

## 11.1 点云数据结构与GPU存储优化

激光雷达作为自动驾驶的核心传感器，每秒产生数百万个3D点。典型的64线激光雷达在10Hz频率下，每帧包含约120,000个点，每个点包含位置(x,y,z)、强度(intensity)、时间戳等信息。实时处理这些海量数据需要充分利用GPU的并行计算能力，而高效的数据结构设计是性能优化的基础。

### 点云数据的特性与挑战

点云数据具有以下独特特性：

1. **稀疏性**：点云在3D空间中分布不均匀，远处稀疏、近处密集
2. **无序性**：原始点云没有固定的拓扑结构
3. **动态范围大**：坐标值范围从几米到上百米
4. **噪声敏感**：包含测量噪声和动态物体

这些特性给GPU处理带来挑战：

```
内存访问模式挑战：
┌─────────────────────────────────┐
│  稀疏点云 → 不规则内存访问      │
│  动态大小 → 内存分配开销        │
│  空间查询 → 随机访问模式        │
└─────────────────────────────────┘
```

### AoS vs SoA布局选择

点云数据在GPU上有两种主要存储布局：

**AoS (Array of Structures)**：
```
Point points[N];  // 每个点的所有属性连续存储
struct Point {
    float x, y, z;
    float intensity;
    uint32_t timestamp;
};
```

**SoA (Structure of Arrays)**：
```
struct PointCloud {
    float* x;        // 所有x坐标连续
    float* y;        // 所有y坐标连续  
    float* z;        // 所有z坐标连续
    float* intensity;
    uint32_t* timestamp;
};
```

性能对比分析：

| 操作类型 | AoS性能 | SoA性能 | 原因分析 |
|---------|---------|---------|----------|
| 单点完整访问 | 优秀 | 较差 | AoS局部性好 |
| 批量坐标计算 | 较差 | 优秀 | SoA合并访问 |
| 空间查询 | 中等 | 优秀 | SoA便于SIMD |
| 内存带宽利用 | 60-70% | 85-95% | SoA避免浪费 |

对于大多数点云算法，SoA布局能获得1.5-3倍的性能提升，因为：
- 坐标计算（距离、法向量）只需访问xyz
- 向量化指令(float4)可一次加载多个点的同一属性
- 避免加载不需要的属性浪费带宽

### 动态点云的内存管理

自动驾驶场景中点云大小动态变化（隧道vs开阔路面），需要高效的GPU内存管理策略：

**1. 内存池预分配**
```
初始化阶段：
┌──────────────────────────────┐
│ 预分配最大容量 (如200K点)     │
│ ┌────┬────┬────┬────┬────┐  │
│ │Pool│Pool│Pool│Pool│Free│  │
│ └────┴────┴────┴────┴────┘  │
└──────────────────────────────┘

运行时动态调整：
- 小点云：使用部分内存池
- 大点云：扩展或使用多池
```

**2. 双缓冲流水线**
```
Frame N:   [GPU处理] ← Buffer A
           同时
Frame N+1: [CPU→GPU传输] → Buffer B

下一时刻切换：
Frame N+1: [GPU处理] ← Buffer B  
Frame N+2: [CPU→GPU传输] → Buffer A
```

这种设计可以完全隐藏PCIe传输延迟，实现零拷贝开销。

**3. 统一内存与按需分页**

CUDA统一内存(Unified Memory)简化编程但需谨慎使用：
```
优点：
- 自动数据迁移
- 超额分配支持
- 简化指针管理

缺点：
- 页错误开销 (首次访问~10μs)
- 难以控制预取
- 可能触发抖动
```

最佳实践：静态场景地图用UM，动态点云用显式管理。

### 空间局部性优化

提高空间局部性是优化缓存命中率的关键：

**1. Morton编码重排序**

将3D坐标映射到1D Morton码，保持空间邻近性：
```
原始顺序（按扫描线）：
点1(0,0,0) → 点2(100,0,0) → 点3(0,1,0)
缓存命中率低，空间跳跃大

Morton重排序后：
点1(0,0,0) → 点3(0,1,0) → 点5(1,0,0)  
空间相邻点在内存中也相邻
```

Morton编码通过交织xyz的二进制位实现：
```
x = 0b0011 (3)
y = 0b0101 (5)  
z = 0b0110 (6)
Morton = 0b001101011110 (交织: z1y1x1z0y0x0...)
```

**2. 分块处理（Tiling）**

将点云划分为空间块，每个线程块处理一个空间块：
```
┌─────┬─────┬─────┐
│ B0  │ B1  │ B2  │  64x64x64m空间
├─────┼─────┼─────┤  划分为8x8x8m块
│ B3  │ B4  │ B5  │  每块分配给一个
├─────┼─────┼─────┤  线程块处理
│ B6  │ B7  │ B8  │  
└─────┴─────┴─────┘
```

优势：
- L2缓存局部性提升3-5倍
- 减少全局内存访问50%+
- 便于负载均衡（动态调度块）

**3. 数据压缩与量化**

利用点云的有限精度特性压缩存储：
```
原始: float32 x,y,z (12字节/点)
压缩: int16 x,y,z (6字节/点)
      范围[-327.68m, 327.67m]
      精度: 0.01m (厘米级)

压缩率: 50%
带宽提升: 2x
精度损失: <1cm (可接受)
```

对于颜色/强度等属性，8位量化通常足够。

### 向量化访存优化

利用CUDA的向量类型提升内存吞吐：

**1. float4向量加载**
```cuda
// 低效：3次独立加载
float x = points_x[idx];
float y = points_y[idx];  
float z = points_z[idx];

// 高效：1次向量加载（需要额外padding）
float4 point = reinterpret_cast<float4*>(points)[idx];
// point.x, point.y, point.z, point.w(padding)
```

性能提升：25-40%（取决于其他计算开销）

**2. 共享内存的向量化填充**
```cuda
// 每个线程加载4个点到共享内存
float4* shared_vec = reinterpret_cast<float4*>(shared_points);
int vec_idx = threadIdx.x;
shared_vec[vec_idx] = global_vec[block_offset + vec_idx];
__syncthreads();
```

这种方式可以达到>90%的内存带宽利用率。

### 内存访问模式优化实例

下面展示一个优化前后的距离计算核函数对比：

**优化前（AoS布局，非合并访问）：**
```cuda
__global__ void computeDistance_v1(Point* points, float* distances, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Point p = points[idx];  // 加载20字节，只用12字节
        distances[idx] = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
    }
}
// 性能：~40GB/s带宽利用率
```

**优化后（SoA布局，向量化访问）：**
```cuda
__global__ void computeDistance_v2(float4* points_xyz, float* distances, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 p = points_xyz[idx];  // 一次加载xyz+padding
        distances[idx] = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
    }
}
// 性能：~380GB/s带宽利用率（接近理论峰值）
```

优化效果：9.5倍带宽提升，4-6倍整体性能提升。

## 11.2 空间索引结构的并行构建

空间索引是点云处理的核心数据结构，支持高效的最近邻搜索、范围查询和空间划分。传统的串行构建算法在GPU上难以直接并行化，本节介绍适合GPU的并行构建策略。

### KD-Tree的GPU并行构建算法

KD-Tree是k维空间的二叉搜索树，在3D点云中广泛应用。GPU并行构建的核心挑战是树结构的递归特性与GPU的SIMT执行模型不匹配。

**传统串行构建的问题：**
```
串行递归构建：
1. 选择分割维度和分割点
2. 划分点集为左右子集
3. 递归构建左右子树

问题：
- 递归深度不确定 → GPU栈溢出
- 分支不平衡 → warp divergence
- 指针操作频繁 → 内存不合并
```

**GPU并行构建策略：**

1. **自顶向下的层次并行**

将树的构建按层次进行，每层并行处理所有节点：

```
Level 0: [Root]           - 1个节点
Level 1: [L] [R]          - 2个节点并行
Level 2: [LL][LR][RL][RR] - 4个节点并行
...
Level k: 2^k个节点并行处理
```

实现要点：
```cuda
// 每个线程块处理一个节点的分割
__global__ void buildKDTreeLevel(
    float* points,          // 点云数据
    int* node_ranges,       // 每个节点的点范围
    int* split_dims,        // 分割维度
    float* split_values,    // 分割值
    int level,              // 当前层
    int num_nodes           // 该层节点数
) {
    int node_id = blockIdx.x;
    if (node_id >= num_nodes) return;
    
    // 获取该节点负责的点范围
    int start = node_ranges[node_id * 2];
    int end = node_ranges[node_id * 2 + 1];
    
    // 协作计算中位数（使用共享内存）
    __shared__ float shared_coords[256];
    int split_dim = level % 3;  // 循环选择xyz
    
    // 并行加载坐标到共享内存
    cooperativeLoadCoords(points, shared_coords, start, end, split_dim);
    
    // 并行快速选择算法找中位数
    float median = parallelQuickSelect(shared_coords, end - start);
    
    if (threadIdx.x == 0) {
        split_dims[node_id] = split_dim;
        split_values[node_id] = median;
    }
    
    // 并行划分点集
    parallelPartition(points, start, end, split_dim, median);
}
```

2. **基于Morton码的隐式构建**

利用Morton码的空间填充特性，无需显式构建树结构：

```
Morton码排序后的隐式KD-Tree：
┌────────────────────────┐
│ 点按Morton码排序       │
│ [P0,P1,P2...Pn]       │
└────────────────────────┘
         ↓
隐式树结构（无需指针）：
- 节点i的左子：2i+1
- 节点i的右子：2i+2
- 完全平衡树
```

优势：
- 无指针操作，纯数组访问
- 完全平衡，无warp divergence
- 构建时间O(n log n)，仅排序开销

3. **小规模子树的串行处理**

当子树规模小于阈值（如32点）时，切换到串行处理：

```cuda
if (num_points < SERIAL_THRESHOLD) {
    // 单线程串行构建小子树
    if (threadIdx.x == 0) {
        buildSerialKDTree(points, start, end, node);
    }
} else {
    // 继续并行构建
    launchParallelSplit<<<blocks, threads>>>(...);
}
```

这种混合策略避免了小任务的并行开销。

### Octree的自底向上构建

Octree将3D空间递归划分为8个子空间，适合均匀分布的点云。GPU上采用自底向上构建策略更高效。

**构建流程：**

1. **体素化与哈希**

将点云映射到最细粒度的体素网格：

```cuda
__global__ void voxelizePoints(
    float3* points,
    int* voxel_keys,    // Morton编码的体素键
    int n_points,
    float voxel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;
    
    float3 p = points[idx];
    // 计算体素坐标
    int3 voxel_coord = make_int3(
        p.x / voxel_size,
        p.y / voxel_size,
        p.z / voxel_size
    );
    
    // Morton编码作为键
    voxel_keys[idx] = mortonEncode3D(voxel_coord);
}
```

2. **并行去重与压缩**

使用推力库的unique操作去除重复体素：

```cuda
thrust::sort(voxel_keys, voxel_keys + n_points);
int n_unique = thrust::unique(voxel_keys, voxel_keys + n_points) - voxel_keys;
```

3. **层次并行聚合**

从叶节点开始，逐层向上构建：

```
Level 0 (叶子): [V000][V001]...[V111] 
                    ↓ 8个合并为1个
Level 1:        [N00][N01]...[N11]
                    ↓ 8个合并为1个  
Level 2:        [N0][N1]
                    ↓
Level 3:        [Root]
```

并行聚合核函数：
```cuda
__global__ void buildOctreeLevel(
    int* curr_level_nodes,   // 当前层节点
    int* next_level_nodes,   // 下一层节点
    int* node_children,      // 子节点索引
    int curr_level_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= curr_level_size) return;
    
    int node_key = curr_level_nodes[idx];
    int parent_key = node_key >> 3;  // 父节点键
    int child_idx = node_key & 7;    // 在父节点中的位置
    
    // 原子操作记录父子关系
    int parent_idx = atomicAdd(&next_level_count, 0);
    node_children[parent_idx * 8 + child_idx] = idx;
    
    // 标记父节点存在
    atomicOr(&next_level_exists[parent_key], 1);
}
```

### Morton编码与空间填充曲线

Morton编码是构建空间索引的关键技术，通过比特交织实现空间到线性的映射。

**高效的GPU Morton编码：**

```cuda
__device__ inline uint32_t expandBits(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

__device__ inline uint32_t mortonEncode3D(int x, int y, int z) {
    return (expandBits(z) << 2) | (expandBits(y) << 1) | expandBits(x);
}
```

这种位操作实现比循环快10倍以上。

**空间填充曲线的应用：**

1. **缓存友好的遍历顺序**
```
线性扫描 → Morton曲线扫描
缓存命中率: 45% → 85%
性能提升: 2.3x
```

2. **快速范围查询**
```
给定空间范围 → Morton码范围
只需比较整数范围，无需3D计算
```

3. **并行基数排序**
```cuda
// Morton码是32位整数，适合GPU基数排序
__global__ void radixSortMortonCodes(
    uint32_t* keys,
    int* values,
    int n
) {
    // 每位4-bit的基数排序
    for (int bit = 0; bit < 32; bit += 4) {
        countingSort(keys, values, n, bit, 16);
    }
}
```

### 平衡树的负载均衡策略

不平衡的空间划分导致严重的负载不均，需要动态平衡策略。

**1. 工作窃取（Work Stealing）**

```cuda
__shared__ int shared_queue[256];
__shared__ int queue_size;

__device__ void processWithStealing() {
    while (true) {
        int work_item;
        
        // 尝试从本地队列获取
        if (threadIdx.x < queue_size) {
            work_item = shared_queue[threadIdx.x];
        } else {
            // 从其他线程块窃取
            work_item = stealFromOtherBlock();
        }
        
        if (work_item < 0) break;
        processNode(work_item);
    }
}
```

**2. 动态并行分配**

根据子树大小动态分配计算资源：

```cuda
__global__ void dynamicTreeBuild(Node* nodes, int* subtree_sizes) {
    int node_id = blockIdx.x;
    int subtree_size = subtree_sizes[node_id];
    
    if (subtree_size > LARGE_THRESHOLD) {
        // 大子树：启动新的kernel
        dim3 blocks(subtree_size / 256);
        buildLargeSubtree<<<blocks, 256>>>(nodes[node_id]);
    } else if (subtree_size > MEDIUM_THRESHOLD) {
        // 中等子树：使用整个线程块
        buildMediumSubtree(nodes[node_id]);
    } else {
        // 小子树：单线程处理
        if (threadIdx.x == 0) {
            buildSmallSubtree(nodes[node_id]);
        }
    }
}
```

**3. 自适应分割策略**

根据点分布选择最优分割：

```cuda
__device__ float computeSplitCost(
    float* points,
    int start, int end,
    int dim, float split_value
) {
    int left_count = 0, right_count = 0;
    float left_extent = 0, right_extent = 0;
    
    // 并行统计左右子集
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        if (points[i * 3 + dim] < split_value) {
            atomicAdd(&left_count, 1);
            atomicMax(&left_extent, fabsf(points[i * 3 + dim] - split_value));
        } else {
            atomicAdd(&right_count, 1);
            atomicMax(&right_extent, fabsf(points[i * 3 + dim] - split_value));
        }
    }
    __syncthreads();
    
    // SAH (Surface Area Heuristic) 成本
    return left_count * left_extent + right_count * right_extent;
}
```

### 性能优化技巧

1. **内存合并优化**
```cuda
// 使用SoA布局提升合并访问
struct KDNode_SoA {
    float* split_values;     // 所有节点的分割值
    int* split_dims;         // 所有节点的分割维度
    int* left_children;      // 左子节点索引
    int* right_children;     // 右子节点索引
};
```

2. **共享内存缓存**
```cuda
__shared__ float cached_points[256 * 3];  // 缓存256个点
__shared__ int cached_indices[256];

// 协作加载到共享内存
if (threadIdx.x < 256) {
    int pid = node_start + threadIdx.x;
    cached_points[threadIdx.x * 3 + 0] = points[pid].x;
    cached_points[threadIdx.x * 3 + 1] = points[pid].y;
    cached_points[threadIdx.x * 3 + 2] = points[pid].z;
    cached_indices[threadIdx.x] = pid;
}
__syncthreads();
```

3. **指令级并行**
```cuda
// 展开循环，增加ILP
#pragma unroll 4
for (int i = 0; i < n; i += 4) {
    float4 coords = reinterpret_cast<float4*>(points)[i/4];
    keys[i+0] = mortonEncode(coords.x);
    keys[i+1] = mortonEncode(coords.y);
    keys[i+2] = mortonEncode(coords.z);
    keys[i+3] = mortonEncode(coords.w);
}
```

典型性能数据（100万点）：
- CPU串行KD-Tree：450ms
- GPU并行KD-Tree：12ms (37x加速)
- GPU Morton Octree：8ms (56x加速)

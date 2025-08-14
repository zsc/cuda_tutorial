# 第18章：大规模点云重建与网格化

本章深入探讨大规模点云数据的GPU加速重建技术。我们将学习如何利用CUDA并行化经典的三维重建算法，包括Poisson重建、Marching Cubes等，并实现高效的网格后处理流程。这些技术在自动驾驶的高精地图构建、具身智能的环境建模等场景中发挥着关键作用。

## 18.1 Poisson重建的GPU加速

### 18.1.1 Poisson重建原理回顾

Poisson表面重建是一种将带法向量的点云转换为水密网格的全局优化方法。其核心思想是将表面重建问题转化为求解Poisson方程：

```
∇²χ = ∇·V
```

其中χ是指示函数（indicator function），V是由点云法向量构建的向量场。该方法的优势在于对噪声鲁棒，能生成光滑的水密表面。

### 18.1.2 八叉树并行构建

Poisson重建的第一步是构建自适应八叉树。在GPU上并行构建八叉树的关键挑战包括：

1. **Morton编码**：使用Z-order曲线将3D坐标映射到1D空间
2. **并行节点分裂**：基于点密度的自适应细分
3. **层级遍历**：自底向上和自顶向下的并行遍历策略

```
Morton编码示意图：
    Y
    ↑
  3 │ 10  11  14  15
  2 │ 08  09  12  13
  1 │ 02  03  06  07
  0 │ 00  01  04  05
    └────────────────→ X
      0   1   2   3

3D扩展：交错x,y,z的二进制位
```

### 18.1.3 稀疏线性系统求解

Poisson方程离散化后形成大规模稀疏线性系统Ax=b。GPU上的高效求解策略：

**共轭梯度法（CG）并行化**：
```
算法流程：
1. r₀ = b - Ax₀
2. p₀ = r₀
3. for k = 0, 1, 2, ...
   αₖ = (rₖᵀrₖ)/(pₖᵀApₖ)
   xₖ₊₁ = xₖ + αₖpₖ
   rₖ₊₁ = rₖ - αₖApₖ
   βₖ = (rₖ₊₁ᵀrₖ₊₁)/(rₖᵀrₖ)
   pₖ₊₁ = rₖ₊₁ + βₖpₖ
```

关键并行操作：
- **稀疏矩阵-向量乘法（SpMV）**：使用CSR格式，每个线程处理一行
- **向量点积**：使用分段归约，结合warp shuffle优化
- **向量更新**：完全并行，带宽受限操作

**多重网格方法（Multigrid）**：
```
V-Cycle示意：
细网格  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         ↘                                     ↗
          ↘ 限制(Restriction)     延拓(Prolongation) ↗
           ↘                                   ↗
中网格      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              ↘                         ↗
               ↘                       ↗
粗网格         ━━━━━━━━━━━━━━━━━━━━━
                    直接求解
```

### 18.1.4 GPU优化策略

**内存访问优化**：
1. **纹理内存缓存八叉树节点**：利用空间局部性
2. **共享内存缓存邻居信息**：减少全局内存访问
3. **向量化加载**：使用float4提高带宽利用率

**计算优化**：
1. **Warp级原语加速归约**：
```cuda
// 使用warp shuffle进行快速归约
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

2. **混合精度计算**：
- 使用FP32进行主要计算
- FP16存储中间结果
- 关键累加使用FP64避免精度损失

3. **动态并行优化八叉树遍历**：
```cuda
__global__ void traverseOctree(Node* nodes, int level) {
    if (level > 0 && threadIdx.x == 0) {
        // 动态启动子网格处理子节点
        dim3 childGrid(8);
        traverseOctree<<<childGrid, 256>>>(
            nodes[blockIdx.x].children, level-1);
    }
}
```

### 18.1.5 性能分析与瓶颈

**典型性能瓶颈**：
1. **不规则内存访问**：八叉树遍历的随机访问模式
2. **负载不均衡**：自适应细分导致的工作量差异
3. **同步开销**：多重网格的层级间同步

**性能指标**（RTX 4090）：
- 100万点重建：~200ms（CPU：~8s）
- 1000万点重建：~2.5s（CPU：~120s）
- 内存带宽利用率：~65%
- SM占用率：~80%

## 18.2 Marching Cubes并行化

### 18.2.1 算法原理与查找表

Marching Cubes是将体数据（如距离场）转换为三角网格的经典算法。核心思想是遍历每个体素（8个顶点的立方体），根据顶点的内外状态生成三角形。

**查找表结构**：
```
立方体顶点编号：
    4 ────── 5
   /│       /│
  / │      / │
 7 ────── 6  │
 │  0 ────│─ 1
 │ /      │ /
 │/       │/
 3 ────── 2

256种配置 = 2^8（每个顶点内/外）
```

**边索引表**：
```cuda
__constant__ int edgeTable[256];  // 每个配置激活的边
__constant__ int triTable[256][16]; // 每个配置的三角形顶点
```

关键优化：将查找表放入常量内存，所有线程访问同一地址时可达到全带宽。

### 18.2.2 并行体素遍历

**两遍扫描策略**：
1. **第一遍：计数**
```cuda
__global__ void countVertices(float* volume, int* vertexCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 获取8个顶点的值
    float v[8];
    loadCubeVertices(idx, volume, v);
    
    // 计算配置索引
    int cubeIndex = 0;
    for(int i = 0; i < 8; i++)
        if(v[i] < isoValue) cubeIndex |= (1 << i);
    
    // 查表获取顶点数
    int nVerts = numVertsTable[cubeIndex];
    vertexCount[idx] = nVerts;
}
```

2. **前缀和分配内存**：
```cuda
// 使用thrust或CUB进行并行前缀和
thrust::exclusive_scan(vertexCount, vertexCount + numVoxels, 
                       vertexOffset);
```

3. **第二遍：生成顶点**

### 18.2.3 顶点生成与索引

**线性插值计算顶点位置**：
```cuda
__device__ float3 vertexInterp(float3 p1, float3 p2, 
                               float v1, float v2, float iso) {
    float t = (iso - v1) / (v2 - v1);
    return p1 + t * (p2 - p1);
}
```

**共享内存优化**：
```cuda
__global__ void generateTriangles() {
    __shared__ float3 vertexCache[12 * BLOCK_SIZE];
    __shared__ int indexCache[15 * BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // 协作加载体素数据到共享内存
    loadToShared(volume, vertexCache);
    __syncthreads();
    
    // 生成三角形
    int cubeIndex = computeCubeIndex(vertexCache, tid);
    if(cubeIndex != 0 && cubeIndex != 255) {
        generateTrianglesForVoxel(cubeIndex, vertexCache, 
                                 indexCache, tid);
    }
}
```

### 18.2.4 流压缩与输出管理

**无锁原子分配**：
```cuda
__device__ int allocateVertices(int count, int* globalCounter) {
    return atomicAdd(globalCounter, count);
}
```

**流压缩优化**：
使用CUB的DeviceSelect进行高效流压缩：
```cuda
// 移除退化三角形
cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
                           d_triangles, d_flags, d_output,
                           d_num_selected, num_triangles);
```

**内存池管理**：
```cuda
class TrianglePool {
    float3* vertices;
    int* indices;
    int* counters;
    
    void resize(int estimatedTriangles) {
        // 预分配1.5倍估计大小避免重分配
        int capacity = estimatedTriangles * 1.5;
        cudaMalloc(&vertices, capacity * 3 * sizeof(float3));
        cudaMalloc(&indices, capacity * 3 * sizeof(int));
    }
};
```

### 18.2.5 扩展：Dual Marching Cubes

Dual Marching Cubes通过在对偶网格上操作，生成更高质量的网格：

**优势**：
- 避免了原始MC的二义性问题
- 生成的网格拓扑更加规则
- 更适合后续的网格优化

**实现要点**：
```cuda
__global__ void dualMarchingCubes() {
    // 在体素中心而非边上生成顶点
    float3 center = computeVoxelCenter(idx);
    
    // 使用Hermite数据（位置+梯度）
    float3 gradient = computeGradient(center);
    
    // QEF（二次误差函数）求解最优顶点位置
    float3 vertex = solveQEF(center, gradient);
}
```

## 18.3 网格简化与优化

### 18.3.1 边折叠算法并行化

边折叠（Edge Collapse）是网格简化的核心操作。GPU并行化的挑战在于处理依赖关系和保持网格一致性。

**基本边折叠操作**：
```
折叠前：        折叠后：
    v1              v_new
   /│\              /|\
  / │ \            / | \
 /  │  \          /  |  \
v3──v2──v4  →   v3───────v4
    
边(v1,v2)折叠为新顶点v_new
```

**并行策略：独立集方法**：
```cuda
__global__ void findIndependentEdges(Edge* edges, bool* canCollapse) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    Edge e = edges[eid];
    
    // 检查边的两个顶点是否被其他边标记
    bool independent = true;
    for(int i = 0; i < e.v1.numNeighbors; i++) {
        if(atomicCAS(&vertexLocked[e.v1.neighbors[i]], 0, 1) != 0) {
            independent = false;
            break;
        }
    }
    
    canCollapse[eid] = independent;
}
```

### 18.3.2 误差度量计算

**二次误差度量（QEM）**：
```cuda
struct Quadric {
    float4 q[10];  // 对称4x4矩阵的10个独立元素
    
    __device__ float computeError(float3 v) {
        // v^T Q v 计算
        float error = 
            v.x * v.x * q[0] + 2 * v.x * v.y * q[1] + 
            2 * v.x * v.z * q[2] + 2 * v.x * q[3] +
            v.y * v.y * q[4] + 2 * v.y * v.z * q[5] + 
            2 * v.y * q[6] + v.z * v.z * q[7] + 
            2 * v.z * q[8] + q[9];
        return error;
    }
    
    __device__ void add(const Quadric& other) {
        #pragma unroll
        for(int i = 0; i < 10; i++)
            q[i] += other.q[i];
    }
};
```

**最优顶点位置求解**：
```cuda
__device__ float3 computeOptimalPosition(const Quadric& q) {
    // 求解线性系统 ∇error = 0
    float3x3 A = extractMatrix(q);
    float3 b = extractVector(q);
    
    // 使用Cramer法则（3x3矩阵）
    float det = determinant(A);
    if(abs(det) > 1e-6) {
        return solveLinearSystem(A, b, det);
    } else {
        // 退化情况：使用中点
        return (v1 + v2) * 0.5f;
    }
}
```

### 18.3.3 优先队列管理

**GPU友好的优先队列**：
```cuda
class GPUPriorityQueue {
    struct HeapNode {
        float cost;
        int edgeId;
    };
    
    HeapNode* heap;
    int* size;
    
    __device__ void insertBatch(HeapNode* items, int count) {
        // 批量插入+并行堆化
        int oldSize = atomicAdd(size, count);
        
        // 拷贝到堆尾
        memcpy(&heap[oldSize], items, count * sizeof(HeapNode));
        
        // 并行上浮
        parallelHeapify(oldSize, oldSize + count);
    }
    
    __device__ void parallelHeapify(int start, int end) {
        // 使用分段并行堆化算法
        for(int level = __log2f(end); level >= 0; level--) {
            int stride = 1 << level;
            if(threadIdx.x < stride) {
                int idx = start + threadIdx.x * 2;
                heapifyDown(idx);
            }
            __syncthreads();
        }
    }
};
```

### 18.3.4 拓扑保持策略

**流形保持检查**：
```cuda
__device__ bool preservesManifold(Edge e) {
    // 1. 链接条件检查
    int sharedNeighbors = 0;
    for(int i = 0; i < e.v1.valence; i++) {
        for(int j = 0; j < e.v2.valence; j++) {
            if(e.v1.neighbors[i] == e.v2.neighbors[j])
                sharedNeighbors++;
        }
    }
    
    // 边界边应有1个共享邻居，内部边应有2个
    bool isBoundary = e.v1.boundary || e.v2.boundary;
    return sharedNeighbors == (isBoundary ? 1 : 2);
}

__device__ bool preventsFoldover(Edge e, float3 newPos) {
    // 检查折叠后是否产生翻转的三角形
    for(int i = 0; i < e.v1.numFaces; i++) {
        Face f = faces[e.v1.faces[i]];
        float3 normal_before = computeNormal(f);
        
        // 模拟折叠后的法向
        float3 normal_after = computeNormalAfterCollapse(f, newPos);
        
        if(dot(normal_before, normal_after) < 0.1f)  // 阈值
            return false;
    }
    return true;
}
```

### 18.3.5 GPU友好的数据结构

**半边数据结构的GPU实现**：
```cuda
struct HalfEdge {
    int vertex;      // 指向的顶点
    int opposite;    // 对边
    int next;        // 下一条半边
    int face;        // 所属面
};

struct CompactMesh {
    // SOA布局提高内存合并
    float3* positions;
    float3* normals;
    HalfEdge* halfedges;
    int* vertexEdges;    // 每个顶点的一条出边
    
    __device__ void collapseEdge(int edgeId) {
        HalfEdge he = halfedges[edgeId];
        HalfEdge opp = halfedges[he.opposite];
        
        // 更新拓扑连接
        atomicExch(&halfedges[he.next].opposite, opp.next);
        atomicExch(&halfedges[opp.next].opposite, he.next);
        
        // 标记删除的元素
        atomicExch(&halfedges[edgeId].vertex, -1);
    }
};
```

**内存池与压缩**：
```cuda
__global__ void compactMesh(CompactMesh mesh, int* validFlags) {
    // 使用流压缩移除已删除的元素
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < mesh.numVertices) {
        if(mesh.positions[tid].x != DELETED_MARKER) {
            validFlags[tid] = 1;
        } else {
            validFlags[tid] = 0;
        }
    }
}

// 使用CUB进行流压缩
cub::DeviceSelect::Flagged(...)
```

## 18.4 纹理映射与混合

### 18.4.1 UV参数化

UV参数化将3D网格表面映射到2D纹理空间。GPU加速的关键在于并行化优化过程。

**最小化畸变的参数化**：
```cuda
__global__ void computeParameterization(Vertex* vertices, Face* faces,
                                       float2* uvCoords) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // LSCM（Least Squares Conformal Maps）方法
    float2 gradient = float2(0, 0);
    
    // 计算共形能量梯度
    for(int i = 0; i < vertices[vid].numFaces; i++) {
        Face f = faces[vertices[vid].faces[i]];
        float2 localGrad = computeConformalGradient(f, vid);
        gradient += localGrad;
    }
    
    // 梯度下降更新
    uvCoords[vid] -= 0.01f * gradient;
}

__device__ float2 computeConformalGradient(Face f, int vid) {
    // 计算角度畸变
    float3 p0 = vertices[f.v0].pos;
    float3 p1 = vertices[f.v1].pos;
    float3 p2 = vertices[f.v2].pos;
    
    float2 uv0 = uvCoords[f.v0];
    float2 uv1 = uvCoords[f.v1];
    float2 uv2 = uvCoords[f.v2];
    
    // 共形映射的Jacobian矩阵
    float2x2 J = computeJacobian(p0, p1, p2, uv0, uv1, uv2);
    
    // 最小化 ||J - cR||²，其中R是旋转矩阵
    return computeGradient(J);
}
```

### 18.4.2 多视图纹理融合

从多个视角的图像中提取并融合纹理信息：

**视图选择与权重计算**：
```cuda
__global__ void selectBestViews(Face* faces, Camera* cameras, 
                                int* bestViews, float* weights) {
    int fid = blockIdx.x * blockDim.x + threadIdx.x;
    Face f = faces[fid];
    
    float3 faceNormal = computeFaceNormal(f);
    float3 faceCenter = computeFaceCenter(f);
    
    float maxScore = -1.0f;
    int bestView = -1;
    
    // 评估每个相机视角
    for(int c = 0; c < numCameras; c++) {
        float3 viewDir = normalize(cameras[c].position - faceCenter);
        
        // 视角质量评分
        float angleCos = dot(viewDir, faceNormal);
        float distance = length(cameras[c].position - faceCenter);
        float resolution = cameras[c].focalLength / distance;
        
        float score = angleCos * resolution;
        
        if(score > maxScore) {
            maxScore = score;
            bestView = c;
        }
    }
    
    bestViews[fid] = bestView;
    weights[fid] = maxScore;
}
```

**纹理采样与混合**：
```cuda
__global__ void blendTextures(float4* outputTexture, 
                              cudaTextureObject_t* inputTextures,
                              float* blendWeights) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float4 color = float4(0, 0, 0, 0);
    float totalWeight = 0;
    
    // 多视图加权混合
    for(int v = 0; v < numViews; v++) {
        float2 uv = projectToView(x, y, v);
        
        if(isValidProjection(uv)) {
            float4 sample = tex2D<float4>(inputTextures[v], uv.x, uv.y);
            float w = blendWeights[v] * computeSeamWeight(uv);
            
            color += sample * w;
            totalWeight += w;
        }
    }
    
    outputTexture[y * width + x] = color / totalWeight;
}
```

### 18.4.3 接缝消除技术

**Poisson图像编辑**：
```cuda
__global__ void poissonBlending(float4* texture, int* seamMask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(seamMask[idx]) {
        // 在接缝处求解Poisson方程
        float4 laplacian = computeLaplacian(texture, idx);
        float4 gradient = computeSeamGradient(texture, idx);
        
        // Jacobi迭代
        texture[idx] = 0.25f * (
            texture[idx - 1] + texture[idx + 1] +
            texture[idx - width] + texture[idx + width] - gradient
        );
    }
}
```

**图割优化**：
```cuda
__global__ void graphCutSeam(int* labels, float* dataCost, 
                             float* smoothCost) {
    // α-expansion算法的并行实现
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int currentLabel = labels[pixel];
    
    // 计算切换标签的代价
    float costKeep = dataCost[pixel * numLabels + currentLabel];
    float costSwitch = FLT_MAX;
    
    for(int l = 0; l < numLabels; l++) {
        if(l != currentLabel) {
            float cost = dataCost[pixel * numLabels + l];
            
            // 添加平滑项
            for(int n = 0; n < 4; n++) {
                int neighbor = getNeighbor(pixel, n);
                if(neighbor >= 0) {
                    cost += smoothCost[abs(l - labels[neighbor])];
                }
            }
            
            costSwitch = min(costSwitch, cost);
        }
    }
    
    // 更新标签
    if(costSwitch < costKeep) {
        labels[pixel] = argmin(costSwitch);
    }
}
```

### 18.4.4 纹理图集生成

**矩形打包算法**：
```cuda
struct TextureChart {
    int id;
    int width, height;
    int x, y;  // 在图集中的位置
};

__global__ void packCharts(TextureChart* charts, int* atlasSize) {
    // 使用扫描线算法并行打包
    __shared__ int scanline[MAX_ATLAS_WIDTH];
    
    int tid = threadIdx.x;
    int chartId = blockIdx.x;
    
    if(tid == 0) {
        // 找到最低的可用位置
        int bestY = MAX_ATLAS_HEIGHT;
        int bestX = 0;
        
        for(int x = 0; x <= *atlasSize - charts[chartId].width; x++) {
            int maxY = 0;
            for(int w = 0; w < charts[chartId].width; w++) {
                maxY = max(maxY, scanline[x + w]);
            }
            
            if(maxY < bestY) {
                bestY = maxY;
                bestX = x;
            }
        }
        
        charts[chartId].x = bestX;
        charts[chartId].y = bestY;
        
        // 更新扫描线
        for(int w = 0; w < charts[chartId].width; w++) {
            scanline[bestX + w] = bestY + charts[chartId].height;
        }
    }
}
```

### 18.4.5 GPU纹理压缩

**实时DXT压缩**：
```cuda
__global__ void compressDXT1(uchar4* input, uint2* output) {
    __shared__ float3 colors[16];
    
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int tid = threadIdx.x;
    
    // 加载4x4块到共享内存
    if(tid < 16) {
        int x = blockX * 4 + tid % 4;
        int y = blockY * 4 + tid / 4;
        uchar4 pixel = input[y * width + x];
        colors[tid] = make_float3(pixel.x, pixel.y, pixel.z) / 255.0f;
    }
    __syncthreads();
    
    if(tid == 0) {
        // PCA找到主轴
        float3 mean = computeMean(colors);
        float3 axis = computePrincipalAxis(colors, mean);
        
        // 投影到主轴找到端点
        float minProj = FLT_MAX, maxProj = -FLT_MAX;
        int minIdx = 0, maxIdx = 0;
        
        for(int i = 0; i < 16; i++) {
            float proj = dot(colors[i] - mean, axis);
            if(proj < minProj) { minProj = proj; minIdx = i; }
            if(proj > maxProj) { maxProj = proj; maxIdx = i; }
        }
        
        // 生成调色板
        float3 c0 = colors[maxIdx];
        float3 c1 = colors[minIdx];
        
        // 编码索引
        uint indices = 0;
        for(int i = 0; i < 16; i++) {
            int idx = findClosestColor(colors[i], c0, c1);
            indices |= (idx << (i * 2));
        }
        
        // 打包输出
        output[blockY * gridDim.x + blockX] = 
            make_uint2(packColor565(c0, c1), indices);
    }
}
```

## 18.5 LOD（细节层次）生成

### 18.5.1 LOD策略选择

不同的LOD策略适用于不同场景。GPU并行化需要考虑各策略的特点：

**离散LOD vs 连续LOD**：
```cuda
enum LODStrategy {
    DISCRETE_LOD,      // 预生成固定几个级别
    CONTINUOUS_LOD,    // 实时调整细节
    HLOD,             // 层次化LOD，用于大规模场景
    PROGRESSIVE_MESH  // 渐进式网格
};

__global__ void selectLODLevel(Object* objects, Camera camera,
                               int* lodLevels) {
    int oid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float3 objPos = objects[oid].center;
    float distance = length(camera.position - objPos);
    float screenSize = objects[oid].radius / distance * camera.fov;
    
    // 基于屏幕覆盖率选择LOD
    if(screenSize > 0.5f) lodLevels[oid] = 0;      // 最高细节
    else if(screenSize > 0.1f) lodLevels[oid] = 1; // 中等细节
    else if(screenSize > 0.02f) lodLevels[oid] = 2; // 低细节
    else lodLevels[oid] = 3;                        // 最低细节
}
```

### 18.5.2 渐进网格构建

**边折叠序列记录**：
```cuda
struct CollapseRecord {
    int v0, v1;           // 折叠的边
    float3 newPos;        // 新顶点位置
    int affectedFaces[8]; // 受影响的面
    int numAffected;
};

__global__ void buildProgressiveMesh(Mesh* mesh, 
                                     CollapseRecord* records) {
    // 并行评估所有边的折叠代价
    __shared__ float costs[256];
    __shared__ int edges[256];
    
    int tid = threadIdx.x;
    int eid = blockIdx.x * blockDim.x + tid;
    
    costs[tid] = computeCollapseCost(mesh->edges[eid]);
    edges[tid] = eid;
    __syncthreads();
    
    // 块内找最小代价
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && costs[tid] > costs[tid + s]) {
            costs[tid] = costs[tid + s];
            edges[tid] = edges[tid + s];
        }
        __syncthreads();
    }
    
    // 记录折叠信息
    if(tid == 0) {
        int bestEdge = edges[0];
        records[blockIdx.x] = createCollapseRecord(mesh, bestEdge);
    }
}
```

**渐进传输优化**：
```cuda
struct ProgressiveStream {
    // 基础网格
    float3* baseVertices;
    int3* baseFaces;
    int baseVertCount;
    
    // 细化记录（压缩存储）
    struct RefinementOp {
        uint16_t parentVertex;    // 父顶点索引
        int8_t deltaX, deltaY, deltaZ; // 位置增量（量化）
        uint8_t faceConfig;       // 面拆分配置
    } *refinements;
    
    __device__ void refineToLevel(int targetLevel) {
        for(int level = currentLevel; level < targetLevel; level++) {
            applyRefinement(refinements[level]);
        }
    }
};
```

### 18.5.3 视觉质量度量

**屏幕空间误差**：
```cuda
__device__ float computeScreenSpaceError(Triangle tri, Camera cam) {
    // 投影三角形到屏幕空间
    float2 p0 = projectToScreen(tri.v0, cam);
    float2 p1 = projectToScreen(tri.v1, cam);
    float2 p2 = projectToScreen(tri.v2, cam);
    
    // 计算屏幕空间面积
    float screenArea = abs((p1.x - p0.x) * (p2.y - p0.y) - 
                          (p2.x - p0.x) * (p1.y - p0.y)) * 0.5f;
    
    // 计算几何误差
    float geometricError = computeHausdorffDistance(tri, originalMesh);
    
    // 组合度量
    return screenArea * geometricError;
}
```

**感知质量度量**：
```cuda
__global__ void computePerceptualError(Mesh* lod, Mesh* original,
                                       float* errors) {
    int fid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算法向量差异
    float3 lodNormal = computeFaceNormal(lod->faces[fid]);
    float3 origNormal = findClosestNormal(original, lod->faces[fid]);
    float normalError = 1.0f - dot(lodNormal, origNormal);
    
    // 计算轮廓保持度
    float silhouetteError = computeSilhouetteDeviation(lod, original, fid);
    
    // 计算纹理坐标畸变
    float uvDistortion = computeUVDistortion(lod->faces[fid]);
    
    // 加权组合
    errors[fid] = 0.4f * normalError + 
                  0.4f * silhouetteError + 
                  0.2f * uvDistortion;
}
```

### 18.5.4 实时LOD切换

**无缝过渡技术**：
```cuda
__global__ void blendLODs(Vertex* lod0, Vertex* lod1, 
                          float blendFactor, Vertex* output) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 找到对应顶点
    int vid1 = findCorrespondingVertex(vid, lod0, lod1);
    
    if(vid1 >= 0) {
        // 线性插值位置和法向
        output[vid].position = lerp(lod0[vid].position, 
                                   lod1[vid1].position, blendFactor);
        output[vid].normal = normalize(lerp(lod0[vid].normal,
                                           lod1[vid1].normal, blendFactor));
    } else {
        // 渐隐处理孤立顶点
        output[vid] = lod0[vid];
        output[vid].alpha = 1.0f - blendFactor;
    }
}
```

**地理裁剪LOD（Geomorphing）**：
```cuda
__global__ void geomorphLOD(Vertex* vertices, float* morphWeights,
                            float3* targetPositions) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float t = morphWeights[vid];
    
    // 平滑过渡到目标位置
    vertices[vid].position = vertices[vid].position * (1 - t) + 
                             targetPositions[vid] * t;
    
    // 重新计算法向量
    if(t > 0.01f) {
        recomputeVertexNormal(&vertices[vid]);
    }
}
```

### 18.5.5 内存管理优化

**LOD缓存策略**：
```cuda
class LODCache {
    struct CacheEntry {
        int objectId;
        int lodLevel;
        void* gpuData;
        int timestamp;
        int refCount;
    };
    
    CacheEntry* entries;
    int* lruList;
    
    __device__ void* fetchLOD(int objId, int level) {
        int hash = (objId * 31 + level) % CACHE_SIZE;
        
        // 检查缓存命中
        if(entries[hash].objectId == objId && 
           entries[hash].lodLevel == level) {
            atomicAdd(&entries[hash].refCount, 1);
            updateLRU(hash);
            return entries[hash].gpuData;
        }
        
        // 缓存未命中，触发异步加载
        requestAsyncLoad(objId, level, hash);
        return nullptr;
    }
    
    __device__ void evictLRU() {
        int victim = lruList[CACHE_SIZE - 1];
        if(entries[victim].refCount == 0) {
            freeGPUMemory(entries[victim].gpuData);
            entries[victim].objectId = -1;
        }
    }
};
```

**流式LOD加载**：
```cuda
__global__ void streamLODData(LODRequest* requests, int numRequests) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(rid < numRequests) {
        LODRequest req = requests[rid];
        
        // 计算需要的内存
        int vertexCount = getVertexCount(req.objId, req.level);
        int faceCount = getFaceCount(req.objId, req.level);
        
        // 分配GPU内存
        void* gpuMem = allocateFromPool(vertexCount, faceCount);
        
        // 异步传输数据
        cudaMemcpyAsync(gpuMem, 
                       getLODData(req.objId, req.level),
                       getDataSize(vertexCount, faceCount),
                       cudaMemcpyHostToDevice,
                       req.stream);
        
        // 更新缓存
        updateCache(req.objId, req.level, gpuMem);
    }
}
```

## 本章小结

本章深入探讨了大规模点云重建与网格化的GPU加速技术。我们学习了：

1. **Poisson重建加速**：通过八叉树并行构建、稀疏线性系统的GPU求解，实现了40倍以上的加速比
2. **Marching Cubes并行化**：使用两遍扫描、流压缩和查找表优化，高效生成三角网格
3. **网格简化优化**：实现了GPU友好的边折叠算法，包括QEM误差度量和拓扑保持
4. **纹理映射技术**：掌握了UV参数化、多视图融合、接缝消除等关键技术
5. **LOD系统设计**：构建了完整的多细节层次系统，支持渐进传输和实时切换

关键性能指标：
- Poisson重建：1000万点 ~2.5秒（RTX 4090）
- Marching Cubes：512³体素 ~50ms
- 网格简化：100万面简化到10万面 ~200ms
- LOD切换：<1ms延迟，支持上千个对象

## 练习题

### 基础题

1. **八叉树构建优化**
   实现一个并行的自适应八叉树构建算法，要求支持不均匀点云分布。
   
   **提示**：使用Morton编码进行空间排序，利用原子操作处理节点分裂。
   
   <details>
   <summary>参考答案</summary>
   
   使用Z-order曲线将3D坐标映射到1D，先对点进行排序，然后并行构建节点。关键是使用原子操作atomicMax来确定每个节点的细分深度，避免竞争条件。采用自底向上的构建策略，先创建叶节点，再合并生成父节点。
   </details>

2. **Marching Cubes查找表生成**
   编写代码自动生成Marching Cubes的256种配置查找表。
   
   **提示**：利用对称性减少独立配置数量。
   
   <details>
   <summary>参考答案</summary>
   
   256种配置实际只有15种基本情况，其余通过旋转和镜像变换得到。首先生成15种基本配置的三角形模板，然后通过24种对称变换（8个顶点的置换）生成完整查找表。使用位操作快速计算配置索引。
   </details>

3. **简单QEM实现**
   实现基本的二次误差度量计算，用于评估边折叠代价。
   
   **提示**：QEM可以表示为4x4对称矩阵。
   
   <details>
   <summary>参考答案</summary>
   
   每个顶点关联一个4x4的二次型矩阵Q，表示到相邻平面的距离平方和。边折叠时，新顶点的Q矩阵是两个端点Q矩阵之和。折叠代价通过求解线性系统∇(v^T Q v) = 0得到最优位置，再计算该位置的误差值。
   </details>

### 挑战题

4. **并行Poisson求解器**
   实现一个高效的GPU多重网格求解器，用于Poisson重建。
   
   **提示**：使用红黑Gauss-Seidel作为平滑器。
   
   <details>
   <summary>参考答案</summary>
   
   实现V-cycle多重网格，每层使用红黑着色的Gauss-Seidel迭代。红黑着色允许并行更新同色节点。限制和延拓操作使用三线性插值。粗网格直接求解使用共轭梯度法。关键优化包括：纹理内存缓存系数矩阵，共享内存缓存模板值，使用shuffle指令加速归约。
   </details>

5. **实时网格简化系统**
   设计一个支持视点相关简化的实时系统，能根据观察角度动态调整网格细节。
   
   **提示**：结合屏幕空间误差和几何误差。
   
   <details>
   <summary>参考答案</summary>
   
   使用双缓冲机制，一个用于渲染，一个用于简化。根据视点计算每个面的重要性分数，优先简化背向面和远处细节。使用GPU优先队列管理边折叠操作，每帧限制操作数量保证实时性。实现增量式更新，避免全局重建。
   </details>

6. **无缝纹理图集生成**
   实现一个GPU加速的纹理图集打包算法，最小化纹理浪费并消除接缝。
   
   **提示**：使用最大矩形算法和Poisson混合。
   
   <details>
   <summary>参考答案</summary>
   
   先使用并行的最大矩形算法找到最优打包方案，每个线程负责一个chart的放置。使用扫描线算法维护可用空间。对于接缝，在chart边界扩展几个像素，使用Poisson方程混合相邻chart的颜色。最后使用mipmap金字塔填充未使用区域，避免采样错误。
   </details>

7. **HLOD自动生成**
   实现层次化LOD（HLOD）系统，自动合并远处的多个对象。
   
   **提示**：使用空间聚类和代理几何体。
   
   <details>
   <summary>参考答案</summary>
   
   使用八叉树或KD树进行空间划分，每个节点存储子对象的简化版本。当节点覆盖的屏幕空间小于阈值时，用单个代理几何体替换所有子对象。代理几何体通过体素化和Marching Cubes生成。使用impostor技术为极远距离的对象群生成billboard。
   </details>

8. **GPU加速的网格修复**
   设计一个自动修复网格拓扑错误的GPU算法，处理孔洞、自相交等问题。
   
   **提示**：使用体素化检测和修复拓扑问题。
   
   <details>
   <summary>参考答案</summary>
   
   首先体素化网格得到水密的距离场，然后用Marching Cubes重新提取表面。对于小孔洞，使用advancing front方法并行填充。对于自相交，通过射线投射检测并使用BSP树分割相交面。使用并行的连通分量标记算法移除孤立的小部件。最后运行Laplacian平滑改善网格质量。
   </details>

## 常见陷阱与错误

1. **八叉树遍历的内存访问模式**
   - 错误：随机访问导致缓存未命中率高
   - 正确：使用Morton顺序改善空间局部性

2. **Marching Cubes的重复顶点**
   - 错误：每个体素独立生成顶点，造成大量重复
   - 正确：使用哈希表或边索引去重

3. **网格简化的拓扑破坏**
   - 错误：盲目折叠边导致非流形结构
   - 正确：实施链接条件检查

4. **纹理接缝的可见性**
   - 错误：简单的线性混合在光照下仍可见
   - 正确：使用Poisson混合或图割优化

5. **LOD切换的突变**
   - 错误：直接切换造成明显跳变
   - 正确：使用geomorphing或alpha混合

6. **内存泄漏和碎片化**
   - 错误：频繁的cudaMalloc/cudaFree
   - 正确：使用内存池和压缩策略

## 最佳实践检查清单

### 算法选择
- [ ] 根据输入数据特点选择合适的重建算法
- [ ] 评估质量vs速度的权衡
- [ ] 考虑内存限制选择合适的数据结构

### 性能优化
- [ ] 使用纹理内存缓存只读数据
- [ ] 合并内存访问，使用向量化load/store
- [ ] 平衡寄存器使用和占用率
- [ ] 实现多流并发和异步操作

### 质量保证
- [ ] 实施拓扑一致性检查
- [ ] 验证法向量方向正确性
- [ ] 测试极端情况（退化三角形、孤立顶点等）
- [ ] 确保数值稳定性

### 内存管理
- [ ] 实现内存池避免碎片化
- [ ] 使用流压缩移除无效元素
- [ ] 监控GPU内存使用情况
- [ ] 实施内存预算控制

### 系统集成
- [ ] 设计清晰的API接口
- [ ] 提供同步和异步两种模式
- [ ] 实现错误恢复机制
- [ ] 支持增量式更新
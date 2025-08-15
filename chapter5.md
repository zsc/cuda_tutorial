# 第5章：寄存器优化与常量内存

寄存器是GPU中最快的存储层次，每个线程可以独占使用一定数量的寄存器。与共享内存需要显式管理不同，寄存器由编译器自动分配，但理解其分配机制和优化策略对于实现极致性能至关重要。本章将深入探讨寄存器优化技术、常量内存的高效使用，以及通过联合体和位操作实现的底层优化。我们将通过一个完整的卷积优化案例，展示如何在寄存器级别实现性能突破。

## 5.1 寄存器分配与压力分析

寄存器是GPU计算的核心资源，直接影响内核的性能上限。不同于CPU的寄存器重命名机制，GPU寄存器是静态分配的，这既是挑战也是机遇——通过精确控制寄存器使用，我们可以实现极致的性能优化。

### 5.1.1 寄存器架构概述

现代GPU的寄存器文件是一个巨大的SRAM阵列，每个SM拥有数万个32位寄存器。以Ampere架构为例：

```
SM寄存器文件结构：
┌─────────────────────────────────────┐
│         Register File (64KB)         │
├─────────────────────────────────────┤
│  65536个32位寄存器 / SM              │
│  最多256个寄存器 / 线程              │
│  支持同时2048个活跃线程             │
└─────────────────────────────────────┘

寄存器组织方式：
- 物理视图：统一的寄存器池
- 逻辑视图：每个线程私有的寄存器集合
- 访问延迟：1个时钟周期（无bank conflict）
```

寄存器的关键特性：
1. **零延迟访问**：寄存器访问不需要等待，是最快的存储
2. **线程私有**：每个线程独占其分配的寄存器，无需同步
3. **静态分配**：编译时决定每个线程使用的寄存器数量
4. **有限资源**：寄存器数量直接限制活跃线程数（占用率）

### 5.1.2 寄存器分配机制

NVCC编译器负责将CUDA代码中的变量映射到寄存器：

```cuda
// 寄存器分配示例
__global__ void kernel() {
    // 自动变量通常分配到寄存器
    float a = 1.0f;        // 1个寄存器
    double b = 2.0;        // 2个寄存器（64位）
    int c[4];              // 4个寄存器（如果不溢出）
    
    // 寄存器压力来源：
    float matrix[8][8];    // 64个寄存器！可能溢出到local memory
}
```

编译器的寄存器分配策略：
1. **活跃变量分析**：只为同时活跃的变量分配寄存器
2. **寄存器着色**：通过图着色算法最小化寄存器使用
3. **生命周期分析**：变量超出作用域后释放寄存器
4. **寄存器合并**：不同生命周期的变量可共享寄存器

查看寄存器使用情况：
```bash
nvcc -Xptxas="-v" kernel.cu
# 输出: Used 48 registers, 360 bytes smem
```

### 5.1.3 寄存器压力的影响

寄存器压力是指内核需要的寄存器数量接近或超过硬件限制：

```
占用率计算公式：
活跃warp数 = min(
    SM最大warp数,
    SM寄存器总数 / (每线程寄存器数 × 32)
)

例：A100 GPU
- SM寄存器数：65536
- 最大warp数：64
- 若每线程用32个寄存器：65536/(32×32) = 64 warps ✓
- 若每线程用64个寄存器：65536/(64×32) = 32 warps ✗
```

寄存器压力的影响链：
```
高寄存器使用 → 低占用率 → 延迟隐藏能力下降 → 性能下降
     ↓              ↓              ↓                ↓
 溢出到L1/L2   活跃线程少    stall增加      带宽利用率低
```

优化寄存器压力的技术：
1. **减少活跃变量**：及时释放不再使用的变量
2. **循环分解**：将大循环拆分为多个小循环
3. **函数内联控制**：避免过度内联导致寄存器膨胀
4. **寄存器数量限制**：使用`__launch_bounds__`

### 5.1.4 寄存器溢出（Register Spilling）

当寄存器需求超过硬件限制时，编译器将部分变量溢出到本地内存（L1缓存）：

```cuda
// 寄存器溢出示例
__global__ void spillExample() {
    // 大数组很可能溢出
    float largeArray[256];  // 需要256个寄存器！
    
    // 编译器会将部分元素放到local memory
    // 访问时产生 LDL/STL 指令（Local Load/Store）
    for(int i = 0; i < 256; i++) {
        largeArray[i] = i * 2.0f;  // STL指令
    }
    
    // 读取时从local memory加载
    float sum = 0;
    for(int i = 0; i < 256; i++) {
        sum += largeArray[i];      // LDL指令
    }
}
```

检测寄存器溢出：
```bash
# 编译时查看
nvcc -Xptxas="-v" kernel.cu
# 输出包含: 0 bytes spill stores, 0 bytes spill loads

# 运行时使用Nsight Compute
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum kernel
```

溢出的性能影响：
- **延迟增加**：从~1周期增加到~30周期
- **带宽消耗**：占用L1缓存带宽
- **缓存污染**：影响其他数据的缓存命中率

避免溢出的策略：
```cuda
// 策略1：使用共享内存替代大数组
__global__ void noSpill_v1() {
    __shared__ float sharedArray[256];
    // 共享内存不占用寄存器
}

// 策略2：分块处理
__global__ void noSpill_v2() {
    const int CHUNK = 32;
    float chunk[CHUNK];  // 只需32个寄存器
    for(int c = 0; c < 8; c++) {
        // 处理每个块...
    }
}

// 策略3：限制寄存器数量
__global__ __launch_bounds__(256, 4) 
void noSpill_v3() {
    // 编译器会更激进地优化寄存器使用
}
```

### 5.1.5 占用率与寄存器的权衡

占用率和寄存器使用量存在天然的矛盾，需要找到最佳平衡点：

```
性能模型：
Performance = f(占用率, ILP, 寄存器复用率)

其中：
- 高占用率 → 更好的延迟隐藏
- 多寄存器 → 更好的数据复用，减少内存访问
- 需要实验找到最优点
```

寄存器-占用率权衡分析：
```cuda
// 版本1：高占用率，低寄存器
__global__ void high_occupancy() {
    // 使用24个寄存器，100%占用率
    float a = blockIdx.x;
    float b = threadIdx.x;
    float c = a + b;
    // 频繁访问global memory
    g_data[threadIdx.x] = c;
}

// 版本2：低占用率，高寄存器
__global__ void low_occupancy() {
    // 使用64个寄存器，50%占用率
    float reg[16];  // 寄存器数组
    
    // 数据预取到寄存器
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        reg[i] = g_data[threadIdx.x * 16 + i];
    }
    
    // 在寄存器中计算，避免内存访问
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        reg[i] = reg[i] * 2.0f + 1.0f;
    }
    
    // 写回
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        g_data[threadIdx.x * 16 + i] = reg[i];
    }
}
```

自动驾驶场景的权衡示例：
```cuda
// 激光雷达点云处理：选择高寄存器使用
__global__ void pointcloud_processing() {
    // 每个线程处理一个点，需要大量中间变量
    float point[3];      // 点坐标
    float transform[16]; // 变换矩阵
    float features[8];   // 特征向量
    
    // 使用80+寄存器，但避免了重复内存访问
    // 对于计算密集型任务，这种权衡是值得的
}

// 图像预处理：选择高占用率
__global__ void image_preprocessing() {
    // 简单的像素级操作
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 只需少量寄存器
    uchar4 pixel = tex2D<uchar4>(tex, x, y);
    float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
    output[idx] = gray;
    
    // 内存访问是瓶颈，需要高占用率隐藏延迟
}
```

优化决策流程：
```
1. 分析内核特征
   ├─ 计算密集型 → 优先寄存器复用
   └─ 内存密集型 → 优先高占用率

2. 性能测试
   ├─ 测试不同寄存器限制 (-maxrregcount)
   ├─ 测试不同block大小
   └─ 使用Nsight Compute分析

3. 迭代优化
   ├─ 监控寄存器溢出
   ├─ 监控stall原因
   └─ 调整寄存器使用策略
```

## 5.2 寄存器重用与别名技术

寄存器重用是高性能CUDA编程的核心技术。通过巧妙的数据布局和计算顺序安排，我们可以最大化每个寄存器的利用率，减少内存访问次数，显著提升计算密度。

### 5.2.1 寄存器重用模式

寄存器重用的本质是让同一个数据在寄存器中停留尽可能长的时间，服务于多次计算：

```cuda
// 模式1：时间重用 - 同一数据多次使用
__global__ void temporal_reuse() {
    float data = global_mem[idx];  // 加载一次
    
    float result1 = data * 2.0f;   // 第一次使用
    float result2 = data + 3.0f;   // 第二次使用
    float result3 = sqrt(data);    // 第三次使用
    
    // data在寄存器中被重用3次，避免3次内存访问
}

// 模式2：空间重用 - 邻近数据共享计算
__global__ void spatial_reuse() {
    // 滑动窗口：3个寄存器缓存邻近数据
    float left = global_mem[idx-1];
    float center = global_mem[idx];
    float right = global_mem[idx+1];
    
    // 1D卷积计算
    float result = 0.25f * left + 0.5f * center + 0.25f * right;
    
    // 滑动窗口更新（寄存器轮转）
    left = center;
    center = right;
    right = global_mem[idx+2];
}

// 模式3：生产者-消费者重用
__global__ void producer_consumer() {
    // 寄存器作为小型FIFO
    float reg0, reg1, reg2, reg3;
    
    // 流水线填充
    reg0 = compute_stage1(input[0]);
    reg1 = compute_stage1(input[1]);
    
    // 稳态处理
    for(int i = 2; i < N; i++) {
        reg2 = compute_stage1(input[i]);
        reg3 = compute_stage2(reg0, reg1, reg2);
        output[i-2] = compute_stage3(reg3);
        
        // 寄存器轮转
        reg0 = reg1;
        reg1 = reg2;
    }
}
```

矩阵乘法中的寄存器重用策略：
```cuda
// 寄存器阻塞：每个线程计算 TM×TN 的输出块
template<int TM, int TN>
__global__ void gemm_register_blocking() {
    // 为结果矩阵分配寄存器
    float c[TM][TN] = {0};
    
    // 为输入数据分配寄存器
    float a_reg[TM];
    float b_reg[TN];
    
    for(int k = 0; k < K; k++) {
        // 加载A的一列到寄存器
        #pragma unroll
        for(int i = 0; i < TM; i++) {
            a_reg[i] = A[row + i][k];
        }
        
        // 加载B的一行到寄存器
        #pragma unroll
        for(int j = 0; j < TN; j++) {
            b_reg[j] = B[k][col + j];
        }
        
        // 外积更新：最大化寄存器重用
        #pragma unroll
        for(int i = 0; i < TM; i++) {
            #pragma unroll
            for(int j = 0; j < TN; j++) {
                c[i][j] += a_reg[i] * b_reg[j];
                // 每个a_reg[i]被重用TN次
                // 每个b_reg[j]被重用TM次
            }
        }
    }
}
```

### 5.2.2 循环展开与寄存器调度

循环展开是控制寄存器分配的关键技术，通过展开可以暴露更多的并行性和重用机会：

```cuda
// 基础版本：编译器可能无法优化
__global__ void basic_loop() {
    float sum = 0;
    for(int i = 0; i < 256; i++) {
        sum += data[i] * weight[i];
    }
}

// 手动展开版本：显式寄存器调度
__global__ void unrolled_loop() {
    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    
    // 4路展开，4个独立的累加器
    for(int i = 0; i < 256; i += 4) {
        // 批量加载到寄存器
        float d0 = data[i+0], w0 = weight[i+0];
        float d1 = data[i+1], w1 = weight[i+1];
        float d2 = data[i+2], w2 = weight[i+2];
        float d3 = data[i+3], w3 = weight[i+3];
        
        // 并行计算，无数据依赖
        sum0 += d0 * w0;
        sum1 += d1 * w1;
        sum2 += d2 * w2;
        sum3 += d3 * w3;
    }
    
    // 最终归约
    float sum = sum0 + sum1 + sum2 + sum3;
}

// 高级展开：软件流水线
__global__ void software_pipelined() {
    // 预取第一批数据
    float4 data_reg = reinterpret_cast<float4*>(data)[0];
    float4 weight_reg = reinterpret_cast<float4*>(weight)[0];
    
    float4 sum = make_float4(0, 0, 0, 0);
    
    #pragma unroll 8
    for(int i = 1; i < 64; i++) {
        // 预取下一批数据（隐藏延迟）
        float4 next_data = reinterpret_cast<float4*>(data)[i];
        float4 next_weight = reinterpret_cast<float4*>(weight)[i];
        
        // 计算当前批数据
        sum.x += data_reg.x * weight_reg.x;
        sum.y += data_reg.y * weight_reg.y;
        sum.z += data_reg.z * weight_reg.z;
        sum.w += data_reg.w * weight_reg.w;
        
        // 寄存器轮转
        data_reg = next_data;
        weight_reg = next_weight;
    }
}
```

循环展开的性能影响分析：
```
展开因子选择：
┌─────────────┬──────────┬────────────┬─────────────┐
│ 展开因子    │ 寄存器   │ ILP        │ 代码大小    │
├─────────────┼──────────┼────────────┼─────────────┤
│ 1 (不展开)  │ 最少     │ 低         │ 小          │
│ 2-4        │ 适中     │ 中等       │ 适中        │
│ 8-16       │ 较多     │ 高         │ 大          │
│ 32+        │ 很多     │ 饱和       │ 很大        │
└─────────────┴──────────┴────────────┴─────────────┘

选择原则：
- 计算密集型：大展开因子（8-16）
- 内存密集型：小展开因子（2-4）
- 寄存器压力大：保守展开（1-2）
```

### 5.2.3 寄存器别名与重命名

寄存器别名技术允许我们在逻辑上重新组织寄存器的使用，提高代码可读性和优化机会：

```cuda
// 使用union实现寄存器别名
union RegisterAlias {
    float4 vec;
    float arr[4];
    struct {
        float x, y, z, w;
    };
};

__global__ void register_aliasing() {
    RegisterAlias data;
    
    // 作为向量加载
    data.vec = reinterpret_cast<float4*>(global_mem)[idx];
    
    // 作为数组处理
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        data.arr[i] *= 2.0f;
    }
    
    // 作为分量访问
    float len = sqrt(data.x*data.x + data.y*data.y + 
                    data.z*data.z + data.w*data.w);
}

// 寄存器重命名优化数据流
__global__ void register_renaming() {
    // 逻辑上的寄存器重命名
    float input0 = global_mem[idx];
    float input1 = global_mem[idx+1];
    
    // 第一阶段计算
    float stage1_out0 = input0 * 2.0f;
    float stage1_out1 = input1 * 3.0f;
    
    // 重命名：复用input寄存器
    input0 = stage1_out0 + stage1_out1;  // 新用途
    input1 = stage1_out0 - stage1_out1;  // 新用途
    
    // 第二阶段计算
    float final0 = sqrt(input0);
    float final1 = sqrt(input1);
}
```

### 5.2.4 数据依赖性分析

理解和优化数据依赖是寄存器优化的关键：

```cuda
// RAW (Read After Write) 依赖 - 真依赖
__device__ float raw_dependency() {
    float a = 1.0f;      // 写
    float b = a * 2.0f;  // 读（必须等待a）
    return b;
}

// WAR (Write After Read) 依赖 - 反依赖
__device__ void war_dependency() {
    float a = data[0];   // 读
    data[0] = 2.0f;      // 写（可通过寄存器重命名消除）
}

// WAW (Write After Write) 依赖 - 输出依赖
__device__ void waw_dependency() {
    float a = 1.0f;      // 写
    a = 2.0f;            // 写（可优化掉第一次写）
}

// 打破依赖链的技术
__global__ void break_dependencies() {
    // 原始版本：长依赖链
    float a = input[0];
    a = a * 2.0f;
    a = a + 3.0f;
    a = sqrt(a);
    
    // 优化版本：并行计算
    float a0 = input[0];
    float a1 = input[1];
    float a2 = input[2];
    float a3 = input[3];
    
    // 独立计算，无依赖
    a0 = a0 * 2.0f;
    a1 = a1 * 2.0f;
    a2 = a2 * 2.0f;
    a3 = a3 * 2.0f;
    
    a0 = a0 + 3.0f;
    a1 = a1 + 3.0f;
    a2 = a2 + 3.0f;
    a3 = a3 + 3.0f;
    
    a0 = sqrt(a0);
    a1 = sqrt(a1);
    a2 = sqrt(a2);
    a3 = sqrt(a3);
}
```

依赖性分析工具：
```cuda
// 使用 restrict 关键字消除指针别名
__global__ void pointer_aliasing(
    float* __restrict__ input,
    float* __restrict__ output
) {
    // 编译器知道input和output不会重叠
    // 可以更激进地优化寄存器使用
}

// 使用 const 限定符优化只读访问
__global__ void const_optimization(
    const float* __restrict__ input
) {
    // 编译器可以缓存到常量缓存或纹理缓存
    // 减少寄存器压力
}
```

### 5.2.5 编译器优化指示符

CUDA提供了丰富的编译指示符来控制寄存器分配：

```cuda
// 1. 循环展开控制
__global__ void unroll_control() {
    #pragma unroll 8  // 精确展开8次
    for(int i = 0; i < 32; i++) {
        // 循环体
    }
    
    #pragma unroll    // 完全展开
    for(int i = 0; i < KNOWN_SIZE; i++) {
        // 编译时常量大小
    }
    
    #pragma nounroll  // 禁止展开
    for(int i = 0; i < dynamic_size; i++) {
        // 动态大小或想节省寄存器
    }
}

// 2. 内联控制
__device__ __forceinline__ float aggressive_inline() {
    // 强制内联，可能增加寄存器使用
    return complex_computation();
}

__device__ __noinline__ float conservative_inline() {
    // 禁止内联，通过函数调用节省寄存器
    return simple_computation();
}

// 3. Launch bounds 限制
__global__ __launch_bounds__(256, 8)
void optimized_kernel() {
    // 告诉编译器：
    // - 最大256线程/块
    // - 期望8个块/SM
    // 编译器据此优化寄存器分配
}

// 4. 寄存器数量限制
// 编译选项：nvcc -maxrregcount=32
// 或使用 launch_bounds 间接控制

// 5. 优化等级控制
// -O0: 无优化
// -O1: 基本优化
// -O2: 默认优化
// -O3: 激进优化（可能增加寄存器）

// 6. 使用内建变量提示
__global__ void builtin_hints() {
    // 使用 __ldg() 提示只读访问
    float val = __ldg(&global_mem[idx]);
    
    // 使用 __builtin_assume() 提供值域信息
    __builtin_assume(idx >= 0 && idx < 1024);
    
    // 使用 volatile 防止过度优化
    volatile float no_optimize = 1.0f;
}
```

自动驾驶场景的寄存器优化示例：
```cuda
// 激光雷达数据处理：最大化寄存器重用
__global__ void lidar_point_transform() {
    // 变换矩阵常驻寄存器（16个寄存器）
    float4 row0 = make_float4(transform[0], transform[1], 
                              transform[2], transform[3]);
    float4 row1 = make_float4(transform[4], transform[5], 
                              transform[6], transform[7]);
    float4 row2 = make_float4(transform[8], transform[9], 
                              transform[10], transform[11]);
    
    // 批量处理点，重用变换矩阵
    #pragma unroll 4
    for(int i = 0; i < points_per_thread; i++) {
        float4 point = points[thread_offset + i];
        
        // 矩阵乘法，变换矩阵在寄存器中重用
        float x = dot(row0, point);
        float y = dot(row1, point);
        float z = dot(row2, point);
        
        transformed[thread_offset + i] = make_float3(x, y, z);
    }
}

## 5.3 常量内存与纹理内存使用

常量内存和纹理内存是GPU提供的特殊内存空间，它们通过专用缓存提供优化的只读访问模式。虽然在现代GPU中统一内存架构逐渐取代了这些传统内存空间，但理解它们的工作原理对于优化只读数据访问仍然至关重要。

### 5.3.1 常量内存架构

常量内存是一个64KB的只读内存空间，通过专用的常量缓存提供广播式访问优化。其架构设计针对所有线程访问相同地址的场景进行了特殊优化：

```
常量内存层次结构：
┌─────────────────────────────────────┐
│     Device Memory (64KB常量区)       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   L1.5 常量缓存 (每个SM 48KB)        │
│   - 专用只读缓存                     │
│   - 支持广播优化                     │
│   - 1个时钟周期延迟                  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        Warp调度器分发                │
│   - 相同地址：1次访问广播32线程      │
│   - 不同地址：串行化访问             │
└─────────────────────────────────────┘
```

常量内存的关键特性：
1. **广播优化**：当warp中所有线程访问相同地址时，只需一次内存事务
2. **缓存局部性**：48KB的L1.5缓存，命中率高时性能接近寄存器
3. **编译时绑定**：常量内存在编译时分配，内核启动前初始化
4. **全局可见性**：所有SM共享同一常量内存视图

常量内存的使用方式：
```cuda
// 声明常量内存
__constant__ float c_weight[256];
__constant__ float4 c_transform[4];

// 主机端初始化
void initConstants() {
    float weight[256] = { /* ... */ };
    cudaMemcpyToSymbol(c_weight, weight, sizeof(weight));
    
    float4 transform[4] = { /* ... */ };
    cudaMemcpyToSymbol(c_transform, transform, sizeof(transform));
}

// 设备端访问
__global__ void useConstants() {
    // 广播访问：所有线程读相同索引
    float w = c_weight[blockIdx.x];  // 高效
    
    // 分散访问：不同线程读不同索引
    float w2 = c_weight[threadIdx.x]; // 低效，会串行化
}
```

### 5.3.2 常量缓存机制

常量缓存的工作机制决定了其性能特征：

```cuda
// 广播机制示例
__global__ void broadcast_pattern() {
    // 场景1：完美广播 - 所有线程访问相同地址
    float same_value = c_weight[5];  
    // 硬件行为：1次缓存访问，结果广播给32个线程
    // 延迟：~1-4周期
    
    // 场景2：部分广播 - 线程访问少数不同地址
    int idx = threadIdx.x / 8;  // 4个不同值
    float grouped_value = c_weight[idx];
    // 硬件行为：4次串行访问
    // 延迟：~4-16周期
    
    // 场景3：完全分散 - 每个线程访问不同地址
    float scattered_value = c_weight[threadIdx.x];
    // 硬件行为：32次串行访问（最坏情况）
    // 延迟：~32-128周期
}

// 优化常量内存访问模式
__global__ void optimized_constant_access() {
    // 策略1：重组访问模式，增加广播机会
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // 每个warp访问相同的常量
    float warp_constant = c_weight[warp_id];
    
    // 策略2：使用常量内存存储查找表
    int table_idx = __float2int_rn(input * 255.0f);
    float lut_value = c_lut[table_idx];  // 随机访问，但有缓存
    
    // 策略3：混合使用常量和共享内存
    __shared__ float s_weight[32];
    if(lane_id == 0) {
        s_weight[warp_id] = c_weight[blockIdx.x * 4 + warp_id];
    }
    __syncthreads();
    float weight = s_weight[warp_id];  // 从共享内存广播
}
```

常量缓存的性能模型：
```
访问延迟计算：
Latency = Base_Latency + Serialization_Penalty × Unique_Addresses

其中：
- Base_Latency: 1-4周期（缓存命中）
- Serialization_Penalty: 1-4周期/地址
- Unique_Addresses: warp中的唯一地址数（1-32）

带宽计算：
Effective_Bandwidth = Theoretical_Bandwidth / Unique_Addresses
```

### 5.3.3 纹理内存特性

纹理内存最初为图形渲染设计，但其空间局部性优化使其在某些CUDA应用中仍有价值：

```
纹理内存特性：
┌────────────────────────────────────┐
│         纹理内存优势                │
├────────────────────────────────────┤
│ • 2D/3D空间局部性缓存优化           │
│ • 硬件插值（线性、双线性、三线性）   │
│ • 边界处理模式（clamp、wrap）        │
│ • 自动类型转换和归一化              │
│ • 独立的纹理缓存（不占用L1）         │
└────────────────────────────────────┘
```

纹理内存的使用场景：
```cuda
// 1. 图像处理：利用2D空间局部性
texture<float, cudaTextureType2D> tex_image;

__global__ void image_filter() {
    float x = blockIdx.x * blockDim.x + threadIdx.x;
    float y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 硬件双线性插值
    float val = tex2D(tex_image, x + 0.5f, y + 0.5f);
    
    // 9点模板滤波，利用2D缓存局部性
    float sum = 0;
    for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
            sum += tex2D(tex_image, x + dx, y + dy);
        }
    }
}

// 2. 体渲染：3D纹理的三线性插值
texture<float, cudaTextureType3D> tex_volume;

__global__ void volume_rendering() {
    // 光线步进
    float3 pos = ray_origin;
    float3 dir = ray_direction;
    
    for(int i = 0; i < MAX_STEPS; i++) {
        // 硬件三线性插值
        float density = tex3D(tex_volume, pos.x, pos.y, pos.z);
        
        // 累积渲染
        color += density * step_size;
        pos += dir * step_size;
    }
}

// 3. 查找表：利用硬件归一化
texture<float, cudaTextureType1D> tex_lut;

__global__ void color_mapping() {
    float input = data[idx];
    
    // 自动归一化到[0,1]并查表
    float output = tex1D(tex_lut, input);
}
```

### 5.3.4 纹理缓存优化

纹理缓存针对空间局部性优化，理解其工作原理有助于设计高效的访问模式：

```cuda
// 纹理缓存行为分析
__global__ void texture_cache_behavior() {
    // 模式1：行主序访问（缓存友好）
    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) {
            float val = tex2D(tex, x, y);
            // 相邻线程访问相邻像素，缓存命中率高
        }
    }
    
    // 模式2：列主序访问（缓存不友好）
    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            float val = tex2D(tex, x, y);
            // 跨行访问，可能导致缓存颠簸
        }
    }
    
    // 模式3：块访问（最优）
    const int TILE = 16;
    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;
    
    for(int dy = 0; dy < TILE; dy++) {
        for(int dx = 0; dx < TILE; dx++) {
            float val = tex2D(tex, bx + dx, by + dy);
            // 2D块局部性，充分利用纹理缓存
        }
    }
}

// 自动驾驶场景：鱼眼图像畸变校正
__global__ void fisheye_undistort(
    cudaTextureObject_t fisheye_tex,
    float* output,
    const float* distortion_map
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 从畸变映射获取源坐标
    float2 src_coord = reinterpret_cast<float2*>(distortion_map)[y * width + x];
    
    // 纹理硬件自动处理：
    // 1. 边界钳制（超出范围的坐标）
    // 2. 双线性插值（非整数坐标）
    // 3. 缓存优化（相邻像素可能映射到相近区域）
    float4 pixel = tex2D<float4>(fisheye_tex, src_coord.x, src_coord.y);
    
    output[y * width + x] = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
}
```

纹理缓存优化策略：
```cuda
// 策略1：使用纹理对象（Kepler+）替代纹理引用
__global__ void modern_texture_usage(cudaTextureObject_t tex_obj) {
    // 纹理对象优势：
    // - 运行时绑定
    // - 作为参数传递
    // - 更好的编译器优化
    float val = tex2D<float>(tex_obj, x, y);
}

// 策略2：层次化纹理（Mipmap）
__global__ void mipmap_sampling(cudaTextureObject_t tex_mipmap) {
    // 根据采样密度选择合适的细节层次
    float lod = computeLOD(sampling_rate);
    float val = tex2DLod<float>(tex_mipmap, x, y, lod);
}

// 策略3：数组纹理批处理
__global__ void texture_array_batch(cudaTextureObject_t tex_array) {
    // 处理多个相似的2D纹理
    int layer = blockIdx.z;
    float val = tex2DLayered<float>(tex_array, x, y, layer);
}
```

### 5.3.5 LDG指令与只读缓存

Kepler架构引入的LDG（Load Global）指令通过只读数据缓存提供了常量内存的替代方案：

```cuda
// LDG指令的使用方式
__global__ void ldg_usage(const float* __restrict__ input) {
    // 方式1：显式使用__ldg内联函数
    float val1 = __ldg(&input[idx]);
    
    // 方式2：const __restrict__指针（编译器自动推断）
    const float* __restrict__ ptr = input;
    float val2 = ptr[idx];  // 编译器生成LDG指令
    
    // 方式3：通过const引用
    const float& ref = input[idx];
    float val3 = ref;  // 也会生成LDG指令
}

// LDG vs 常量内存对比
__constant__ float c_data[1024];
__global__ void comparison_kernel(const float* __restrict__ g_data) {
    // 常量内存：适合广播访问
    float c_val = c_data[blockIdx.x];  // 所有线程读相同值
    
    // LDG缓存：适合分散访问
    float g_val = __ldg(&g_data[threadIdx.x + blockIdx.x * blockDim.x]);
    
    // 性能特征对比：
    // 常量内存：64KB容量，广播优化，编译时分配
    // LDG缓存：48KB/SM，通用缓存，运行时绑定
}
```

只读缓存的高级优化：
```cuda
// 预取和缓存控制
__global__ void cache_control_kernel(const float* __restrict__ data) {
    // 缓存预取提示
    __builtin_prefetch(&data[idx + 32], 0, 3);
    // 参数：地址，读/写(0/1)，时间局部性(0-3)
    
    // 缓存驱逐策略控制
    float val = __ldcs(&data[idx]);  // 缓存流式访问（不污染缓存）
    float val2 = __ldlu(&data[idx]); // 最后一次使用（提示可驱逐）
    
    // 向量化LDG访问
    float4 vec = __ldg4((float4*)&data[idx]);
}

// 具身智能场景：点云特征提取
__global__ void pointcloud_features(
    const float4* __restrict__ points,      // 点坐标(x,y,z,intensity)
    const float* __restrict__ kd_tree,      // KD树结构（只读）
    float* features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 通过LDG加载点坐标
    float4 point = __ldg(&points[idx]);
    
    // 遍历KD树查找近邻（树结构适合只读缓存）
    int node_idx = 0;
    while(node_idx >= 0) {
        // KD树节点通过只读缓存访问
        float split_val = __ldg(&kd_tree[node_idx * 4]);
        float split_dim = __ldg(&kd_tree[node_idx * 4 + 1]); 
        int left_child = __ldg(&kd_tree[node_idx * 4 + 2]);
        int right_child = __ldg(&kd_tree[node_idx * 4 + 3]);
        
        // 树遍历逻辑
        float point_val = (split_dim == 0) ? point.x : 
                         (split_dim == 1) ? point.y : point.z;
        node_idx = (point_val < split_val) ? left_child : right_child;
    }
}
```

只读数据优化决策树：
```
数据访问模式分析：
├─ 所有线程访问相同数据？
│  └─ 是 → 使用常量内存
│  └─ 否 ↓
├─ 数据具有2D/3D空间局部性？
│  └─ 是 → 使用纹理内存
│  └─ 否 ↓
├─ 数据大小超过64KB？
│  └─ 是 → 使用LDG指令
│  └─ 否 → 考虑常量内存或LDG
```

## 5.4 联合体与位操作优化

在追求极致性能的CUDA编程中，位级操作和类型转换技巧是不可忽视的优化手段。通过联合体实现零开销的类型转换、利用位操作实现高效的数据打包，以及使用特殊的数学函数，我们可以在指令级别榨取GPU的每一分性能。

### 5.4.1 联合体的内存布局

联合体(union)允许多个成员共享同一块内存，这在CUDA中可用于高效的类型转换和数据重解释：

```cuda
// 基础联合体布局
union BasicUnion {
    float f;
    int i;
    unsigned int u;
    struct {
        unsigned short low;
        unsigned short high;
    } parts;
};

// 内存布局示意：
// ┌────────────────────────────────┐
// │         32 bits                 │
// ├────────────────────────────────┤
// │ f: IEEE 754 float              │
// │ i: signed integer              │
// │ u: unsigned integer            │
// │ parts.low | parts.high         │
// └────────────────────────────────┘

__device__ float fast_abs(float x) {
    BasicUnion u;
    u.f = x;
    u.u &= 0x7FFFFFFF;  // 清除符号位
    return u.f;
}

// 向量类型的联合体
union Vec4Union {
    float4 vec;
    float arr[4];
    struct {
        float x, y, z, w;
    } comp;
    struct {
        float2 xy;
        float2 zw;
    } pairs;
    unsigned int raw[4];
};

__global__ void vector_manipulation() {
    Vec4Union data;
    
    // 作为向量加载
    data.vec = reinterpret_cast<float4*>(global_mem)[idx];
    
    // 作为数组处理
    float sum = 0;
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        sum += data.arr[i];
    }
    
    // 作为分量对处理
    float2 diff = data.pairs.xy - data.pairs.zw;
    
    // 位级操作
    data.raw[0] &= 0xFFFFFFF0;  // 清除低4位
}
```

高级联合体技巧：
```cuda
// 多类型数据打包
union PackedData {
    struct {
        float x, y, z;      // 96 bits
        unsigned int flags; // 32 bits
    } fields;
    float4 packed;          // 128 bits aligned
    unsigned int raw[4];
};

// 颜色格式转换
union ColorUnion {
    struct {
        unsigned char r, g, b, a;
    } rgba8;
    unsigned int packed_rgba;
    struct {
        unsigned short r : 5;
        unsigned short g : 6;
        unsigned short b : 5;
    } rgb565;
    float4 normalized;  // [0,1]范围
};

__device__ unsigned int float_to_rgba8(float4 color) {
    ColorUnion u;
    u.rgba8.r = __float2int_rn(color.x * 255.0f);
    u.rgba8.g = __float2int_rn(color.y * 255.0f);
    u.rgba8.b = __float2int_rn(color.z * 255.0f);
    u.rgba8.a = __float2int_rn(color.w * 255.0f);
    return u.packed_rgba;
}

// 自动驾驶场景：激光雷达点压缩
union LidarPoint {
    struct {
        float x, y, z;          // 坐标
        unsigned char intensity; // 强度
        unsigned char layer;     // 激光层
        unsigned short flags;    // 标志位
    } fields;                   // 16 bytes
    
    float4 compressed;           // 16 bytes aligned
    
    struct {
        int x_fixed : 20;        // 20位定点数坐标
        int y_fixed : 20;        // 精度：0.1mm
        int z_fixed : 20;
        unsigned intensity : 8;
        unsigned layer : 4;
    } packed;                    // 8 bytes
};
```

### 5.4.2 类型双关（Type Punning）

类型双关允许将一种类型的位模式解释为另一种类型，在CUDA中这是零开销的操作：

```cuda
// 浮点数位操作
__device__ float fast_inverse_sqrt(float x) {
    // Quake III的快速平方根倒数算法
    union { float f; int i; } conv;
    conv.f = x;
    conv.i = 0x5f3759df - (conv.i >> 1);  // 魔数
    conv.f = conv.f * (1.5f - 0.5f * x * conv.f * conv.f);  // Newton迭代
    return conv.f;
}

// 浮点数分类
__device__ int float_classify(float x) {
    union { float f; unsigned int u; } conv;
    conv.f = x;
    
    unsigned int exp = (conv.u >> 23) & 0xFF;
    unsigned int frac = conv.u & 0x7FFFFF;
    
    if(exp == 0) {
        return frac == 0 ? FP_ZERO : FP_SUBNORMAL;
    } else if(exp == 0xFF) {
        return frac == 0 ? FP_INFINITE : FP_NAN;
    }
    return FP_NORMAL;
}

// 高效的范围映射
__device__ unsigned char float_to_unorm8(float x) {
    // 将[0,1]映射到[0,255]，饱和处理
    union { float f; unsigned int u; } conv;
    conv.f = fmaxf(0.0f, fminf(1.0f, x)) * 255.0f + 0.5f;
    
    // 提取整数部分（避免类型转换）
    int exp = ((conv.u >> 23) & 0xFF) - 127;
    int mantissa = (conv.u & 0x7FFFFF) | 0x800000;
    return (unsigned char)(mantissa >> (23 - exp));
}

// 双精度模拟（使用两个float）
struct Double2 {
    float hi, lo;
    
    __device__ Double2 add(const Double2& b) const {
        // Knuth's two-sum algorithm
        float s = hi + b.hi;
        float v = s - hi;
        float e = (hi - (s - v)) + (b.hi - v) + lo + b.lo;
        return {s + e, e - (s + e - s)};
    }
};
```

内存重解释的高级用法：
```cuda
// 结构体数组 vs 数组结构体
struct AoS {  // Array of Structures
    float x, y, z, w;
};

struct SoA {  // Structure of Arrays
    float* x;
    float* y; 
    float* z;
    float* w;
};

// 使用union实现零拷贝转换
union AoS_SoA_Converter {
    AoS aos[256];
    struct {
        float x[256];
        float y[256];
        float z[256];
        float w[256];
    } soa;
};

// SIMD友好的数据布局转换
__global__ void transpose_4x4_blocks(float* data) {
    union {
        float4 rows[4];
        float elements[16];
    } block;
    
    // 加载4x4块
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        block.rows[i] = reinterpret_cast<float4*>(data)[base + i];
    }
    
    // 转置（使用shuffle）
    // ... 转置逻辑 ...
    
    // 写回
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        reinterpret_cast<float4*>(data)[base + i] = block.rows[i];
    }
}
```

### 5.4.3 位级并行操作

GPU提供了丰富的位操作指令，可以实现高效的并行位处理：

```cuda
// 基础位操作
__device__ int population_count(unsigned int x) {
    return __popc(x);  // 硬件指令，计算1的个数
}

__device__ int find_first_set(unsigned int x) {
    return __ffs(x);   // 找到第一个设置的位（从1开始）
}

__device__ int count_leading_zeros(unsigned int x) {
    return __clz(x);   // 计算前导零
}

// 位反转
__device__ unsigned int bit_reverse(unsigned int x) {
    return __brev(x);  // 硬件位反转
}

// 并行位提取和打包
__device__ unsigned int extract_bits(unsigned int x, int pos, int len) {
    return __bfe(x, pos, len);  // Bit Field Extract
}

__device__ unsigned int insert_bits(unsigned int x, unsigned int y, 
                                   int pos, int len) {
    return __bfi(x, y, pos, len);  // Bit Field Insert
}

// 高级位操作：Morton编码（Z-order）
__device__ unsigned int morton_encode_2d(unsigned int x, unsigned int y) {
    // 交错x和y的位
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    
    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;
    
    return x | (y << 1);
}

// 位掩码并行处理
__global__ void parallel_bitmask_ops(unsigned int* masks, unsigned int* data) {
    unsigned int mask = masks[blockIdx.x];
    unsigned int value = data[threadIdx.x];
    
    // 并行处理32个位
    int my_bit = threadIdx.x % 32;
    bool bit_set = (mask >> my_bit) & 1;
    
    if(bit_set) {
        // 仅处理设置的位
        value = process_bit(value, my_bit);
    }
    
    // 使用warp投票收集结果
    unsigned int result = __ballot_sync(0xFFFFFFFF, bit_set);
}
```

布隆过滤器的GPU实现：
```cuda
// 高性能布隆过滤器
class BloomFilter {
    unsigned int* bits;
    int size;
    
    __device__ unsigned int hash1(unsigned int x) {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        return (x >> 16) ^ x;
    }
    
    __device__ unsigned int hash2(unsigned int x) {
        x = ((x >> 16) ^ x) * 0x119de1f3;
        x = ((x >> 16) ^ x) * 0x119de1f3;
        return (x >> 16) ^ x;
    }
    
public:
    __device__ void insert(unsigned int key) {
        unsigned int h1 = hash1(key) % (size * 32);
        unsigned int h2 = hash2(key) % (size * 32);
        
        // 原子设置位
        atomicOr(&bits[h1 / 32], 1U << (h1 % 32));
        atomicOr(&bits[h2 / 32], 1U << (h2 % 32));
    }
    
    __device__ bool contains(unsigned int key) {
        unsigned int h1 = hash1(key) % (size * 32);
        unsigned int h2 = hash2(key) % (size * 32);
        
        return (bits[h1 / 32] & (1U << (h1 % 32))) &&
               (bits[h2 / 32] & (1U << (h2 % 32)));
    }
};
```

### 5.4.4 快速数学函数

CUDA提供了牺牲精度换取速度的快速数学函数：

```cuda
// 快速数学函数对比
__global__ void math_functions_comparison() {
    float x = 1.5f;
    
    // 标准函数（高精度，慢）
    float s1 = sinf(x);         // ~32 cycles
    float c1 = cosf(x);         // ~32 cycles
    float e1 = expf(x);         // ~20 cycles
    float l1 = logf(x);         // ~20 cycles
    float r1 = 1.0f / sqrtf(x); // ~16 cycles
    
    // 快速函数（低精度，快）
    float s2 = __sinf(x);       // ~8 cycles
    float c2 = __cosf(x);       // ~8 cycles  
    float e2 = __expf(x);       // ~4 cycles
    float l2 = __logf(x);       // ~4 cycles
    float r2 = rsqrtf(x);       // ~4 cycles
    
    // 近似函数（最快，精度最低）
    float r3 = __frsqrt_rn(x);  // ~2 cycles
    float d3 = __fdividef(1.0f, x); // ~6 cycles vs 20 for /
}

// 组合快速函数实现复杂运算
__device__ float fast_exp2(float x) {
    // 2^x = 2^(int + frac)
    float xi = floorf(x);
    float xf = x - xi;
    
    // 整数部分：位操作
    union { float f; unsigned int u; } conv;
    conv.u = (unsigned int)((127 + xi) * (1 << 23));
    
    // 小数部分：多项式近似
    float poly = 1.0f + xf * (0.6931f + xf * 0.2416f);
    
    return conv.f * poly;
}

// SIMD风格的向量运算
__device__ float4 fast_normalize(float4 v) {
    float inv_len = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w);
    return make_float4(v.x * inv_len, v.y * inv_len, 
                      v.z * inv_len, v.w * inv_len);
}

// 自动驾驶场景：快速三角函数表
__constant__ float sin_table[256];
__constant__ float cos_table[256];

__device__ float2 fast_sincos(float angle) {
    // 将角度映射到[0, 255]
    int idx = __float2int_rn(angle * 255.0f / (2.0f * M_PI)) & 0xFF;
    return make_float2(sin_table[idx], cos_table[idx]);
}
```

特殊函数优化：
```cuda
// 快速指数函数（用于神经网络）
__device__ float fast_exp_nn(float x) {
    // 限制范围避免溢出
    x = fmaxf(-88.0f, fminf(88.0f, x));
    
    // 使用位操作和多项式近似
    union { float f; int i; } conv;
    conv.i = __float2int_rn(12102203.0f * x) + 127 * (1 << 23);
    
    // 修正项
    float correction = 0.05f * (x - __logf(conv.f));
    return conv.f * (1.0f + correction);
}

// 快速sigmoid（用于激活函数）
__device__ float fast_sigmoid(float x) {
    // 使用有理函数近似
    return 1.0f / (1.0f + fast_exp_nn(-x));
}

// 快速tanh
__device__ float fast_tanh(float x) {
    float e2x = fast_exp_nn(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}
```

### 5.4.5 定点数运算优化

定点数运算可以在某些场景下提供比浮点数更高的性能：

```cuda
// 定点数类型定义
template<int FRAC_BITS>
struct Fixed32 {
    int value;
    
    static constexpr int SCALE = 1 << FRAC_BITS;
    static constexpr float INV_SCALE = 1.0f / SCALE;
    
    __device__ Fixed32(float f) : value(__float2int_rn(f * SCALE)) {}
    __device__ Fixed32(int i) : value(i << FRAC_BITS) {}
    
    __device__ float to_float() const { return value * INV_SCALE; }
    
    __device__ Fixed32 operator+(const Fixed32& b) const {
        return Fixed32{value + b.value};
    }
    
    __device__ Fixed32 operator*(const Fixed32& b) const {
        // 64位中间结果避免溢出
        long long prod = (long long)value * b.value;
        return Fixed32{(int)(prod >> FRAC_BITS)};
    }
};

// Q格式定点数（DSP风格）
typedef Fixed32<16> Q16_16;  // 16位整数，16位小数
typedef Fixed32<24> Q8_24;   // 8位整数，24位小数

// 定点数三角函数（使用CORDIC算法）
__device__ Q16_16 cordic_atan2(Q16_16 y, Q16_16 x) {
    const int ITERATIONS = 16;
    __constant__ int atan_table[16] = {
        0x1921FB54, 0x0ED63382, 0x07D6DD7E, // ...预计算的arctan值
    };
    
    int angle = 0;
    int xi = x.value;
    int yi = y.value;
    
    #pragma unroll
    for(int i = 0; i < ITERATIONS; i++) {
        int xi_shift = xi >> i;
        int yi_shift = yi >> i;
        
        if(yi > 0) {
            xi += yi_shift;
            yi -= xi_shift;
            angle += atan_table[i];
        } else {
            xi -= yi_shift;
            yi += xi_shift;
            angle -= atan_table[i];
        }
    }
    
    return Q16_16{angle};
}

// 混合精度运算
__global__ void mixed_precision_kernel(
    const half* input,      // FP16输入
    Fixed32<12>* temp,      // 定点中间结果
    float* output           // FP32输出
) {
    // FP16 → 定点
    half h = input[idx];
    Fixed32<12> fixed(__half2float(h));
    
    // 定点运算（快速）
    fixed = fixed * Fixed32<12>(2.5f);
    fixed = fixed + Fixed32<12>(1.0f);
    
    // 定点 → FP32
    output[idx] = fixed.to_float();
}

// 具身智能场景：IMU数据处理
struct IMU_Sample {
    Fixed32<14> accel_x, accel_y, accel_z;  // 加速度
    Fixed32<14> gyro_x, gyro_y, gyro_z;     // 陀螺仪
    
    __device__ Fixed32<14> magnitude() const {
        // 定点数平方和
        auto mag2 = accel_x * accel_x + 
                   accel_y * accel_y + 
                   accel_z * accel_z;
        
        // 定点数开方（Newton迭代）
        Fixed32<14> x = mag2;
        Fixed32<14> half = Fixed32<14>(0.5f);
        
        #pragma unroll 4
        for(int i = 0; i < 4; i++) {
            x = (x + mag2 / x) * half;
        }
        
        return x;
    }
};
```

定点数优化的实际应用：
```cuda
// 图像处理：定点数卷积
__global__ void fixed_point_convolution(
    const unsigned char* input,
    signed char* kernel,  // INT8量化的卷积核
    unsigned char* output
) {
    const int KERNEL_SIZE = 3;
    const int FRAC_BITS = 8;
    
    int sum = 0;
    
    #pragma unroll
    for(int ky = 0; ky < KERNEL_SIZE; ky++) {
        #pragma unroll
        for(int kx = 0; kx < KERNEL_SIZE; kx++) {
            int pixel = input[(y + ky) * width + (x + kx)];
            int weight = kernel[ky * KERNEL_SIZE + kx];
            sum += pixel * weight;  // 定点乘法
        }
    }
    
    // 归一化和饱和
    sum = (sum + (1 << (FRAC_BITS - 1))) >> FRAC_BITS;  // 四舍五入
    sum = max(0, min(255, sum));  // 饱和到[0,255]
    
    output[y * width + x] = (unsigned char)sum;
}
```

## 5.5 案例：寄存器级的卷积优化

### 5.5.1 卷积算法分析

### 5.5.2 寄存器阻塞策略

### 5.5.3 数据复用模式设计

### 5.5.4 性能分析与调优

### 5.5.5 与cuDNN的性能对比

## 本章小结

## 练习题

## 常见陷阱与错误

## 最佳实践检查清单

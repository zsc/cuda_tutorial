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

### 5.3.1 常量内存架构

### 5.3.2 常量缓存机制

### 5.3.3 纹理内存特性

### 5.3.4 纹理缓存优化

### 5.3.5 LDG指令与只读缓存

## 5.4 联合体与位操作优化

### 5.4.1 联合体的内存布局

### 5.4.2 类型双关（Type Punning）

### 5.4.3 位级并行操作

### 5.4.4 快速数学函数

### 5.4.5 定点数运算优化

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

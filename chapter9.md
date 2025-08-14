# 第9章：张量核心与混合精度计算

## 本章概览

张量核心（Tensor Core）是NVIDIA从Volta架构开始引入的专用硬件单元，为深度学习工作负载提供了革命性的性能提升。通过支持混合精度计算，张量核心能够在保持模型精度的同时，实现4-8倍的吞吐量提升和显著的能效改进。在自动驾驶的感知模型和具身智能的决策网络中，张量核心已成为实现实时推理的关键技术。

本章将深入探讨张量核心的硬件架构、编程接口和优化策略。你将学习如何使用WMMA API编写高效的张量核心代码，掌握混合精度训练的核心技术，并通过Transformer模型的实际案例，体验张量核心带来的性能飞跃。更重要的是，我们将详细讨论数值稳定性问题，确保你能在追求极致性能的同时保持计算的正确性。

### 学习目标

完成本章学习后，你将能够：
- 理解张量核心的硬件架构和计算原理
- 熟练使用WMMA API进行张量核心编程
- 设计和实现混合精度训练策略
- 处理数值精度相关的各种挑战
- 将张量核心应用于实际的深度学习模型加速

## 9.1 Tensor Core架构与编程模型

### 9.1.1 张量核心硬件架构

张量核心是专门为执行矩阵乘累加（Matrix Multiply-Accumulate, MMA）操作而设计的硬件单元。每个SM包含多个张量核心，它们能够在单个时钟周期内完成大规模的矩阵运算。

```
张量核心计算模型：
D = A × B + C

其中：
- A: M×K 矩阵
- B: K×N 矩阵  
- C: M×N 矩阵（累加器）
- D: M×N 矩阵（结果）
```

硬件架构特点：

1. **并行计算单元**：每个张量核心包含大量的乘累加单元（例如V100的张量核心包含64个FMA单元）

2. **分层架构**：
```
SM Level:
┌─────────────────────────────────┐
│         Streaming              │
│       Multiprocessor           │
│  ┌──────────┬──────────┐       │
│  │  Tensor  │  Tensor  │       │
│  │  Core 0  │  Core 1  │ ...   │
│  └──────────┴──────────┘       │
└─────────────────────────────────┘

Tensor Core Level:
┌─────────────────────────────────┐
│        Tensor Core              │
│  ┌────────────────────────┐     │
│  │   64 FMA Units         │     │
│  │  (Fused Multiply-Add)  │     │
│  └────────────────────────┘     │
│  ┌────────────────────────┐     │
│  │  Register File         │     │
│  │  (Fragment Storage)    │     │
│  └────────────────────────┘     │
└─────────────────────────────────┘
```

3. **支持的数据类型演进**：
- Volta (V100): FP16输入，FP16/FP32累加
- Turing (T4): 增加INT8/INT4支持
- Ampere (A100): 增加BF16、TF32支持
- Hopper (H100): 增加FP8支持，引入Transformer Engine

### 9.1.2 矩阵片段（Matrix Fragments）

张量核心编程的核心概念是矩阵片段（Matrix Fragment），它是存储在寄存器中的矩阵子块：

```
Warp级协作模型：
┌─────────────────────────────────────┐
│            Warp (32 threads)        │
├─────────────────────────────────────┤
│  Thread 0: Fragment A[0], B[0]...   │
│  Thread 1: Fragment A[1], B[1]...   │
│  ...                                 │
│  Thread 31: Fragment A[31], B[31]...│
└─────────────────────────────────────┘

每个线程持有完整矩阵的一部分数据
所有线程协作完成矩阵运算
```

片段的特性：
- **分布式存储**：矩阵数据分散在warp的32个线程中
- **不透明格式**：片段的内部布局对程序员不可见
- **硬件优化**：布局由硬件决定以最大化性能

### 9.1.3 编程模型层次

张量核心提供多个编程抽象层次：

1. **WMMA API（Warp Matrix Multiply-Accumulate）**：
   - CUDA C++原生接口
   - 直接映射到PTX指令
   - 最大的控制灵活性

2. **CUTLASS**：
   - 模板化的高性能库
   - 提供优化的GEMM实现
   - 支持自定义epilogue

3. **cuBLAS/cuDNN**：
   - 高层库函数
   - 自动选择最优实现
   - 易于使用但灵活性较低

4. **框架集成（TensorFlow/PyTorch）**：
   - 自动混合精度（AMP）
   - 透明的张量核心加速
   - 最少的代码修改

## 9.2 WMMA API详解

### 9.2.1 WMMA基础操作

WMMA API提供了三个核心操作来完成张量核心计算：

```cpp
// 1. 加载操作 - 从内存加载矩阵片段
wmma::load_matrix_sync(fragment, ptr, stride);

// 2. 计算操作 - 执行矩阵乘累加
wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

// 3. 存储操作 - 将结果写回内存
wmma::store_matrix_sync(ptr, fragment, stride);
```

关键概念：
- **sync后缀**：表示warp级同步操作，所有线程必须参与
- **stride参数**：矩阵的leading dimension，用于正确索引
- **fragment**：分布在warp所有线程中的矩阵数据

### 9.2.2 Fragment声明与管理

Fragment是WMMA编程的核心数据结构：

```cpp
// Fragment模板参数说明
template<
    typename Use,        // matrix_a, matrix_b, accumulator
    int m, int n, int k, // 矩阵维度
    typename T,          // 数据类型
    typename Layout      // row_major, col_major (可选)
>
class fragment;

// 实际使用示例
namespace wmma {
    // A矩阵片段 (M x K)
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    
    // B矩阵片段 (K x N)
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    
    // 累加器片段 (M x N)
    fragment<accumulator, 16, 16, 16, float> c_frag;
}
```

支持的矩阵尺寸组合：
```
架构        支持的(M, N, K)组合
--------------------------------------
Volta:      (16, 16, 16), (32, 8, 16), (8, 32, 16)
Turing:     上述 + (8, 8, 16) for INT8
Ampere:     上述 + (16, 8, 8), (8, 16, 8) for TF32
Hopper:     上述 + (16, 16, 8) for FP8
```

### 9.2.3 完整GEMM实现

下面是一个优化的GEMM内核实现，展示了WMMA的实际应用：

```cpp
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void wmma_gemm_optimized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    // 常量定义
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // 共享内存声明
    __shared__ half smem_a[TILE_M][TILE_K];
    __shared__ half smem_b[TILE_K][TILE_N];
    
    // 计算warp和线程的位置
    const int warpId = threadIdx.x / warpSize;
    const int laneId = threadIdx.x % warpSize;
    const int warpsPerBlock = blockDim.x / warpSize;
    
    // 计算输出块的位置
    const int blockRow = blockIdx.y * TILE_M;
    const int blockCol = blockIdx.x * TILE_N;
    
    // Fragment声明
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // 初始化累加器
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // 主循环：沿K维度分块
    for (int k = 0; k < K; k += TILE_K) {
        // 协作加载数据到共享内存
        // 每个线程负责加载一部分数据
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        
        // 加载A矩阵块
        if (blockRow + tidy < M && k + tidx < K) {
            smem_a[tidy][tidx] = A[(blockRow + tidy) * K + (k + tidx)];
        } else {
            smem_a[tidy][tidx] = 0.0f;
        }
        
        // 加载B矩阵块
        if (k + tidy < K && blockCol + tidx < N) {
            smem_b[tidy][tidx] = B[(k + tidy) * N + (blockCol + tidx)];
        } else {
            smem_b[tidy][tidx] = 0.0f;
        }
        
        __syncthreads();
        
        // 使用张量核心计算
        for (int i = 0; i < TILE_K; i += WMMA_K) {
            // 计算warp负责的子块位置
            int warpRow = warpId / (TILE_N / WMMA_N) * WMMA_M;
            int warpCol = warpId % (TILE_N / WMMA_N) * WMMA_N;
            
            // 加载fragments
            wmma::load_matrix_sync(a_frag, &smem_a[warpRow][i], TILE_K);
            wmma::load_matrix_sync(b_frag, &smem_b[i][warpCol], TILE_N);
            
            // 执行矩阵乘累加
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        __syncthreads();
    }
    
    // 写回结果
    int warpRow = warpId / (TILE_N / WMMA_N) * WMMA_M;
    int warpCol = warpId % (TILE_N / WMMA_N) * WMMA_N;
    
    if (blockRow + warpRow < M && blockCol + warpCol < N) {
        // 应用alpha和beta缩放
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        
        // 加载原始C矩阵
        float* c_ptr = C + (blockRow + warpRow) * N + (blockCol + warpCol);
        wmma::load_matrix_sync(c_frag, c_ptr, N, wmma::mem_row_major);
        
        // 缩放并累加
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        
        // 存储最终结果
        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
    }
}
```

### 9.2.4 性能优化技巧

1. **数据重用最大化**：
```
每个A fragment被N/WMMA_N个B fragments重用
每个B fragment被M/WMMA_M个A fragments重用
优化目标：最大化fragment在寄存器中的驻留时间
```

2. **共享内存bank conflict避免**：
```cpp
// 使用padding避免bank conflict
__shared__ half smem_a[TILE_M][TILE_K + 1];  // +1 padding
__shared__ half smem_b[TILE_K][TILE_N + 1];
```

3. **异步数据传输**：
```cpp
// 使用异步拷贝指令（Ampere+）
__pipeline_memcpy_async(&smem_a[row][col], 
                        &global_a[row * K + col], 
                        sizeof(half));
__pipeline_commit();
__pipeline_wait_prior(0);
```

## 9.3 混合精度训练策略

### 9.3.1 混合精度训练原理

混合精度训练通过在不同计算阶段使用不同数值精度来平衡性能和精度：

```
混合精度训练流程：
┌──────────────────┐
│   FP32 主权重     │ ←─── 保持高精度副本
└────────┬─────────┘
         │ 转换
    ┌────▼─────┐
    │ FP16权重  │
    └────┬─────┘
         │
┌────────▼─────────┐
│   前向传播        │ ←─── FP16计算
│  (Tensor Core)   │
└────────┬─────────┘
         │
    ┌────▼─────┐
    │ FP16损失  │
    └────┬─────┘
         │ 损失缩放
    ┌────▼─────┐
    │缩放后损失 │
    └────┬─────┘
         │
┌────────▼─────────┐
│   反向传播        │ ←─── FP16计算
│  (Tensor Core)   │
└────────┬─────────┘
         │
    ┌────▼─────┐
    │FP16梯度  │
    └────┬─────┘
         │ 反缩放
    ┌────▼─────┐
    │FP32梯度  │
    └────┬─────┘
         │
┌────────▼─────────┐
│  权重更新(FP32)   │ ←─── 高精度更新
└──────────────────┘
```

关键优势：
1. **性能提升**：张量核心加速，内存带宽减半
2. **内存节省**：激活值和梯度使用FP16存储
3. **精度保持**：FP32主权重确保收敛质量

### 9.3.2 损失缩放机制

损失缩放是混合精度训练的核心技术，用于防止梯度下溢：

```
FP16动态范围问题：
┌────────────────────────────────┐
│     FP32: ±3.4×10^38           │
│     FP16: ±65,504              │
│                                 │
│  梯度分布:                      │
│  ┌──────────────────────┐      │
│  │ 大部分梯度 < 2^-24    │      │
│  │ FP16下溢为0!          │      │
│  └──────────────────────┘      │
└────────────────────────────────┘
```

**静态损失缩放**：
```cpp
const float loss_scale = 1024.0f;  // 固定缩放因子

// 前向传播
output = model_forward(input);
loss = compute_loss(output, target);

// 应用损失缩放
scaled_loss = loss * loss_scale;

// 反向传播
scaled_gradients = backward(scaled_loss);

// 梯度反缩放
for (auto& grad : scaled_gradients) {
    grad = grad / loss_scale;
}

// 权重更新
optimizer.step(gradients);
```

**动态损失缩放算法**：
```cpp
class DynamicLossScaler {
private:
    float scale = 65536.0f;     // 初始缩放因子
    int growth_factor = 2;       // 增长因子
    int backoff_factor = 2;      // 回退因子
    int growth_interval = 2000;  // 增长间隔
    int _growth_tracker = 0;     // 增长计数器
    
public:
    bool scale_and_check(Tensor& loss, Tensor& gradients) {
        // 应用损失缩放
        loss *= scale;
        
        // 反向传播
        gradients = compute_gradients(loss);
        
        // 检查是否有inf/nan
        if (has_inf_or_nan(gradients)) {
            // 减小缩放因子
            scale /= backoff_factor;
            _growth_tracker = 0;
            return false;  // 跳过本次更新
        }
        
        // 梯度反缩放
        gradients /= scale;
        
        // 更新增长计数器
        _growth_tracker++;
        
        // 定期增大缩放因子
        if (_growth_tracker >= growth_interval) {
            scale *= growth_factor;
            _growth_tracker = 0;
        }
        
        return true;  // 可以更新权重
    }
};
```

### 9.3.3 自动混合精度实现

现代深度学习框架提供了自动混合精度（AMP）支持：

**PyTorch AMP示例（伪代码）**：
```python
# 初始化
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = torch.cuda.amp.GradScaler()

# 训练循环
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # 自动混合精度前向传播
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # 损失缩放和反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪（可选）
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步骤
        scaler.step(optimizer)
        scaler.update()
```

**关键组件**：
1. **autocast上下文**：自动选择合适的精度
2. **GradScaler**：处理损失缩放和梯度反缩放
3. **黑白名单机制**：特定层强制使用FP32

### 9.3.4 精度敏感层处理

某些操作对数值精度特别敏感，需要特殊处理：

```cpp
// 自定义混合精度策略
class MixedPrecisionPolicy {
    // FP32白名单（始终使用FP32）
    const std::set<std::string> fp32_ops = {
        "BatchNorm",      // 批归一化
        "LayerNorm",      // 层归一化
        "Softmax",        // Softmax激活
        "CrossEntropy",   // 交叉熵损失
        "L2Loss"          // L2损失
    };
    
    // FP16黑名单（避免使用FP16）
    const std::set<std::string> fp16_blacklist = {
        "Exp",            // 指数运算
        "Log",            // 对数运算
        "Pow",            // 幂运算
        "Sum",            // 大规模求和
        "Prod"            // 大规模乘积
    };
    
    // 自适应精度选择
    DataType select_precision(const std::string& op_type, 
                              const TensorShape& shape) {
        // 强制FP32
        if (fp32_ops.count(op_type)) {
            return DataType::FP32;
        }
        
        // 避免FP16
        if (fp16_blacklist.count(op_type)) {
            return DataType::FP32;
        }
        
        // 小张量使用FP32（避免开销）
        if (shape.num_elements() < 1024) {
            return DataType::FP32;
        }
        
        // 默认使用FP16
        return DataType::FP16;
    }
};
```

### 9.3.5 梯度累积与混合精度

在小批量训练时，梯度累积需要特别注意：

```cpp
// 混合精度梯度累积
class MixedPrecisionGradientAccumulator {
private:
    std::vector<Tensor> fp32_grad_buffers;  // FP32梯度缓冲
    int accumulation_steps;
    int current_step = 0;
    
public:
    void accumulate(const std::vector<Tensor>& fp16_grads, 
                   float loss_scale) {
        for (size_t i = 0; i < fp16_grads.size(); i++) {
            // 转换为FP32并反缩放
            Tensor fp32_grad = fp16_grads[i].to(DataType::FP32);
            fp32_grad /= loss_scale;
            
            // 累积梯度
            if (current_step == 0) {
                fp32_grad_buffers[i] = fp32_grad;
            } else {
                fp32_grad_buffers[i] += fp32_grad;
            }
        }
        
        current_step++;
        
        // 达到累积步数，执行更新
        if (current_step >= accumulation_steps) {
            // 平均梯度
            for (auto& grad : fp32_grad_buffers) {
                grad /= accumulation_steps;
            }
            
            // 应用梯度更新
            optimizer.step(fp32_grad_buffers);
            
            // 重置
            current_step = 0;
            clear_buffers();
        }
    }
};
```

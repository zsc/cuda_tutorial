# 第20章：CUDA Graph与内核融合

在现代AI推理系统中，单个内核的优化已经接近硬件极限，而系统级的优化成为了提升性能的关键。CUDA Graph通过消除CPU-GPU同步开销实现了执行流的极致优化，而内核融合则通过减少内存访问次数大幅提升了算子效率。本章将深入探讨这两项关键技术，并结合JIT编译和自动调优框架，帮助你构建端到端的高性能推理系统。通过学习本章，你将掌握将离散的CUDA内核组织成高效执行图的方法，理解不同层次的融合策略，并能够针对特定硬件和工作负载进行自动优化。

## 20.1 CUDA Graph构建与优化

### 20.1.1 Graph基本概念与执行模型

CUDA Graph是CUDA 10引入的革命性特性，它将一系列CUDA操作（内核启动、内存拷贝、同步等）封装成一个可重复执行的图结构。与传统的流式执行相比，Graph模式具有以下优势：

1. **极低的启动开销**：Graph实例化后，整个执行序列通过单次API调用完成，避免了反复的内核启动开销
2. **优化的调度**：驱动程序可以预先分析整个执行图，进行全局优化
3. **确定性执行**：固定的执行模式有利于性能预测和调试

Graph的核心组件包括：
- **节点（Node）**：代表单个操作，如内核启动、memcpy、event记录等
- **边（Edge）**：定义节点间的依赖关系
- **图模板（Graph Template）**：定义执行拓扑的蓝图
- **图实例（Executable Graph）**：可以实际执行的图对象

```
Graph执行流程：
     创建           实例化            执行
Template -----> Executable -----> Launch
  Graph           Graph           (重复)
    |               |                |
  定义拓扑        资源分配         高效执行
```

### 20.1.2 流捕获机制详解

流捕获（Stream Capture）是构建CUDA Graph最便捷的方式，它能够自动记录流上的操作序列：

```cpp
// 基本流捕获模式
cudaGraph_t graph;
cudaStream_t stream;
cudaStreamCreate(&stream);

// 开始捕获
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 记录操作序列
kernel1<<<grid1, block1, 0, stream>>>(...);
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
kernel2<<<grid2, block2, 0, stream>>>(...);

// 结束捕获
cudaStreamEndCapture(stream, &graph);

// 实例化图
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// 执行图（可重复执行）
for (int i = 0; i < iterations; i++) {
    cudaGraphLaunch(graphExec, stream);
}
```

流捕获支持三种模式：
- **Global模式**：捕获所有相关流的操作
- **Local模式**：仅捕获指定流的操作
- **Relaxed模式**：允许某些操作逃逸捕获

对于复杂的多流场景，需要careful处理流间同步：

```cpp
// 多流捕获与同步
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

kernel_a<<<grid, block, 0, stream1>>>(...);
cudaEventRecord(event1, stream1);

// stream2等待stream1
cudaStreamWaitEvent(stream2, event1);
kernel_b<<<grid, block, 0, stream2>>>(...);

// 合并回主流
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);
kernel_c<<<grid, block, 0, stream1>>>(...);

cudaStreamEndCapture(stream1, &graph);
```

### 20.1.3 手动Graph构建与节点管理

对于需要精细控制的场景，手动构建Graph提供了最大的灵活性：

```cpp
// 手动创建Graph节点
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// 添加内核节点
cudaGraphNode_t kernelNode1, kernelNode2;
cudaKernelNodeParams kernelParams = {0};
kernelParams.func = (void*)myKernel;
kernelParams.gridDim = gridDim;
kernelParams.blockDim = blockDim;
kernelParams.sharedMemBytes = 0;
kernelParams.kernelParams = kernelArgs;

cudaGraphAddKernelNode(&kernelNode1, graph, nullptr, 0, &kernelParams);

// 添加内存拷贝节点
cudaGraphNode_t memcpyNode;
cudaMemcpy3DParms memcpyParams = {0};
// 配置memcpyParams...
cudaGraphAddMemcpyNode(&memcpyNode, graph, &kernelNode1, 1, &memcpyParams);

// 设置依赖关系
cudaGraphNode_t dependencies[] = {kernelNode1, memcpyNode};
cudaGraphAddKernelNode(&kernelNode2, graph, dependencies, 2, &kernelParams2);
```

Graph支持的节点类型包括：
- Kernel节点：GPU内核执行
- Memcpy节点：内存传输操作
- Memset节点：内存初始化
- Host节点：CPU回调函数
- Child Graph节点：嵌套子图
- Event节点：事件记录与等待
- Conditional节点：条件执行（CUDA 12.3+）

### 20.1.4 Graph更新与条件执行

动态更新Graph参数是实现灵活执行的关键：

```cpp
// 更新内核参数
cudaKernelNodeParams updatedParams;
cudaGraphKernelNodeGetParams(kernelNode, &updatedParams);
updatedParams.kernelParams[0] = newValue;
cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &updatedParams);

// 批量更新多个节点
cudaGraphExecUpdate(graphExec, graph, &updateResult);
if (updateResult != cudaGraphExecUpdateSuccess) {
    // 需要重新实例化
    cudaGraphExecDestroy(graphExec);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
}
```

CUDA 12.3引入的条件节点支持动态控制流：

```cpp
// 创建条件节点
cudaGraphNode_t conditionalNode;
cudaGraphConditionalHandle handle;
cudaGraphAddConditionalNode(&conditionalNode, graph, deps, numDeps, &handle);

// 设置条件分支
cudaGraph_t graphIf, graphElse;
// 构建分支图...
cudaGraphConditionalNodeSetParams(conditionalNode, &handle, graphIf, graphElse);
```

### 20.1.5 性能优化技巧

Graph优化的关键在于最大化并行度和最小化同步点：

1. **合并小内核**：将多个小内核合并成一个节点，减少调度开销
2. **异步操作链**：尽可能使用异步操作，避免隐式同步
3. **预分配资源**：Graph实例化时预分配所有资源，避免运行时分配
4. **图分区**：将大图分解为多个子图，便于并行执行和更新

```cpp
// 优化示例：使用持久化内核减少启动开销
__global__ void persistentKernel(volatile int* flag, ...) {
    while (*flag == 0) {
        // 执行工作
        processWork();
        __syncthreads();
    }
}
```

性能分析要点：
- 使用Nsight Systems查看Graph执行时间线
- 监控Graph实例化开销
- 分析节点间的依赖关系是否合理
- 评估Graph更新频率对性能的影响

## 20.2 内核融合策略

### 20.2.1 融合的动机与收益分析

内核融合是优化GPU程序的核心技术之一，其主要动机包括：

1. **减少内存访问**：中间结果保持在寄存器或共享内存中，避免全局内存往返
2. **降低启动开销**：减少内核启动次数，特别是对小规模问题
3. **提高数据局部性**：融合后的内核能更好地利用缓存
4. **增加并行机会**：跨操作的并行优化

融合收益的量化分析：
```
未融合：Kernel1(读A,写B) + Kernel2(读B,写C) + Kernel3(读C,写D)
内存访问：2N + 2N + 2N = 6N

融合后：FusedKernel(读A,写D)  
内存访问：2N

理论加速比：3x（仅考虑内存带宽）
```

实际收益受多个因素影响：
- 算术强度（compute/memory ratio）
- 寄存器压力
- 共享内存使用量
- 占用率变化

### 20.2.2 垂直融合技术

垂直融合（Vertical Fusion）将producer-consumer关系的内核合并：

```cpp
// 未融合版本
__global__ void relu(float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void add_bias(float* x, float* bias, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + bias[idx % BIAS_SIZE];
    }
}

// 垂直融合版本
__global__ void fused_relu_bias(float* x, float* bias, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fmaxf(0.0f, x[idx]);  // ReLU
        y[idx] = val + bias[idx % BIAS_SIZE];  // Add bias
    }
}
```

高级垂直融合模式：

```cpp
// 多层融合with共享内存优化
template<int TILE_SIZE>
__global__ void fused_conv_bn_relu(
    float* input, float* weight, float* bn_params,
    float* output, int H, int W, int C) {
    
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    // 1. 卷积计算
    float conv_result = 0.0f;
    // 卷积逻辑...
    
    // 2. BatchNorm（融合在寄存器级）
    float mean = bn_params[0];
    float var = bn_params[1];
    float gamma = bn_params[2];
    float beta = bn_params[3];
    
    float normalized = (conv_result - mean) / sqrtf(var + 1e-5f);
    float bn_result = gamma * normalized + beta;
    
    // 3. ReLU激活
    output[idx] = fmaxf(0.0f, bn_result);
}
```

### 20.2.3 水平融合技术

水平融合（Horizontal Fusion）将独立的操作合并以提高硬件利用率：

```cpp
// 水平融合多个独立的GEMV操作
__global__ void fused_multi_gemv(
    float* A1, float* x1, float* y1,
    float* A2, float* x2, float* y2,
    int M, int N) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum1 = 0.0f, sum2 = 0.0f;
        
        // 交错执行两个GEMV，提高指令级并行
        for (int col = 0; col < N; col++) {
            sum1 += A1[row * N + col] * x1[col];
            sum2 += A2[row * N + col] * x2[col];
        }
        
        y1[row] = sum1;
        y2[row] = sum2;
    }
}
```

动态批处理融合：

```cpp
// 将多个小批次请求融合执行
template<int MAX_BATCH>
__global__ void fused_batch_inference(
    float** inputs, float** outputs,
    int* sizes, int batch_count) {
    
    __shared__ int work_assignment[MAX_BATCH];
    
    // 动态工作分配
    if (threadIdx.x == 0) {
        int offset = 0;
        for (int b = 0; b < batch_count; b++) {
            work_assignment[b] = offset;
            offset += sizes[b];
        }
    }
    __syncthreads();
    
    // 并行处理多个批次
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = find_batch(global_idx, work_assignment);
    int local_idx = global_idx - work_assignment[batch_id];
    
    if (batch_id < batch_count && local_idx < sizes[batch_id]) {
        process_item(inputs[batch_id], outputs[batch_id], local_idx);
    }
}
```

### 20.2.4 Element-wise操作融合

Element-wise操作是融合的理想目标，因为它们通常是memory-bound的：

```cpp
// 融合复杂的element-wise表达式
template<typename T>
__global__ void fused_gelu_dropout_scale(
    T* input, T* output, float* dropout_mask,
    float scale, float dropout_prob, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    T x = input[idx];
    
    // GELU: x * Φ(x)
    const T cdf_const = 0.5f;
    const T sqrt_2_over_pi = 0.7978845608028654f;
    const T fitting_const = 0.044715f;
    
    T x_cubed = x * x * x;
    T tanh_arg = sqrt_2_over_pi * (x + fitting_const * x_cubed);
    T tanh_result = tanhf(tanh_arg);
    T gelu_result = cdf_const * x * (1.0f + tanh_result);
    
    // Dropout
    float keep_prob = 1.0f - dropout_prob;
    T dropout_result = dropout_mask[idx] > dropout_prob ? 
                       gelu_result / keep_prob : 0;
    
    // Scale
    output[idx] = dropout_result * scale;
}
```

向量化融合优化：

```cpp
// 使用向量化加载/存储提高带宽利用率
__global__ void fused_elementwise_vec4(
    float4* input1, float4* input2, float4* output, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float4 a = input1[idx];
    float4 b = input2[idx];
    
    // 融合多个操作
    float4 result;
    result.x = fmaf(a.x, b.x, 1.0f);  // a*b+1
    result.y = fmaf(a.y, b.y, 1.0f);
    result.z = fmaf(a.z, b.z, 1.0f);
    result.w = fmaf(a.w, b.w, 1.0f);
    
    // ReLU
    result.x = fmaxf(result.x, 0.0f);
    result.y = fmaxf(result.y, 0.0f);
    result.z = fmaxf(result.z, 0.0f);
    result.w = fmaxf(result.w, 0.0f);
    
    output[idx] = result;
}
```

### 20.2.5 Reduction融合技术

Reduction操作的融合需要特殊考虑，因为它们改变了数据维度：

```cpp
// 融合reduction和后续操作
template<int BLOCK_SIZE>
__global__ void fused_reduction_softmax(
    float* input, float* output, int M, int N) {
    
    extern __shared__ float shared[];
    float* row_max = shared;
    float* row_sum = &shared[BLOCK_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Phase 1: 找到行最大值（融合）
    float thread_max = -FLT_MAX;
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, input[row * N + col]);
    }
    
    row_max[tid] = thread_max;
    __syncthreads();
    
    // 树形归约找最大值
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            row_max[tid] = fmaxf(row_max[tid], row_max[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_val = row_max[0];
    
    // Phase 2: 计算exp和sum（融合）
    float thread_sum = 0.0f;
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float exp_val = expf(input[row * N + col] - max_val);
        thread_sum += exp_val;
        // 临时存储exp值以复用
        output[row * N + col] = exp_val;
    }
    
    row_sum[tid] = thread_sum;
    __syncthreads();
    
    // 树形归约求和
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            row_sum[tid] += row_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_val = row_sum[0];
    
    // Phase 3: 归一化（融合）
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        output[row * N + col] /= sum_val;
    }
}
```

## 20.3 自动调优框架

### 20.3.1 Profile-Guided Optimization

基于性能剖析的优化是自动调优的基础：

```cpp
class AutoTuner {
private:
    struct KernelConfig {
        dim3 gridDim;
        dim3 blockDim;
        int sharedMem;
        std::map<std::string, int> params;
        float executionTime;
    };
    
    std::vector<KernelConfig> searchSpace;
    
public:
    void profileKernel(void* kernel, void** args, KernelConfig& config) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            cudaLaunchKernel(kernel, config.gridDim, config.blockDim,
                           args, config.sharedMem, 0);
        }
        cudaDeviceSynchronize();
        
        // Profile
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            cudaLaunchKernel(kernel, config.gridDim, config.blockDim,
                           args, config.sharedMem, 0);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        config.executionTime = milliseconds / 100.0f;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    KernelConfig findBestConfig(void* kernel, void** args) {
        KernelConfig bestConfig;
        float bestTime = FLT_MAX;
        
        for (auto& config : searchSpace) {
            profileKernel(kernel, args, config);
            if (config.executionTime < bestTime) {
                bestTime = config.executionTime;
                bestConfig = config;
            }
        }
        
        return bestConfig;
    }
};
```

### 20.3.2 搜索空间定义与剪枝

有效的搜索空间定义是快速收敛的关键：

```cpp
class SearchSpaceGenerator {
private:
    struct Constraints {
        int maxThreadsPerBlock;
        int maxSharedMemory;
        int warpSize;
        int smCount;
    };
    
public:
    std::vector<KernelConfig> generateSearchSpace(
        int problemSize, Constraints& hw) {
        
        std::vector<KernelConfig> configs;
        
        // 生成block size候选
        std::vector<int> blockSizes = {64, 128, 256, 512, 1024};
        
        // 生成tile size候选
        std::vector<int> tileSizes = {8, 16, 32, 64};
        
        // 生成unroll factor候选
        std::vector<int> unrollFactors = {1, 2, 4, 8};
        
        for (int bs : blockSizes) {
            if (bs > hw.maxThreadsPerBlock) continue;
            
            for (int ts : tileSizes) {
                int sharedMem = ts * ts * sizeof(float);
                if (sharedMem > hw.maxSharedMemory) continue;
                
                for (int uf : unrollFactors) {
                    KernelConfig config;
                    config.blockDim = dim3(bs);
                    config.gridDim = dim3((problemSize + bs - 1) / bs);
                    config.sharedMem = sharedMem;
                    config.params["TILE_SIZE"] = ts;
                    config.params["UNROLL_FACTOR"] = uf;
                    
                    // 启发式剪枝
                    if (isValidConfig(config, hw)) {
                        configs.push_back(config);
                    }
                }
            }
        }
        
        return configs;
    }
    
    bool isValidConfig(KernelConfig& config, Constraints& hw) {
        // 占用率检查
        int blocksPerSM = hw.maxThreadsPerBlock / config.blockDim.x;
        if (blocksPerSM < 1) return false;
        
        // 寄存器压力估计
        int estimatedRegs = config.params["TILE_SIZE"] * 2;
        if (estimatedRegs > 255) return false;
        
        return true;
    }
};
```

### 20.3.3 贝叶斯优化策略

贝叶斯优化通过建立性能模型来指导搜索：

```cpp
class BayesianOptimizer {
private:
    // 高斯过程模型
    class GaussianProcess {
    public:
        void fit(std::vector<KernelConfig>& configs,
                std::vector<float>& performances) {
            // 构建协方差矩阵
            // 使用RBF核函数
        }
        
        std::pair<float, float> predict(KernelConfig& config) {
            // 返回均值和方差
            float mean = 0.0f;
            float variance = 0.0f;
            // 计算预测...
            return {mean, variance};
        }
    };
    
    GaussianProcess gp;
    
    float acquisitionFunction(KernelConfig& config) {
        auto [mean, variance] = gp.predict(config);
        
        // Expected Improvement (EI)
        float best_so_far = getBestPerformance();
        float improvement = best_so_far - mean;
        
        if (variance < 1e-6f) return 0.0f;
        
        float Z = improvement / sqrtf(variance);
        float ei = improvement * normcdf(Z) + sqrtf(variance) * normpdf(Z);
        
        return ei;
    }
    
public:
    KernelConfig optimize(void* kernel, void** args, int iterations) {
        std::vector<KernelConfig> evaluated;
        std::vector<float> performances;
        
        // 初始随机采样
        for (int i = 0; i < 10; i++) {
            KernelConfig config = randomSample();
            float perf = evaluateKernel(kernel, args, config);
            evaluated.push_back(config);
            performances.push_back(perf);
        }
        
        // 贝叶斯优化迭代
        for (int iter = 0; iter < iterations; iter++) {
            gp.fit(evaluated, performances);
            
            // 选择下一个采样点
            KernelConfig nextConfig = argmax(acquisitionFunction);
            float perf = evaluateKernel(kernel, args, nextConfig);
            
            evaluated.push_back(nextConfig);
            performances.push_back(perf);
        }
        
        return getBestConfig(evaluated, performances);
    }
};
```

### 20.3.4 遗传算法优化

遗传算法适合离散的大规模搜索空间：

```cpp
class GeneticOptimizer {
private:
    struct Individual {
        KernelConfig config;
        float fitness;
    };
    
    std::vector<Individual> population;
    
    Individual crossover(Individual& parent1, Individual& parent2) {
        Individual child;
        
        // 均匀交叉
        if (rand() % 2) {
            child.config.blockDim = parent1.config.blockDim;
        } else {
            child.config.blockDim = parent2.config.blockDim;
        }
        
        // 参数交叉
        for (auto& [key, value] : parent1.config.params) {
            if (rand() % 2) {
                child.config.params[key] = value;
            } else {
                child.config.params[key] = parent2.config.params[key];
            }
        }
        
        return child;
    }
    
    void mutate(Individual& individual, float mutationRate) {
        if (randf() < mutationRate) {
            // 随机改变block size
            int blockSizes[] = {64, 128, 256, 512};
            int idx = rand() % 4;
            individual.config.blockDim.x = blockSizes[idx];
        }
        
        // 参数突变
        for (auto& [key, value] : individual.config.params) {
            if (randf() < mutationRate) {
                value = mutateParameter(key, value);
            }
        }
    }
    
public:
    KernelConfig evolve(void* kernel, void** args, int generations) {
        const int POP_SIZE = 50;
        const float MUTATION_RATE = 0.1f;
        const float ELITE_RATIO = 0.2f;
        
        // 初始化种群
        population = generateRandomPopulation(POP_SIZE);
        evaluatePopulation(kernel, args);
        
        for (int gen = 0; gen < generations; gen++) {
            std::sort(population.begin(), population.end(),
                     [](auto& a, auto& b) { return a.fitness > b.fitness; });
            
            // 精英保留
            int eliteSize = POP_SIZE * ELITE_RATIO;
            std::vector<Individual> newPopulation(
                population.begin(), population.begin() + eliteSize);
            
            // 交叉和突变
            while (newPopulation.size() < POP_SIZE) {
                Individual parent1 = tournamentSelection();
                Individual parent2 = tournamentSelection();
                Individual child = crossover(parent1, parent2);
                mutate(child, MUTATION_RATE);
                newPopulation.push_back(child);
            }
            
            population = newPopulation;
            evaluatePopulation(kernel, args);
        }
        
        return population[0].config;
    }
};
```

### 20.3.5 性能模型与预测

构建准确的性能模型可以减少实际评估次数：

```cpp
class PerformanceModel {
private:
    struct HardwareFeatures {
        float peakGFLOPS;
        float memoryBandwidth;
        int smCount;
        int l2CacheSize;
    };
    
    HardwareFeatures hw;
    
public:
    float predictKernelTime(KernelConfig& config, 
                           int computeOps, int memoryOps) {
        // Roofline模型预测
        float arithmeticIntensity = (float)computeOps / memoryOps;
        
        float computeTime = computeOps / (hw.peakGFLOPS * 1e9);
        float memoryTime = memoryOps / (hw.memoryBandwidth * 1e9);
        
        // 考虑占用率
        int threadsPerSM = config.blockDim.x * config.blockDim.y;
        float occupancy = min(1.0f, threadsPerSM / 2048.0f);
        
        // 考虑缓存效果
        float cacheHitRate = estimateCacheHitRate(config);
        float effectiveMemTime = memoryTime * (1.0f - cacheHitRate * 0.8f);
        
        return max(computeTime / occupancy, effectiveMemTime);
    }
    
    float estimateCacheHitRate(KernelConfig& config) {
        int workingSet = config.params["TILE_SIZE"] * 
                        config.params["TILE_SIZE"] * sizeof(float);
        
        if (workingSet < hw.l2CacheSize) {
            return 0.8f;  // 高缓存命中率
        } else {
            return 0.2f;  // 低缓存命中率
        }
    }
};
```

## 20.4 JIT编译优化

### 20.4.1 NVRTC运行时编译

NVRTC（NVIDIA Runtime Compilation）允许在运行时编译CUDA代码：

```cpp
class JITCompiler {
private:
    nvrtcProgram prog;
    CUmodule module;
    std::map<std::string, CUfunction> functionCache;
    
public:
    void compileKernel(const std::string& source, 
                       const std::string& kernelName,
                       const std::vector<std::string>& options) {
        // 创建程序
        nvrtcCreateProgram(&prog, source.c_str(), 
                          (kernelName + ".cu").c_str(),
                          0, nullptr, nullptr);
        
        // 编译选项
        std::vector<const char*> opts;
        for (auto& opt : options) {
            opts.push_back(opt.c_str());
        }
        
        // 编译
        nvrtcResult compileResult = nvrtcCompileProgram(prog, 
                                                        opts.size(), 
                                                        opts.data());
        
        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            nvrtcGetProgramLogSize(prog, &logSize);
            std::vector<char> log(logSize);
            nvrtcGetProgramLog(prog, log.data());
            throw std::runtime_error("Compilation failed: " + 
                                   std::string(log.data()));
        }
        
        // 获取PTX
        size_t ptxSize;
        nvrtcGetPTXSize(prog, &ptxSize);
        std::vector<char> ptx(ptxSize);
        nvrtcGetPTX(prog, ptx.data());
        
        // 加载模块
        cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0);
        
        // 获取函数
        CUfunction kernel;
        cuModuleGetFunction(&kernel, module, kernelName.c_str());
        functionCache[kernelName] = kernel;
    }
    
    template<typename... Args>
    void launchKernel(const std::string& kernelName,
                     dim3 grid, dim3 block,
                     Args... args) {
        CUfunction kernel = functionCache[kernelName];
        
        void* kernelArgs[] = { &args... };
        
        cuLaunchKernel(kernel,
                      grid.x, grid.y, grid.z,
                      block.x, block.y, block.z,
                      0, 0,
                      kernelArgs, nullptr);
    }
};
```

### 20.4.2 模板元编程与代码生成

使用模板生成特化的内核代码：

```cpp
class KernelGenerator {
private:
    std::string generateGEMMKernel(int M, int N, int K, 
                                   int TILE_M, int TILE_N, int TILE_K) {
        std::stringstream ss;
        
        ss << "#define TILE_M " << TILE_M << "\n";
        ss << "#define TILE_N " << TILE_N << "\n";
        ss << "#define TILE_K " << TILE_K << "\n";
        ss << "\n";
        
        ss << "__global__ void gemm_kernel(\n";
        ss << "    const float* __restrict__ A,\n";
        ss << "    const float* __restrict__ B,\n";
        ss << "    float* __restrict__ C,\n";
        ss << "    const int M, const int N, const int K) {\n";
        ss << "\n";
        ss << "    __shared__ float As[TILE_M][TILE_K];\n";
        ss << "    __shared__ float Bs[TILE_K][TILE_N];\n";
        ss << "\n";
        ss << "    int bx = blockIdx.x, by = blockIdx.y;\n";
        ss << "    int tx = threadIdx.x, ty = threadIdx.y;\n";
        ss << "\n";
        ss << "    int row = by * TILE_M + ty;\n";
        ss << "    int col = bx * TILE_N + tx;\n";
        ss << "\n";
        ss << "    float sum = 0.0f;\n";
        ss << "\n";
        
        // 主循环
        ss << "    for (int tile = 0; tile < (K + TILE_K - 1) / TILE_K; tile++) {\n";
        ss << "        // Load tiles\n";
        ss << "        if (row < M && tile * TILE_K + tx < K) {\n";
        ss << "            As[ty][tx] = A[row * K + tile * TILE_K + tx];\n";
        ss << "        } else {\n";
        ss << "            As[ty][tx] = 0.0f;\n";
        ss << "        }\n";
        ss << "\n";
        ss << "        if (col < N && tile * TILE_K + ty < K) {\n";
        ss << "            Bs[ty][tx] = B[(tile * TILE_K + ty) * N + col];\n";
        ss << "        } else {\n";
        ss << "            Bs[ty][tx] = 0.0f;\n";
        ss << "        }\n";
        ss << "\n";
        ss << "        __syncthreads();\n";
        ss << "\n";
        
        // 计算
        ss << "        #pragma unroll\n";
        ss << "        for (int k = 0; k < TILE_K; k++) {\n";
        ss << "            sum += As[ty][k] * Bs[k][tx];\n";
        ss << "        }\n";
        ss << "\n";
        ss << "        __syncthreads();\n";
        ss << "    }\n";
        ss << "\n";
        
        // 存储结果
        ss << "    if (row < M && col < N) {\n";
        ss << "        C[row * N + col] = sum;\n";
        ss << "    }\n";
        ss << "}\n";
        
        return ss.str();
    }
    
public:
    CUfunction generateOptimizedGEMM(int M, int N, int K) {
        // 根据问题规模选择tile size
        int TILE_M, TILE_N, TILE_K;
        
        if (M * N < 1024 * 1024) {
            TILE_M = TILE_N = 16;
            TILE_K = 16;
        } else {
            TILE_M = TILE_N = 32;
            TILE_K = 8;
        }
        
        std::string kernel = generateGEMMKernel(M, N, K, 
                                               TILE_M, TILE_N, TILE_K);
        
        JITCompiler compiler;
        compiler.compileKernel(kernel, "gemm_kernel", 
                              {"-arch=sm_80", "-use_fast_math"});
        
        return compiler.getFunction("gemm_kernel");
    }
};
```

### 20.4.3 缓存策略与持久化

JIT编译结果的缓存对性能至关重要：

```cpp
class KernelCache {
private:
    struct CacheEntry {
        std::string source;
        std::string ptx;
        CUmodule module;
        std::map<std::string, CUfunction> functions;
        std::chrono::time_point<std::chrono::steady_clock> lastAccess;
    };
    
    std::unordered_map<size_t, CacheEntry> cache;
    const size_t maxCacheSize = 100;
    
    size_t computeHash(const std::string& source, 
                      const std::vector<std::string>& options) {
        std::hash<std::string> hasher;
        size_t hash = hasher(source);
        
        for (auto& opt : options) {
            hash ^= hasher(opt) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        
        return hash;
    }
    
    void evictLRU() {
        if (cache.size() < maxCacheSize) return;
        
        auto oldest = cache.begin();
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->second.lastAccess < oldest->second.lastAccess) {
                oldest = it;
            }
        }
        
        cuModuleUnload(oldest->second.module);
        cache.erase(oldest);
    }
    
public:
    CUfunction getOrCompile(const std::string& source,
                           const std::string& kernelName,
                           const std::vector<std::string>& options) {
        size_t hash = computeHash(source, options);
        
        // 检查缓存
        auto it = cache.find(hash);
        if (it != cache.end()) {
            it->second.lastAccess = std::chrono::steady_clock::now();
            return it->second.functions[kernelName];
        }
        
        // 编译新内核
        evictLRU();
        
        CacheEntry entry;
        entry.source = source;
        entry.lastAccess = std::chrono::steady_clock::now();
        
        // 编译并缓存
        compileAndCache(entry, kernelName, options);
        cache[hash] = entry;
        
        return entry.functions[kernelName];
    }
    
    void persistCache(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        
        size_t numEntries = cache.size();
        file.write(reinterpret_cast<const char*>(&numEntries), 
                  sizeof(numEntries));
        
        for (auto& [hash, entry] : cache) {
            // 保存hash
            file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
            
            // 保存PTX
            size_t ptxSize = entry.ptx.size();
            file.write(reinterpret_cast<const char*>(&ptxSize), 
                      sizeof(ptxSize));
            file.write(entry.ptx.data(), ptxSize);
        }
    }
    
    void loadCache(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return;
        
        size_t numEntries;
        file.read(reinterpret_cast<char*>(&numEntries), sizeof(numEntries));
        
        for (size_t i = 0; i < numEntries; i++) {
            size_t hash;
            file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
            
            size_t ptxSize;
            file.read(reinterpret_cast<char*>(&ptxSize), sizeof(ptxSize));
            
            CacheEntry entry;
            entry.ptx.resize(ptxSize);
            file.read(entry.ptx.data(), ptxSize);
            
            // 加载模块
            cuModuleLoadData(&entry.module, entry.ptx.data());
            cache[hash] = entry;
        }
    }
};
```

### 20.4.4 动态Kernel Specialization

根据运行时参数生成特化版本：

```cpp
template<typename T>
class DynamicKernelSpecializer {
private:
    struct Specialization {
        std::string name;
        std::string code;
        std::map<std::string, std::string> constants;
    };
    
    std::string generateSpecializedCode(const Specialization& spec) {
        std::string code = spec.code;
        
        // 替换常量
        for (auto& [placeholder, value] : spec.constants) {
            size_t pos = 0;
            while ((pos = code.find(placeholder, pos)) != std::string::npos) {
                code.replace(pos, placeholder.length(), value);
                pos += value.length();
            }
        }
        
        return code;
    }
    
public:
    CUfunction specializeForProblem(int M, int N, int K, bool useTensorCore) {
        Specialization spec;
        spec.name = "specialized_gemm";
        
        // 基础模板
        spec.code = R"(
            extern "C" __global__ void specialized_gemm(
                const TYPE* A, const TYPE* B, TYPE* C) {
                
                const int M = CONST_M;
                const int N = CONST_N;
                const int K = CONST_K;
                
                // 特化的内循环
                #if USE_TENSOR_CORE
                    // Tensor Core路径
                    wmma::fragment<...> a_frag, b_frag, c_frag;
                    // ...
                #else
                    // 常规路径
                    TYPE sum = 0;
                    #pragma unroll UNROLL_FACTOR
                    for (int k = 0; k < K; k++) {
                        sum += A[...] * B[...];
                    }
                #endif
                
                C[...] = sum;
            }
        )";
        
        // 设置常量
        spec.constants["TYPE"] = std::is_same_v<T, float> ? "float" : "half";
        spec.constants["CONST_M"] = std::to_string(M);
        spec.constants["CONST_N"] = std::to_string(N);
        spec.constants["CONST_K"] = std::to_string(K);
        spec.constants["USE_TENSOR_CORE"] = useTensorCore ? "1" : "0";
        
        // 根据K的大小选择展开因子
        int unrollFactor = 1;
        if (K <= 32) unrollFactor = K;
        else if (K <= 128) unrollFactor = 8;
        else unrollFactor = 4;
        spec.constants["UNROLL_FACTOR"] = std::to_string(unrollFactor);
        
        std::string specializedCode = generateSpecializedCode(spec);
        
        // JIT编译
        JITCompiler compiler;
        compiler.compileKernel(specializedCode, spec.name, 
                              {"-arch=sm_80", "-maxrregcount=128"});
        
        return compiler.getFunction(spec.name);
    }
};
```

### 20.4.5 PTX级优化

直接生成优化的PTX代码：

```cpp
class PTXGenerator {
private:
    std::string generateOptimizedPTX(const std::string& operation) {
        std::stringstream ptx;
        
        // PTX头部
        ptx << ".version 7.5\n";
        ptx << ".target sm_80\n";
        ptx << ".address_size 64\n";
        ptx << "\n";
        
        if (operation == "fast_exp") {
            // 快速指数函数实现
            ptx << ".visible .entry fast_exp(\n";
            ptx << "    .param .u64 .ptr.global.align 4 input,\n";
            ptx << "    .param .u64 .ptr.global.align 4 output,\n";
            ptx << "    .param .u32 n\n";
            ptx << ")\n";
            ptx << "{\n";
            ptx << "    .reg .f32 %f<4>;\n";
            ptx << "    .reg .pred %p<2>;\n";
            ptx << "    .reg .b32 %r<8>;\n";
            ptx << "    .reg .b64 %rd<8>;\n";
            ptx << "\n";
            
            // 获取线程索引
            ptx << "    mov.u32 %r1, %ctaid.x;\n";
            ptx << "    mov.u32 %r2, %ntid.x;\n";
            ptx << "    mov.u32 %r3, %tid.x;\n";
            ptx << "    mad.lo.s32 %r4, %r1, %r2, %r3;\n";
            ptx << "\n";
            
            // 边界检查
            ptx << "    ld.param.u32 %r5, [n];\n";
            ptx << "    setp.ge.u32 %p1, %r4, %r5;\n";
            ptx << "    @%p1 bra EXIT;\n";
            ptx << "\n";
            
            // 加载数据
            ptx << "    ld.param.u64 %rd1, [input];\n";
            ptx << "    cvta.to.global.u64 %rd2, %rd1;\n";
            ptx << "    mul.wide.u32 %rd3, %r4, 4;\n";
            ptx << "    add.s64 %rd4, %rd2, %rd3;\n";
            ptx << "    ld.global.f32 %f1, [%rd4];\n";
            ptx << "\n";
            
            // 快速exp近似（使用硬件特殊函数）
            ptx << "    ex2.approx.f32 %f2, %f1;\n";
            ptx << "\n";
            
            // 存储结果
            ptx << "    ld.param.u64 %rd5, [output];\n";
            ptx << "    cvta.to.global.u64 %rd6, %rd5;\n";
            ptx << "    add.s64 %rd7, %rd6, %rd3;\n";
            ptx << "    st.global.f32 [%rd7], %f2;\n";
            ptx << "\n";
            
            ptx << "EXIT:\n";
            ptx << "    ret;\n";
            ptx << "}\n";
        }
        
        return ptx.str();
    }
    
public:
    CUfunction compilePTX(const std::string& operation) {
        std::string ptxCode = generateOptimizedPTX(operation);
        
        // 使用cuModuleLoadData加载PTX
        CUmodule module;
        CUfunction kernel;
        
        cuModuleLoadData(&module, ptxCode.c_str());
        cuModuleGetFunction(&kernel, module, operation.c_str());
        
        return kernel;
    }
};

## 20.5 案例：端到端推理优化

本节将以Transformer模型的推理优化为例，展示如何综合运用CUDA Graph、内核融合、JIT编译等技术实现端到端的性能优化。我们将从一个未优化的baseline实现开始，逐步应用各种优化技术，最终实现10x以上的性能提升。

### 20.5.1 Baseline实现分析

首先分析未优化的Transformer推理实现：

```cpp
// Baseline实现：逐层执行，无融合
class TransformerBaseline {
private:
    // 各层权重
    float *wq, *wk, *wv, *wo;
    float *w1, *w2;
    float *ln1_gamma, *ln1_beta;
    float *ln2_gamma, *ln2_beta;
    
public:
    void forward(float* input, float* output, 
                int batch, int seq_len, int hidden_dim) {
        
        // 临时缓冲区
        float *q, *k, *v, *attn_scores, *attn_output;
        float *ffn_hidden, *residual;
        
        // Layer Norm 1
        layerNorm<<<grid, block>>>(input, ln1_gamma, ln1_beta, 
                                   residual, batch * seq_len, hidden_dim);
        
        // Multi-Head Attention
        // QKV投影（3个独立内核）
        gemm<<<grid, block>>>(residual, wq, q, batch * seq_len, 
                             hidden_dim, hidden_dim);
        gemm<<<grid, block>>>(residual, wk, k, batch * seq_len, 
                             hidden_dim, hidden_dim);
        gemm<<<grid, block>>>(residual, wv, v, batch * seq_len, 
                             hidden_dim, hidden_dim);
        
        // Attention计算
        computeAttention<<<grid, block>>>(q, k, v, attn_scores, 
                                         batch, seq_len, hidden_dim);
        
        // Output投影
        gemm<<<grid, block>>>(attn_scores, wo, attn_output, 
                             batch * seq_len, hidden_dim, hidden_dim);
        
        // Residual连接
        elementwiseAdd<<<grid, block>>>(input, attn_output, residual,
                                       batch * seq_len * hidden_dim);
        
        // Layer Norm 2
        layerNorm<<<grid, block>>>(residual, ln2_gamma, ln2_beta, 
                                   output, batch * seq_len, hidden_dim);
        
        // FFN
        gemm<<<grid, block>>>(output, w1, ffn_hidden, batch * seq_len,
                             hidden_dim, 4 * hidden_dim);
        
        // GELU激活
        gelu<<<grid, block>>>(ffn_hidden, ffn_hidden, 
                             batch * seq_len * 4 * hidden_dim);
        
        // FFN输出
        gemm<<<grid, block>>>(ffn_hidden, w2, output, batch * seq_len,
                             4 * hidden_dim, hidden_dim);
        
        // 最终residual
        elementwiseAdd<<<grid, block>>>(residual, output, output,
                                       batch * seq_len * hidden_dim);
    }
};

// 性能分析结果：
// - 12个独立的内核启动
// - 大量中间结果写入全局内存
// - 无内核间并行
// - 小batch时GPU利用率低
```

### 20.5.2 Graph优化实现

使用CUDA Graph减少内核启动开销：

```cpp
class TransformerGraphOptimized {
private:
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    // 预分配的缓冲区
    struct BufferPool {
        float *q, *k, *v;
        float *attn_scores;
        float *attn_output;
        float *ffn_hidden;
        float *residual[2];  // 双缓冲
        int current_residual = 0;
        
        void swap_residual() {
            current_residual = 1 - current_residual;
        }
    } buffers;
    
    void buildGraph(int batch, int seq_len, int hidden_dim) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 开始捕获
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        
        // 使用多流并行
        cudaStream_t stream_qkv[3];
        cudaEvent_t event_qkv[3];
        
        for (int i = 0; i < 3; i++) {
            cudaStreamCreate(&stream_qkv[i]);
            cudaEventCreate(&event_qkv[i]);
        }
        
        // Layer Norm 1
        layerNorm<<<grid, block, 0, stream>>>(
            input, ln1_gamma, ln1_beta, buffers.residual[0],
            batch * seq_len, hidden_dim);
        
        // 并行QKV投影
        cudaEventRecord(event_start, stream);
        
        cudaStreamWaitEvent(stream_qkv[0], event_start);
        gemm<<<grid, block, 0, stream_qkv[0]>>>(
            buffers.residual[0], wq, buffers.q,
            batch * seq_len, hidden_dim, hidden_dim);
        cudaEventRecord(event_qkv[0], stream_qkv[0]);
        
        cudaStreamWaitEvent(stream_qkv[1], event_start);
        gemm<<<grid, block, 0, stream_qkv[1]>>>(
            buffers.residual[0], wk, buffers.k,
            batch * seq_len, hidden_dim, hidden_dim);
        cudaEventRecord(event_qkv[1], stream_qkv[1]);
        
        cudaStreamWaitEvent(stream_qkv[2], event_start);
        gemm<<<grid, block, 0, stream_qkv[2]>>>(
            buffers.residual[0], wv, buffers.v,
            batch * seq_len, hidden_dim, hidden_dim);
        cudaEventRecord(event_qkv[2], stream_qkv[2]);
        
        // 等待QKV完成
        for (int i = 0; i < 3; i++) {
            cudaStreamWaitEvent(stream, event_qkv[i]);
        }
        
        // Attention计算
        computeAttention<<<grid, block, 0, stream>>>(
            buffers.q, buffers.k, buffers.v, buffers.attn_scores,
            batch, seq_len, hidden_dim);
        
        // 继续执行...
        
        // 结束捕获
        cudaStreamEndCapture(stream, &graph);
        
        // 实例化
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    }
    
public:
    void forward(float* input, float* output,
                int batch, int seq_len, int hidden_dim) {
        // 直接执行预构建的图
        cudaGraphLaunch(graphExec, 0);
        cudaStreamSynchronize(0);
    }
};

// 优化效果：
// - 单次API调用执行整个网络
// - QKV投影并行执行
// - 消除CPU-GPU同步开销
// - 约2x加速
```

### 20.5.3 内核融合优化

融合多个操作减少内存访问：

```cpp
// 融合的Attention内核
template<int HEAD_DIM, int NUM_HEADS>
__global__ void fusedMultiHeadAttention(
    float* input,      // [batch, seq_len, hidden_dim]
    float* wqkv,       // [hidden_dim, 3 * hidden_dim]
    float* wo,         // [hidden_dim, hidden_dim]
    float* output,     // [batch, seq_len, hidden_dim]
    float* ln_gamma, float* ln_beta,
    int batch, int seq_len) {
    
    const int hidden_dim = HEAD_DIM * NUM_HEADS;
    extern __shared__ float shared_mem[];
    
    // 共享内存布局
    float* q_smem = shared_mem;
    float* k_smem = &q_smem[HEAD_DIM * blockDim.x];
    float* v_smem = &k_smem[HEAD_DIM * seq_len];
    float* attn_smem = &v_smem[HEAD_DIM * seq_len];
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (seq_idx >= seq_len) return;
    
    // Step 1: LayerNorm + QKV投影（融合）
    float norm_scale = 1.0f / sqrtf((float)hidden_dim);
    float mean = 0.0f, var = 0.0f;
    
    // 计算LayerNorm统计量
    for (int i = 0; i < hidden_dim; i++) {
        float val = input[batch_idx * seq_len * hidden_dim + 
                         seq_idx * hidden_dim + i];
        mean += val;
    }
    mean /= hidden_dim;
    
    for (int i = 0; i < hidden_dim; i++) {
        float val = input[batch_idx * seq_len * hidden_dim + 
                         seq_idx * hidden_dim + i];
        var += (val - mean) * (val - mean);
    }
    var = sqrtf(var / hidden_dim + 1e-6f);
    
    // 融合LayerNorm和QKV投影
    float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
    
    for (int i = 0; i < hidden_dim; i++) {
        float normalized = (input[...] - mean) / var;
        normalized = normalized * ln_gamma[i] + ln_beta[i];
        
        // 直接投影到QKV
        int qkv_offset = head_idx * HEAD_DIM;
        q_val += normalized * wqkv[i * 3 * hidden_dim + qkv_offset];
        k_val += normalized * wqkv[i * 3 * hidden_dim + hidden_dim + qkv_offset];
        v_val += normalized * wqkv[i * 3 * hidden_dim + 2 * hidden_dim + qkv_offset];
    }
    
    // 存储到共享内存
    q_smem[threadIdx.x * HEAD_DIM + threadIdx.y] = q_val;
    k_smem[seq_idx * HEAD_DIM + threadIdx.y] = k_val;
    v_smem[seq_idx * HEAD_DIM + threadIdx.y] = v_val;
    
    __syncthreads();
    
    // Step 2: Attention计算（融合softmax）
    float attn_sum = 0.0f;
    float max_score = -FLT_MAX;
    
    // Online softmax（单pass）
    for (int i = 0; i < seq_len; i++) {
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            score += q_smem[threadIdx.x * HEAD_DIM + d] * 
                    k_smem[i * HEAD_DIM + d];
        }
        score *= norm_scale;
        
        // Online softmax更新
        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_score = expf(score - max_score);
        
        attn_sum = attn_sum * expf(old_max - max_score) + exp_score;
        attn_smem[i] = exp_score;
    }
    
    // Step 3: 加权求和V + 输出投影（融合）
    float out_val = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        float attn_weight = attn_smem[i] / attn_sum;
        for (int d = 0; d < HEAD_DIM; d++) {
            out_val += attn_weight * v_smem[i * HEAD_DIM + d];
        }
    }
    
    // 输出投影和residual连接（融合）
    int out_idx = batch_idx * seq_len * hidden_dim + 
                 seq_idx * hidden_dim + head_idx * HEAD_DIM;
    
    for (int d = 0; d < HEAD_DIM; d++) {
        float proj_val = 0.0f;
        for (int h = 0; h < NUM_HEADS; h++) {
            proj_val += out_val * wo[...];
        }
        
        // Residual连接
        output[out_idx + d] = input[out_idx + d] + proj_val;
    }
}

// 融合的FFN层
__global__ void fusedFFN(
    float* input, float* output,
    float* w1, float* w2,
    float* ln_gamma, float* ln_beta,
    int batch_seq_len, int hidden_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_seq_len) return;
    
    // 融合LayerNorm + FFN + GELU + 投影
    float mean = 0.0f, var = 0.0f;
    
    // LayerNorm
    for (int i = 0; i < hidden_dim; i++) {
        mean += input[idx * hidden_dim + i];
    }
    mean /= hidden_dim;
    
    for (int i = 0; i < hidden_dim; i++) {
        float val = input[idx * hidden_dim + i];
        var += (val - mean) * (val - mean);
    }
    var = sqrtf(var / hidden_dim + 1e-6f);
    
    // FFN with融合GELU
    for (int out_i = 0; out_i < 4 * hidden_dim; out_i++) {
        float sum = 0.0f;
        
        for (int in_i = 0; in_i < hidden_dim; in_i++) {
            float normalized = (input[idx * hidden_dim + in_i] - mean) / var;
            normalized = normalized * ln_gamma[in_i] + ln_beta[in_i];
            sum += normalized * w1[in_i * 4 * hidden_dim + out_i];
        }
        
        // GELU激活（融合）
        float gelu = 0.5f * sum * (1.0f + tanhf(0.7978845608f * 
                    (sum + 0.044715f * sum * sum * sum)));
        
        // 第二层投影（融合）
        for (int final_i = 0; final_i < hidden_dim; final_i++) {
            atomicAdd(&output[idx * hidden_dim + final_i],
                     gelu * w2[out_i * hidden_dim + final_i]);
        }
    }
}

// 优化效果：
// - 减少70%的内存访问
// - 消除中间结果存储
// - 约5x加速
```

### 20.5.4 动态批处理优化

处理可变长度输入的优化：

```cpp
class DynamicBatchingOptimizer {
private:
    struct Request {
        float* input;
        float* output;
        int seq_len;
        int request_id;
    };
    
    std::queue<Request> pending_requests;
    std::mutex queue_mutex;
    
    // Padding策略
    int getPaddedLength(int seq_len) {
        // 向上取整到32的倍数（warp对齐）
        return ((seq_len + 31) / 32) * 32;
    }
    
public:
    void processBatch() {
        std::vector<Request> batch;
        int total_tokens = 0;
        int max_seq_len = 0;
        
        // 收集请求形成批次
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            
            while (!pending_requests.empty() && batch.size() < 32) {
                Request req = pending_requests.front();
                
                // 检查是否超过token预算
                if (total_tokens + req.seq_len > 4096) break;
                
                pending_requests.pop();
                batch.push_back(req);
                total_tokens += req.seq_len;
                max_seq_len = std::max(max_seq_len, req.seq_len);
            }
        }
        
        if (batch.empty()) return;
        
        // 动态选择执行策略
        if (batch.size() == 1) {
            // 单请求：使用专门优化的小batch内核
            executeSingleRequest(batch[0]);
        } else if (max_seq_len <= 128) {
            // 短序列：使用融合内核
            executeShortSequenceBatch(batch, max_seq_len);
        } else {
            // 长序列：使用分块处理
            executeLongSequenceBatch(batch, max_seq_len);
        }
    }
    
    void executeShortSequenceBatch(std::vector<Request>& batch, 
                                  int max_seq_len) {
        int padded_len = getPaddedLength(max_seq_len);
        int batch_size = batch.size();
        
        // 分配连续内存
        float* batch_input, *batch_output;
        cudaMalloc(&batch_input, batch_size * padded_len * hidden_dim * sizeof(float));
        cudaMalloc(&batch_output, batch_size * padded_len * hidden_dim * sizeof(float));
        
        // 打包输入（with padding）
        for (int b = 0; b < batch_size; b++) {
            cudaMemcpy2D(batch_input + b * padded_len * hidden_dim,
                        padded_len * sizeof(float),
                        batch[b].input,
                        batch[b].seq_len * sizeof(float),
                        batch[b].seq_len * sizeof(float),
                        hidden_dim,
                        cudaMemcpyDeviceToDevice);
            
            // Padding位置设置为0
            if (batch[b].seq_len < padded_len) {
                cudaMemset(batch_input + b * padded_len * hidden_dim + 
                          batch[b].seq_len * hidden_dim,
                          0,
                          (padded_len - batch[b].seq_len) * hidden_dim * sizeof(float));
            }
        }
        
        // 执行融合内核
        dim3 grid(padded_len / 32, batch_size);
        dim3 block(32, 32);  // warp per sequence position
        
        fusedTransformerKernel<<<grid, block>>>(
            batch_input, batch_output,
            weights, batch_size, padded_len, hidden_dim);
        
        // 解包输出
        for (int b = 0; b < batch_size; b++) {
            cudaMemcpy2D(batch[b].output,
                        batch[b].seq_len * sizeof(float),
                        batch_output + b * padded_len * hidden_dim,
                        padded_len * sizeof(float),
                        batch[b].seq_len * sizeof(float),
                        hidden_dim,
                        cudaMemcpyDeviceToDevice);
        }
        
        cudaFree(batch_input);
        cudaFree(batch_output);
    }
};

// 优化效果：
// - 提高GPU利用率
// - 减少内存碎片
// - 约3x加速（相比逐个处理）
```

### 20.5.5 延迟隐藏技术

通过流水线化隐藏延迟：

```cpp
class PipelinedTransformer {
private:
    static const int NUM_STAGES = 3;
    cudaStream_t streams[NUM_STAGES];
    cudaEvent_t events[NUM_STAGES];
    
    // 三级流水线缓冲
    struct PipelineBuffer {
        float* data[NUM_STAGES];
        int current_stage = 0;
        
        float* get_current() { return data[current_stage]; }
        float* get_next() { 
            return data[(current_stage + 1) % NUM_STAGES]; 
        }
        void advance() { 
            current_stage = (current_stage + 1) % NUM_STAGES; 
        }
    } buffers;
    
public:
    void pipelinedForward(float* input, float* output,
                         int num_layers, int batch, int seq_len) {
        
        // 初始化流水线
        for (int i = 0; i < NUM_STAGES; i++) {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&events[i]);
        }
        
        // 预热流水线
        // Stage 0: 第1层的前半部分
        layerForwardPart1<<<grid, block, 0, streams[0]>>>(
            input, buffers.data[0], 0);
        cudaEventRecord(events[0], streams[0]);
        
        // 主循环：流水线执行
        for (int layer = 0; layer < num_layers; layer++) {
            int stage = layer % NUM_STAGES;
            int prev_stage = (stage - 1 + NUM_STAGES) % NUM_STAGES;
            int next_stage = (stage + 1) % NUM_STAGES;
            
            // 等待前一阶段
            if (layer > 0) {
                cudaStreamWaitEvent(streams[stage], events[prev_stage]);
            }
            
            // 执行当前层
            if (layer < num_layers - 1) {
                // 并行执行：当前层后半部分 + 下一层前半部分
                
                // 当前层后半部分
                layerForwardPart2<<<grid, block, 0, streams[stage]>>>(
                    buffers.data[prev_stage], buffers.data[stage], layer);
                
                // 下一层前半部分（在不同流上）
                cudaStreamWaitEvent(streams[next_stage], events[stage]);
                layerForwardPart1<<<grid, block, 0, streams[next_stage]>>>(
                    buffers.data[stage], buffers.data[next_stage], layer + 1);
                cudaEventRecord(events[next_stage], streams[next_stage]);
            } else {
                // 最后一层
                layerForwardPart2<<<grid, block, 0, streams[stage]>>>(
                    buffers.data[prev_stage], output, layer);
            }
            
            cudaEventRecord(events[stage], streams[stage]);
        }
        
        // 同步所有流
        for (int i = 0; i < NUM_STAGES; i++) {
            cudaStreamSynchronize(streams[i]);
        }
    }
};

// 优化效果：
// - 隐藏内存传输延迟
// - 提高计算/内存重叠
// - 约1.5x额外加速
```

### 20.5.6 性能对比与分析

综合所有优化技术的最终实现：

```cpp
class OptimizedTransformerInference {
private:
    // 组合所有优化技术
    TransformerGraphOptimized graph_optimizer;
    DynamicBatchingOptimizer batch_optimizer;
    PipelinedTransformer pipeline_optimizer;
    JITCompiler jit_compiler;
    KernelCache kernel_cache;
    
    struct PerformanceMetrics {
        float baseline_time;
        float graph_time;
        float fusion_time;
        float batching_time;
        float pipeline_time;
        float combined_time;
        
        void print() {
            printf("Performance Analysis:\n");
            printf("Baseline:        %.2f ms\n", baseline_time);
            printf("With Graph:      %.2f ms (%.2fx)\n", 
                   graph_time, baseline_time / graph_time);
            printf("With Fusion:     %.2f ms (%.2fx)\n", 
                   fusion_time, baseline_time / fusion_time);
            printf("With Batching:   %.2f ms (%.2fx)\n", 
                   batching_time, baseline_time / batching_time);
            printf("With Pipeline:   %.2f ms (%.2fx)\n", 
                   pipeline_time, baseline_time / pipeline_time);
            printf("Combined:        %.2f ms (%.2fx)\n", 
                   combined_time, baseline_time / combined_time);
        }
    } metrics;
    
public:
    void benchmark(int batch, int seq_len, int num_layers) {
        // 测试各种优化组合
        float* input, *output;
        allocateBuffers(&input, &output, batch, seq_len);
        
        // Baseline
        metrics.baseline_time = timeKernel([&]() {
            baselineForward(input, output, batch, seq_len, num_layers);
        });
        
        // Graph优化
        metrics.graph_time = timeKernel([&]() {
            graph_optimizer.forward(input, output, batch, seq_len);
        });
        
        // 融合优化
        metrics.fusion_time = timeKernel([&]() {
            fusedForward(input, output, batch, seq_len, num_layers);
        });
        
        // 批处理优化
        metrics.batching_time = timeKernel([&]() {
            batch_optimizer.processBatch();
        });
        
        // 流水线优化
        metrics.pipeline_time = timeKernel([&]() {
            pipeline_optimizer.pipelinedForward(input, output, 
                                               num_layers, batch, seq_len);
        });
        
        // 组合所有优化
        metrics.combined_time = timeKernel([&]() {
            combinedOptimizedForward(input, output, batch, seq_len, num_layers);
        });
        
        metrics.print();
        
        // 详细性能分析
        profileWithNsight();
    }
    
    void combinedOptimizedForward(float* input, float* output,
                                 int batch, int seq_len, int num_layers) {
        // 根据输入特征选择最优策略
        OptimizationStrategy strategy = selectStrategy(batch, seq_len);
        
        if (strategy == SMALL_BATCH) {
            // 小batch：最大化融合
            CUfunction kernel = jit_compiler.specializeForProblem(
                batch, seq_len, hidden_dim, false);
            
            void* args[] = {&input, &output, &weights};
            cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                          block.x, block.y, block.z,
                          shared_mem, 0, args, nullptr);
            
        } else if (strategy == LARGE_BATCH) {
            // 大batch：使用Tensor Core + Graph
            buildTensorCoreGraph(batch, seq_len);
            cudaGraphLaunch(tc_graph_exec, 0);
            
        } else if (strategy == LONG_SEQUENCE) {
            // 长序列：分块处理 + 流水线
            pipeline_optimizer.pipelinedForward(input, output,
                                               num_layers, batch, seq_len);
        }
    }
    
    void profileWithNsight() {
        // 使用CUPTI API收集详细性能数据
        CUpti_ProfilerRange profiler("TransformerInference");
        
        // 内存带宽分析
        float achieved_bandwidth = profiler.getMetric("dram_throughput");
        float peak_bandwidth = 900.0f;  // GB/s for A100
        printf("Memory Bandwidth Utilization: %.1f%%\n", 
               achieved_bandwidth / peak_bandwidth * 100);
        
        // 计算吞吐量分析
        float achieved_tflops = profiler.getMetric("flop_sp_efficiency");
        float peak_tflops = 19.5f;  // TFLOPs for A100
        printf("Compute Utilization: %.1f%%\n", 
               achieved_tflops / peak_tflops * 100);
        
        // 占用率分析
        float occupancy = profiler.getMetric("achieved_occupancy");
        printf("Achieved Occupancy: %.1f%%\n", occupancy * 100);
        
        // 内核效率分析
        float sm_efficiency = profiler.getMetric("sm_efficiency");
        printf("SM Efficiency: %.1f%%\n", sm_efficiency * 100);
    }
};

// 最终优化结果示例：
// Batch=1, Seq=512, Layers=12:
//   Baseline:      45.3 ms
//   Optimized:     3.8 ms (11.9x speedup)
//
// Batch=32, Seq=128, Layers=12:
//   Baseline:      285.6 ms
//   Optimized:     24.2 ms (11.8x speedup)
```

## 20.6 本章小结

本章深入探讨了CUDA Graph和内核融合这两项关键的系统级优化技术，并结合JIT编译和自动调优框架，展示了如何构建高性能的端到端推理系统。

### 核心要点回顾

1. **CUDA Graph技术**
   - Graph通过消除CPU-GPU同步开销，实现了执行流的极致优化
   - 流捕获机制提供了便捷的Graph构建方式，而手动构建提供了最大灵活性
   - Graph更新和条件执行支持动态工作负载
   - 性能提升关键在于最大化并行度和最小化同步点

2. **内核融合策略**
   - 垂直融合将producer-consumer关系的内核合并，减少中间结果存储
   - 水平融合提高硬件利用率，特别适合批处理场景
   - Element-wise操作融合能显著减少内存带宽压力
   - Reduction融合需要特殊的算法设计，如online softmax

3. **自动调优框架**
   - Profile-guided optimization基于实际性能数据进行优化
   - 搜索空间的有效定义和剪枝是快速收敛的关键
   - 贝叶斯优化通过建立性能模型指导搜索
   - 遗传算法适合处理离散的大规模搜索空间

4. **JIT编译优化**
   - NVRTC运行时编译实现了kernel的动态特化
   - 模板元编程结合代码生成提供了灵活性
   - 缓存策略对JIT性能至关重要
   - PTX级优化能够实现极致性能

5. **端到端优化实践**
   - 不同优化技术的组合使用能够产生叠加效应
   - 动态批处理提高了系统的吞吐量
   - 流水线技术有效隐藏了延迟
   - 综合优化可以实现10x以上的性能提升

### 关键公式与度量

1. **Graph执行效率**：
   ```
   Efficiency = (Kernel_Execution_Time) / (Total_Graph_Time)
   Graph_Overhead = Total_Graph_Time - Kernel_Execution_Time
   ```

2. **融合收益评估**：
   ```
   Memory_Reduction = 1 - (Fused_Memory_Access / Original_Memory_Access)
   Speedup = Original_Time / Fused_Time
   ```

3. **自动调优收敛率**：
   ```
   Convergence_Rate = (Best_Performance - Current_Performance) / Iterations
   ```

4. **JIT编译投资回报**：
   ```
   ROI = (Specialized_Kernel_Speedup × Reuse_Count - Compilation_Time) / Compilation_Time
   ```

### 性能优化决策树

```
输入特征分析
    ├─ Batch Size
    │   ├─ 小 (< 4) → 最大化融合 + JIT特化
    │   └─ 大 (≥ 4) → Graph + Tensor Core
    ├─ Sequence Length
    │   ├─ 短 (< 128) → 完全融合
    │   └─ 长 (≥ 128) → 分块 + 流水线
    └─ 计算密度
        ├─ Memory-bound → 融合优先
        └─ Compute-bound → 并行优先
```

## 20.7 练习题

### 基础题

**练习1：Graph基本操作**
构建一个CUDA Graph，包含3个串行的矩阵乘法操作，测量相比独立kernel调用的性能提升。

<details>
<summary>提示</summary>

使用流捕获机制记录操作序列，注意预分配所有需要的缓冲区。比较Graph执行和独立kernel调用的总时间。

</details>

<details>
<summary>答案</summary>

使用cudaStreamBeginCapture开始捕获，依次记录3个GEMM kernel，然后cudaStreamEndCapture结束捕获。实例化Graph后，通过cudaGraphLaunch执行。典型情况下，Graph执行能够减少50-70%的启动开销，特别是对于小矩阵。关键在于消除了kernel间的CPU-GPU同步。

</details>

**练习2：简单内核融合**
将一个element-wise加法操作和ReLU激活函数融合成单个内核，分析内存带宽的节省。

<details>
<summary>提示</summary>

原始版本需要3次内存访问（读A、读B、写C用于加法，读C、写D用于ReLU）。融合版本只需要2次（读A和B，写最终结果）。

</details>

<details>
<summary>答案</summary>

融合内核直接计算`output[i] = max(0.0f, a[i] + b[i])`，减少了50%的内存访问。对于memory-bound的操作，这直接转化为约2x的性能提升。实际加速比取决于缓存命中率和内存带宽利用率。

</details>

**练习3：搜索空间定义**
为一个矩阵转置kernel定义合理的调优搜索空间，包括block size和tile size的候选值。

<details>
<summary>提示</summary>

考虑硬件约束（最大线程数、共享内存大小）和性能因素（bank conflict、占用率）。

</details>

<details>
<summary>答案</summary>

Block size候选：{16×16, 32×8, 8×32}以保持256线程。Tile size候选：{16×16, 32×32}考虑共享内存限制。对于避免bank conflict，tile宽度应该是33而不是32。搜索空间大小：3×2=6种配置，可以快速遍历。

</details>

**练习4：JIT编译场景识别**
列举3个适合使用JIT编译的场景，并说明原因。

<details>
<summary>提示</summary>

考虑哪些情况下运行时信息能够带来显著的优化机会。

</details>

<details>
<summary>答案</summary>

1. 动态形状的矩阵运算：可以将维度作为编译时常量，启用循环展开
2. 稀疏模式已知的运算：根据稀疏结构生成特化代码
3. 用户自定义的激活函数：避免函数指针调用开销
每种场景都能通过特化获得20-50%的性能提升。

</details>

### 挑战题

**练习5：复杂Graph优化**
设计一个包含条件分支的CUDA Graph，实现动态选择不同精度（FP32/FP16）的计算路径。测量不同精度下的性能差异。

<details>
<summary>提示</summary>

使用CUDA 12.3的条件节点功能，或者预构建两个子图，运行时选择执行。需要考虑精度转换的开销。

</details>

<details>
<summary>答案</summary>

创建两个子图，一个用于FP32路径，一个用于FP16路径。使用条件节点根据输入数据的特征（如数值范围）动态选择。FP16路径通常能提供2x的内存带宽优势和Tensor Core加速，但需要处理溢出和精度损失。关键是在Graph构建时就确定好所有可能的执行路径，避免运行时重构。

</details>

**练习6：多层融合策略**
设计一个融合策略，将LayerNorm、GEMM和激活函数三个操作融合，并处理好数值稳定性问题。

<details>
<summary>提示</summary>

LayerNorm需要两次遍历数据（计算统计量和归一化），考虑如何与GEMM的计算模式结合。注意在线算法的使用。

</details>

<details>
<summary>答案</summary>

使用分块策略：每个block负责输出的一部分行。先用一个warp计算LayerNorm统计量（使用Welford算法保证数值稳定），同步后所有线程并行执行归一化和GEMM。关键优化：1)使用共享内存缓存归一化后的数据；2)将GEMM的累加与激活函数融合；3)使用向量化load/store。这种融合能减少75%的全局内存访问。

</details>

**练习7：自适应调优系统**
实现一个自适应的调优系统，能够根据硬件特性和问题规模自动选择最优的kernel配置。系统应该包含离线训练和在线预测两个阶段。

<details>
<summary>提示</summary>

使用机器学习方法建立性能模型。特征包括：问题规模、硬件规格、内存访问模式。可以使用简单的决策树或神经网络。

</details>

<details>
<summary>答案</summary>

离线阶段：收集不同配置下的性能数据，提取特征（矩阵维度、算术强度、硬件SM数等），训练一个轻量级MLP预测最优配置。在线阶段：对新问题提取特征，模型预测top-3配置，快速评估后选择最优。关键是特征工程：包括计算密度、内存访问模式、数据复用度等。这种方法相比暴力搜索能减少90%的调优时间。

</details>

**练习8：端到端优化实战**
给定一个包含Conv-BN-ReLU-Pool的CNN层，设计并实现一个完整的优化方案，包括Graph构建、内核融合和自动调优，目标是达到cuDNN 80%的性能。

<details>
<summary>提示</summary>

分析数据流和复用机会。Conv和BN可以融合，ReLU可以与BN输出阶段融合，Pool可能需要单独处理。使用im2col或implicit GEMM方法。

</details>

<details>
<summary>答案</summary>

优化方案：
1. 使用implicit GEMM实现Conv，避免im2col的内存开销
2. 将BN的统计量预计算并融入Conv的epilogue
3. ReLU直接在写出时应用
4. Pool使用单独的kernel但通过Graph串联，避免同步
5. 使用自动调优找到最优的tile size和block配置
6. 对于常见的层配置，使用JIT生成特化代码

关键实现细节：
- Conv使用1x1、3x3、5x5等专门优化的模板
- BN参数提前fuse成scale和bias
- 使用Tensor Core（如果适用）
- 内存布局优化（NCHW vs NHWC）

通过这些优化，可以达到cuDNN 75-85%的性能，差距主要在于cuDNN使用了更多硬件特定的优化和汇编级调优。

</details>

## 20.8 常见陷阱与错误

### Graph相关陷阱

1. **Graph更新失败**
   ```cpp
   // 错误：假设更新总是成功
   cudaGraphExecKernelNodeSetParams(graphExec, node, &params);
   
   // 正确：检查更新结果
   cudaGraphExecUpdateResult updateResult;
   cudaGraphExecUpdate(graphExec, graph, nullptr, &updateResult);
   if (updateResult != cudaGraphExecUpdateSuccess) {
       // 需要重新实例化
       cudaGraphExecDestroy(graphExec);
       cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
   }
   ```

2. **流捕获泄漏**
   ```cpp
   // 错误：捕获期间使用了默认流
   cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
   cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice); // 使用默认流！
   
   // 正确：所有操作都使用捕获流
   cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
   ```

3. **Graph资源管理**
   ```cpp
   // 错误：没有释放Graph资源
   cudaGraph_t graph;
   cudaGraphCreate(&graph, 0);
   // 使用graph...
   
   // 正确：正确释放
   cudaGraphDestroy(graph);
   cudaGraphExecDestroy(graphExec);
   ```

### 融合相关陷阱

4. **寄存器压力过大**
   ```cpp
   // 错误：过度融合导致寄存器溢出
   __global__ void overFusedKernel() {
       float regs[128];  // 太多寄存器变量
       // 导致寄存器溢出到本地内存
   }
   
   // 正确：平衡融合程度
   __global__ void balancedFusion() {
       float regs[32];  // 适度的寄存器使用
   }
   ```

5. **共享内存bank conflict**
   ```cpp
   // 错误：融合后产生bank conflict
   __shared__ float shared[32][32];  // 32-way bank conflict
   
   // 正确：padding避免conflict
   __shared__ float shared[32][33];  // 避免bank conflict
   ```

### JIT相关陷阱

6. **编译错误处理不当**
   ```cpp
   // 错误：不检查编译结果
   nvrtcCompileProgram(prog, 0, nullptr);
   
   // 正确：检查并获取错误信息
   nvrtcResult result = nvrtcCompileProgram(prog, 0, nullptr);
   if (result != NVRTC_SUCCESS) {
       size_t logSize;
       nvrtcGetProgramLogSize(prog, &logSize);
       char* log = new char[logSize];
       nvrtcGetProgramLog(prog, log);
       printf("Compilation error: %s\n", log);
       delete[] log;
   }
   ```

7. **缓存失效问题**
   ```cpp
   // 错误：不考虑架构差异
   // 在SM_70上编译的代码在SM_80上运行
   
   // 正确：包含架构信息在缓存键中
   std::string cacheKey = sourceHash + "_sm" + std::to_string(smVersion);
   ```

### 调试技巧

1. **使用Nsight Compute分析Graph**
   ```bash
   ncu --target-processes all --graph-profiling=node ./app
   ```

2. **融合效果验证**
   ```cpp
   // 添加验证代码
   void verifyFusion() {
       // 运行未融合版本
       runUnfused(input, expected_output);
       
       // 运行融合版本
       runFused(input, actual_output);
       
       // 比较结果
       float maxError = compareResults(expected_output, actual_output);
       assert(maxError < 1e-5f);
   }
   ```

3. **JIT编译时间监控**
   ```cpp
   class CompilationMonitor {
       std::map<std::string, float> compilationTimes;
       
   public:
       void recordCompilation(const std::string& kernel, float time) {
           compilationTimes[kernel] = time;
           if (time > 1000.0f) {  // 超过1秒
               printf("Warning: %s compilation took %.2f ms\n", 
                      kernel.c_str(), time);
           }
       }
   };
   ```

## 20.9 最佳实践检查清单

### Graph优化检查清单

- [ ] **Graph构建**
  - □ 预分配所有需要的缓冲区
  - □ 使用流捕获vs手动构建的权衡
  - □ 正确处理多流同步
  - □ Graph更新策略明确

- [ ] **性能优化**
  - □ 最小化Graph中的同步点
  - □ 合并小kernel减少节点数
  - □ 使用异步操作
  - □ 考虑Graph分区for大规模图

- [ ] **资源管理**
  - □ 正确释放Graph和GraphExec
  - □ 避免Graph实例化的内存泄漏
  - □ 监控Graph执行时间

### 内核融合检查清单

- [ ] **融合策略**
  - □ 识别producer-consumer关系
  - □ 评估内存带宽节省
  - □ 考虑寄存器压力
  - □ 平衡融合粒度

- [ ] **实现质量**
  - □ 避免bank conflict
  - □ 使用向量化访存
  - □ 正确处理边界条件
  - □ 数值稳定性验证

- [ ] **性能验证**
  - □ 对比融合前后的内存访问
  - □ 测量实际加速比
  - □ 分析占用率变化
  - □ 检查缓存利用率

### JIT优化检查清单

- [ ] **编译策略**
  - □ 识别适合JIT的场景
  - □ 设计合理的模板
  - □ 实现编译缓存
  - □ 处理编译失败

- [ ] **性能考虑**
  - □ 平衡编译时间和执行时间
  - □ 缓存命中率监控
  - □ 内存使用控制
  - □ 多架构支持

- [ ] **代码生成**
  - □ 生成高效的代码
  - □ 利用编译时常量
  - □ 适当的循环展开
  - □ 向量化优化

### 自动调优检查清单

- [ ] **搜索空间**
  - □ 定义完整但精简的搜索空间
  - □ 有效的剪枝策略
  - □ 硬件约束考虑
  - □ 增量式搜索

- [ ] **调优算法**
  - □ 选择合适的优化算法
  - □ 性能模型准确性
  - □ 收敛速度监控
  - □ 过拟合预防

- [ ] **实用性**
  - □ 调优时间预算
  - □ 结果可重现性
  - □ 跨平台移植性
  - □ 调优结果持久化
```
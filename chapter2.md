# 第2章：CUDA编程模型与执行模型

本章深入探讨CUDA的编程模型和执行模型，这是理解GPU并行计算的核心基础。我们将从线程组织结构开始，逐步深入到内核启动、流管理、内存模型以及调试技术。通过本章学习，你将掌握如何高效地组织和管理GPU上的大规模并行计算任务，为后续的性能优化打下坚实基础。

## 2.1 线程层次结构与网格配置

CUDA采用三级线程层次结构来组织大规模并行计算：网格（Grid）、线程块（Block）和线程（Thread）。这种层次化设计既符合GPU硬件架构，又为程序员提供了灵活的并行表达能力。

### 2.1.1 三级层次结构详解

**线程（Thread）** 是CUDA中的最小执行单元。每个线程执行相同的内核代码，但可以通过内置变量访问自己的唯一标识符，从而处理不同的数据。线程拥有自己的寄存器和局部内存，执行时相互独立。

**线程块（Block）** 是一组可以协作的线程集合。同一线程块内的线程可以通过共享内存进行数据交换，并通过同步原语进行协调。线程块的大小受硬件限制，目前最大为1024个线程。线程块可以是一维、二维或三维的，这种多维组织方式便于处理多维数据结构。

**网格（Grid）** 是线程块的集合，代表一次内核启动的所有并行工作。网格也可以是一维、二维或三维的。网格中的线程块相互独立执行，它们之间没有同步机制（除非使用协作组或原子操作）。

```
Grid (3D)
    │
    ├─── Block(0,0,0) ─── Thread(0,0,0), Thread(1,0,0), ...
    │         │
    │         └─── Thread(0,1,0), Thread(1,1,0), ...
    │
    ├─── Block(1,0,0) ─── Thread(0,0,0), Thread(1,0,0), ...
    │         │
    │         └─── Thread(0,1,0), Thread(1,1,0), ...
    └─── ...
```

### 2.1.2 线程索引计算

在内核函数中，每个线程需要计算其全局唯一索引来确定要处理的数据。CUDA提供了内置变量来访问线程和块的索引：

- `threadIdx.x/y/z`：线程在块内的局部索引
- `blockIdx.x/y/z`：块在网格内的索引
- `blockDim.x/y/z`：块的维度
- `gridDim.x/y/z`：网格的维度

对于一维索引计算：
```
全局线程ID = blockIdx.x * blockDim.x + threadIdx.x
```

对于二维索引计算：
```
全局X索引 = blockIdx.x * blockDim.x + threadIdx.x
全局Y索引 = blockIdx.y * blockDim.y + threadIdx.y
线性索引 = 全局Y索引 * 网格宽度 + 全局X索引
```

### 2.1.3 Warp与SIMT执行模型

GPU的实际执行单位是**warp**，每个warp包含32个线程。这32个线程以SIMT（Single Instruction, Multiple Thread）方式执行，即同时执行相同的指令但操作不同的数据。理解warp对于性能优化至关重要：

**Warp调度**：SM上的warp调度器负责选择就绪的warp执行。当某个warp因内存访问或同步而停滞时，调度器会切换到其他warp，从而隐藏延迟。这种零开销的上下文切换是GPU高吞吐量的关键。

**Warp分歧**：当warp内的线程执行不同的代码路径（如if-else分支）时，会发生warp分歧。硬件通过串行执行各个分支来处理分歧，这会降低性能。优化策略包括：
- 重组数据使相邻线程执行相同分支
- 使用无分支的算法（如位操作替代条件判断）
- 利用warp投票函数协调分支决策

### 2.1.4 最优网格配置策略

选择合适的网格和块配置对性能至关重要。需要考虑的因素包括：

**占用率（Occupancy）**：指SM上活跃warp数与最大warp数的比率。高占用率有助于隐藏延迟，但不是唯一决定因素。占用率受以下资源限制：
- 每个SM的最大线程数
- 每个SM的最大块数
- 寄存器使用量
- 共享内存使用量

**块大小选择原则**：
1. 块大小应为32的倍数（warp大小）
2. 通常选择128、256或512个线程
3. 考虑共享内存和寄存器的使用情况
4. 使用occupancy calculator工具辅助决策

**动态网格尺寸计算**：
```cuda
int blockSize;   // 内核的块大小
int minGridSize; // 满足最大占用率的最小网格大小
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
int gridSize = (N + blockSize - 1) / blockSize;  // 向上取整
kernel<<<gridSize, blockSize>>>(args);
```

### 2.1.5 多维网格的应用场景

多维网格特别适合处理多维数据结构：

**二维网格处理图像**：对于M×N的图像，可以使用二维块和二维网格，每个线程处理一个像素。这种映射直观且缓存友好。

**三维网格处理体数据**：在处理医学成像、流体模拟等三维数据时，三维网格提供了自然的映射方式。

**维度选择策略**：
- 数据维度与网格维度匹配可简化索引计算
- 考虑内存访问模式，相邻线程应访问相邻内存
- 某些算法可能需要特殊的线程组织（如矩阵转置）

## 2.2 内核启动与动态并行

内核函数是在GPU上执行的并行代码单元。CUDA提供了灵活的内核启动机制，包括传统的主机端启动和动态并行（设备端启动）。理解这些机制对于构建复杂的GPU应用至关重要。

### 2.2.1 内核函数的声明与定义

CUDA使用特殊的函数类型限定符来区分不同的函数类型：

**`__global__`函数**：内核函数，可从主机调用，在设备上执行。必须返回void，支持模板和重载。

**`__device__`函数**：设备函数，只能从设备调用，在设备上执行。可以有返回值，支持递归（需要特殊编译选项）。

**`__host__`函数**：主机函数，默认类型。可以与`__device__`组合使用，生成主机和设备两个版本。

内核函数的参数传递规则：
- 参数通过常量内存传递（限制4KB）
- 大型结构体应通过指针传递
- 不支持可变参数列表
- 不支持静态变量（除非使用`__shared__`）

### 2.2.2 启动配置参数详解

内核启动使用特殊的执行配置语法：
```cuda
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args);
```

**gridDim**：网格维度，类型为dim3。指定网格中块的数量。
- 最大维度：X(2^31-1), Y(65535), Z(65535)
- 可以使用整数自动转换为(N,1,1)

**blockDim**：块维度，类型为dim3。指定每个块中线程的数量。
- 最大线程数：1024（X×Y×Z ≤ 1024）
- 建议为warp大小（32）的倍数

**sharedMem**：动态共享内存大小（字节）。可选参数，默认为0。
- 与静态共享内存共享48KB/96KB的空间
- 动态分配允许运行时确定大小

**stream**：执行流。可选参数，默认为0（默认流）。
- 用于异步执行和并发管理
- 不同流中的操作可以并发执行

### 2.2.3 动态并行编程模型

动态并行允许GPU内核直接启动其他内核，无需CPU介入。这对递归算法和自适应算法特别有用。

**启用条件**：
- 计算能力3.5及以上
- 编译时添加`-rdc=true -lcudadevrt`
- 链接设备运行时库

**设备端内核启动**：
```cuda
__global__ void parent_kernel() {
    if (threadIdx.x == 0) {
        child_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();  // 设备端同步
    }
}
```

**内存模型**：
- 父内核的全局内存对子内核可见
- 子内核的局部和共享内存独立
- 父子内核间通过全局内存通信

**同步机制**：
- `cudaDeviceSynchronize()`：等待所有子内核完成
- 父内核结束时隐式同步所有子内核
- 注意避免死锁（父等子，子等父）

### 2.2.4 嵌套深度与资源管理

动态并行的嵌套深度和资源使用需要仔细管理：

**嵌套深度限制**：
- 默认最大深度为24层
- 可通过`cudaLimitDevRuntimeSyncDepth`调整
- 深度过大会导致资源耗尽

**资源池管理**：
- 设备端启动使用独立的资源池
- 通过`cudaLimitDevRuntimePendingLaunchCount`控制
- 默认限制2048个待处理的启动

**性能考量**：
- 设备端启动有额外开销（约10μs）
- 适合粗粒度并行（每个子内核做大量工作）
- 细粒度并行应使用协作组

### 2.2.5 递归算法的GPU实现

动态并行使得递归算法在GPU上成为可能。典型应用包括：

**快速排序**：
```cuda
__global__ void quicksort(int* data, int left, int right) {
    if (left < right) {
        int pivot = partition(data, left, right);
        cudaStream_t s1, s2;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        
        quicksort<<<1, 1, 0, s1>>>(data, left, pivot - 1);
        quicksort<<<1, 1, 0, s2>>>(data, pivot + 1, right);
        
        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
    }
}
```

**自适应网格细化**：
- 根据误差估计动态细化网格
- 只在需要的区域增加计算密度
- 适用于自适应有限元、光线追踪等

**树遍历算法**：
- 并行遍历不规则树结构
- 动态负载均衡
- 避免CPU-GPU频繁同步

### 2.2.6 内核启动的性能优化

**启动开销优化**：
- 批量处理小任务，减少启动次数
- 使用持久化内核处理流式数据
- 利用CUDA Graph减少启动开销

**网格规模优化**：
- 确保足够的并行度（至少数千个线程）
- 避免尾部效应（部分块未充分利用）
- 使用网格跨步循环处理大规模数据

**编译优化选项**：
- `-use_fast_math`：使用快速数学函数
- `-maxrregcount`：限制寄存器使用
- `--ptxas-options=-v`：显示资源使用信息

## 2.3 流与事件机制

CUDA流（Stream）是GPU上的操作队列，允许并发执行多个任务。事件（Event）则用于同步和性能测量。掌握流和事件机制是实现高效GPU程序的关键。

### 2.3.1 CUDA流的概念与创建

**流的本质**：流是一个有序的操作序列，同一流中的操作按顺序执行，不同流中的操作可以并发执行。这种机制允许：
- 计算与数据传输重叠
- 多个内核并发执行
- 细粒度的执行控制

**流的类型**：
- **默认流（NULL流）**：隐式同步，与其他所有流同步
- **非默认流**：显式创建，可以并发执行
- **优先级流**：支持不同优先级的任务调度

**流的创建与销毁**：
```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);                      // 创建默认优先级流
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);  // 非阻塞流
cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);  // 优先级流
cudaStreamDestroy(stream);                      // 销毁流
```

### 2.3.2 异步操作与并发执行

CUDA中大部分操作都有异步版本，允许CPU在GPU执行时继续工作：

**异步内存操作**：
```cuda
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
cudaMemsetAsync(ptr, value, size, stream);
cudaMemPrefetchAsync(ptr, size, device, stream);  // 统一内存预取
```

**异步内核执行**：
```cuda
kernel<<<grid, block, 0, stream>>>(args);
```

**并发模式分析**：
```
时间线示例（H2D=主机到设备传输，K=内核，D2H=设备到主机传输）：

默认流：  H2D -----> K -----> D2H ----->
         |<------- 总时间 ------->|

双流并发： Stream1: H2D1 --> K1 --> D2H1 -->
         Stream2:      H2D2 --> K2 --> D2H2 -->
         |<----- 减少的总时间 ----->|
```

### 2.3.3 流同步机制

控制流之间的依赖关系和同步点：

**流级同步**：
```cuda
cudaStreamSynchronize(stream);      // 等待特定流完成
cudaDeviceSynchronize();            // 等待所有流完成
cudaStreamQuery(stream);            // 非阻塞查询流状态
```

**流间依赖**：
```cuda
cudaStreamWaitEvent(stream, event, 0);  // 流等待事件
cudaEvent_t event;
cudaEventCreate(&event);
cudaEventRecord(event, stream1);        // 在stream1中记录事件
cudaStreamWaitEvent(stream2, event, 0); // stream2等待event
```

**回调函数**：
```cuda
void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void* data) {
    // 在流中所有操作完成后执行
}
cudaStreamAddCallback(stream, callback, userData, 0);
```

### 2.3.4 事件的创建与使用

事件是流中的标记点，用于同步和计时：

**事件创建与记录**：
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreateWithFlags(&stop, cudaEventDisableTiming);  // 禁用计时
cudaEventRecord(start, stream);
// ... 执行操作 ...
cudaEventRecord(stop, stream);
```

**事件同步**：
```cuda
cudaEventSynchronize(event);        // 等待事件完成
cudaEventQuery(event);              // 非阻塞查询
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);  // 计算时间差
```

**事件的硬件实现**：
- 事件在GPU时间线上插入标记
- 轻量级（几乎无开销）
- 精度达到纳秒级

### 2.3.5 多流优化策略

**流的数量选择**：
- Hyper-Q支持32个硬件队列
- 实践中4-8个流通常足够
- 过多流增加管理开销

**深度优先vs广度优先**：
```cuda
// 深度优先：每个流完成所有操作
for (int i = 0; i < nStreams; i++) {
    cudaMemcpyAsync(d_a[i], h_a[i], size, cudaMemcpyHostToDevice, stream[i]);
    kernel<<<grid, block, 0, stream[i]>>>(d_a[i]);
    cudaMemcpyAsync(h_a[i], d_a[i], size, cudaMemcpyDeviceToHost, stream[i]);
}

// 广度优先：按操作类型批处理（更好的并发）
for (int i = 0; i < nStreams; i++)
    cudaMemcpyAsync(d_a[i], h_a[i], size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < nStreams; i++)
    kernel<<<grid, block, 0, stream[i]>>>(d_a[i]);
for (int i = 0; i < nStreams; i++)
    cudaMemcpyAsync(h_a[i], d_a[i], size, cudaMemcpyDeviceToHost, stream[i]);
```

**流水线并行模式**：
- 将大任务分割成小块
- 使用循环缓冲区
- 重叠计算和传输

### 2.3.6 CUDA Graph优化

CUDA Graph将一系列操作捕获为图，可以高效重复执行：

**Graph创建与执行**：
```cuda
cudaGraph_t graph;
cudaGraphExec_t instance;
cudaStream_t stream;

// 捕获操作序列
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// ... 记录操作 ...
cudaStreamEndCapture(stream, &graph);

// 实例化并执行
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaGraphLaunch(instance, stream);
```

**Graph优势**：
- 减少CPU启动开销
- 优化GPU调度
- 适合重复执行的工作负载

**Graph更新**：
- 支持参数更新而不重建图
- 动态修改节点
- 条件执行分支

## 2.4 统一内存与虚拟内存管理

统一内存（Unified Memory）是CUDA 6.0引入的革命性特性，它提供了单一的内存地址空间，使CPU和GPU都可以访问相同的数据。这大大简化了内存管理，同时通过智能的页面迁移机制保持高性能。

### 2.4.1 统一内存的工作原理

**统一虚拟地址空间（UVA）**：统一内存建立在UVA基础上，为CPU和GPU提供相同的虚拟地址空间。这意味着指针在两端都有效，无需显式的内存拷贝。

**页面迁移机制**：
- 按需迁移：当处理器访问不在本地的页面时触发页面错误，系统自动迁移页面
- 预取机制：程序可以提示系统预先迁移即将使用的数据
- 并发访问：Pascal架构后支持CPU和GPU同时访问，通过原子操作保证一致性

**内存分配与释放**：
```cuda
// 分配统一内存
void* ptr;
cudaMallocManaged(&ptr, size);
cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);  // 全局可见
cudaMallocManaged(&ptr, size, cudaMemAttachHost);    // 优先CPU

// 释放统一内存
cudaFree(ptr);
```

### 2.4.2 页面错误与迁移策略

**页面错误处理流程**：
1. 处理器访问非本地页面
2. 触发页面错误中断
3. 驱动程序暂停执行
4. 迁移页面到请求方
5. 更新页表
6. 恢复执行

**迁移粒度**：
- 默认页面大小：64KB（可调整）
- 大页面支持：2MB大页面减少TLB压力
- 细粒度控制：通过内存提示API控制迁移行为

**迁移优化策略**：
```cuda
// 预取到GPU
cudaMemPrefetchAsync(ptr, size, deviceId, stream);

// 建议数据位置
cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device);     // 只读优化
cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device); // 首选位置
cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);      // 访问提示
```

### 2.4.3 内存超额订阅

统一内存支持超额订阅，即分配超过GPU物理内存的数据：

**工作机制**：
- 使用系统内存作为后备
- 按需换入换出页面
- LRU算法管理页面置换

**性能影响与优化**：
- 频繁换页导致性能下降
- 工作集优化：确保活跃数据适合GPU内存
- 访问模式优化：局部性原理
- 分批处理：将大数据集分割处理

**驱逐策略控制**：
```cuda
// 设置驱逐优先级
cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
// 钉住内存防止驱逐
cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, deviceId);
```

### 2.4.4 系统级内存管理

**内存池管理**：
```cuda
cudaMemPool_t mempool;
cudaMemPoolProps props = {};
props.allocType = cudaMemAllocationTypePinned;
props.handleTypes = cudaMemHandleTypePosixFileDescriptor;
props.location.type = cudaMemLocationTypeDevice;
props.location.id = device;

cudaMemPoolCreate(&mempool, &props);
cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
```

**虚拟内存API**：
- 细粒度的地址空间控制
- 物理内存的显式管理
- 支持稀疏数据结构

```cuda
// 保留虚拟地址空间
CUdeviceptr ptr;
cuMemAddressReserve(&ptr, size, alignment, 0, 0);

// 创建物理内存
CUmemGenericAllocationHandle handle;
cuMemCreate(&handle, size, &prop, 0);

// 映射物理内存到虚拟地址
cuMemMap(ptr, size, 0, handle, 0);
cuMemSetAccess(ptr, size, &accessDesc, 1);
```

### 2.4.5 跨GPU内存访问

**NVLink/NVSwitch互连**：
- 高带宽GPU间通信（300-600 GB/s）
- 统一内存自动利用NVLink
- 支持原子操作和一致性

**多GPU统一内存模式**：
```cuda
// 多GPU访问同一内存
for (int i = 0; i < nGPUs; i++) {
    cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, i);
}

// GPU间直接访问
__global__ void kernel(float* remote_data) {
    // 可以直接访问其他GPU的数据
    float value = remote_data[threadIdx.x];
}
```

**NUMA感知优化**：
- CPU NUMA节点亲和性
- GPU拓扑感知放置
- 优化数据布局减少跨节点访问

### 2.4.6 性能分析与调优

**性能指标监控**：
```cuda
// 查询统一内存属性
cudaMemRangeAttribute attribute;
cudaMemRangeGetAttribute(&data, &size, cudaMemRangeAttributeReadMostly, ptr, size);

// 页面迁移统计
size_t resident, mapped;
cudaMemGetInfo(&free, &total);
```

**Nsight Systems分析**：
- 页面错误时间线
- 迁移带宽利用率
- 内存驻留分析
- 超额订阅影响

**优化检查清单**：
1. 预取关键数据路径
2. 设置合适的内存提示
3. 避免频繁的小粒度迁移
4. 考虑内存池减少分配开销
5. 监控页面错误率

## 2.5 错误处理与调试技术

CUDA程序的调试比CPU程序更具挑战性，因为涉及大规模并行执行、异步操作和硬件限制。建立系统的错误处理和调试方法是开发高质量GPU程序的基础。

### 2.5.1 CUDA错误类型与检测

**错误类型分类**：

**同步错误**：立即返回的错误
- 无效参数（如空指针、负数大小）
- 资源不足（内存分配失败）
- 设备不支持（功能或计算能力）

**异步错误**：延迟检测的错误
- 内核执行错误（非法内存访问、断言失败）
- 设备端异常（栈溢出、非法指令）
- 硬件错误（ECC错误、温度过高）

**错误检测机制**：
```cuda
// 基本错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// 内核启动后的错误检查
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());        // 检查启动错误
CUDA_CHECK(cudaDeviceSynchronize());   // 检查执行错误
```

### 2.5.2 设备端断言与printf

**设备端断言**：
```cuda
__global__ void kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(idx < n);  // 设备端断言
    assert(data != nullptr);
    
    // 条件断言
    if (idx == 0) {
        assert(blockDim.x <= 1024);
    }
}
```

**设备端printf**：
```cuda
__global__ void debug_kernel(float* data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Block %d: data[0] = %f\n", blockIdx.x, data[0]);
    }
    
    // 条件打印避免输出爆炸
    if (data[threadIdx.x] < 0) {
        printf("Warning: negative value at thread %d\n", threadIdx.x);
    }
}
```

**printf缓冲区管理**：
```cuda
// 设置printf缓冲区大小（默认1MB）
cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024);

// 刷新printf缓冲区
cudaDeviceSynchronize();
```

### 2.5.3 cuda-gdb调试器使用

**基本调试流程**：
```bash
# 编译时添加调试信息
nvcc -g -G program.cu -o program

# 启动调试器
cuda-gdb ./program

# 常用命令
(cuda-gdb) break kernel_name        # 设置断点
(cuda-gdb) run                      # 运行程序
(cuda-gdb) cuda kernel block thread  # 切换焦点
(cuda-gdb) info cuda threads        # 显示线程信息
(cuda-gdb) print variable           # 打印变量
(cuda-gdb) cuda block (1,0,0) thread (32,0,0)  # 切换到特定线程
```

**条件断点**：
```gdb
# 在特定线程设置断点
break kernel if threadIdx.x == 0 && blockIdx.x == 10

# 数据条件断点
break 123 if array[idx] < 0

# 监视点
watch shared_data[5]
```

**内存检查**：
```gdb
# 检查全局内存
print @global &array[0]@100  # 打印100个元素

# 检查共享内存
print @shared &sdata[0]@32

# 检查寄存器
info registers
```

### 2.5.4 cuda-memcheck内存检查

**检查类型**：
```bash
# 内存访问错误检查
cuda-memcheck ./program

# 竞态条件检测
cuda-memcheck --tool racecheck ./program

# 同步检查
cuda-memcheck --tool synccheck ./program

# 初始化检查
cuda-memcheck --tool initcheck ./program

# 内存泄漏检查
cuda-memcheck --leak-check full ./program
```

**常见内存错误**：
- 越界访问（全局、共享、局部内存）
- 未对齐访问
- 非法地址访问
- 栈溢出
- 设备堆溢出

**错误报告解读**：
```
========= Invalid __global__ read of size 4
=========     at 0x00000098 in kernel(int*, int)
=========     by thread (5,0,0) in block (10,0,0)
=========     Address 0x7fff900 is out of bounds
```

### 2.5.5 Compute Sanitizer高级分析

**启用Compute Sanitizer**：
```bash
# 基本使用
compute-sanitizer ./program

# 详细报告
compute-sanitizer --print-level info ./program

# 保存报告
compute-sanitizer --save report.cs ./program

# 检查特定错误类型
compute-sanitizer --tool memcheck --check-device-heap yes ./program
```

**API调用追踪**：
```bash
compute-sanitizer --tool=trace ./program
```

**性能影响分析**：
- memcheck：10-50x减速
- racecheck：20-200x减速
- synccheck：2-5x减速
- 建议：分阶段使用不同工具

### 2.5.6 错误恢复与容错设计

**错误恢复策略**：
```cuda
class CudaContext {
    cudaStream_t stream;
    bool error_state = false;
    
public:
    void execute_kernel() {
        if (error_state) {
            reset_device();
        }
        
        kernel<<<grid, block, 0, stream>>>(args);
        
        cudaError_t error = cudaStreamQuery(stream);
        if (error != cudaSuccess && error != cudaErrorNotReady) {
            error_state = true;
            handle_error(error);
        }
    }
    
    void reset_device() {
        cudaDeviceReset();
        // 重新初始化资源
        error_state = false;
    }
};
```

**检查点机制**：
```cuda
// 定期保存状态
void checkpoint(State* state, int iteration) {
    if (iteration % checkpoint_interval == 0) {
        cudaMemcpy(host_backup, device_state, size, cudaMemcpyDeviceToHost);
        save_to_disk(host_backup);
    }
}

// 错误恢复
void recover_from_error() {
    load_from_disk(host_backup);
    cudaMemcpy(device_state, host_backup, size, cudaMemcpyHostToDevice);
    resume_computation();
}
```

**软错误处理（ECC）**：
```cuda
// 查询ECC状态
int ecc_enabled;
cudaDeviceGetAttribute(&ecc_enabled, cudaDevAttrEccEnabled, device);

// 处理ECC错误
cudaError_t error = cudaGetLastError();
if (error == cudaErrorECCUncorrectable) {
    // 不可纠正错误，需要重新计算
    retry_computation();
}
```

## 本章小结

本章深入探讨了CUDA编程模型和执行模型的核心概念：

**关键概念总结**：
1. **线程层次结构**：Grid-Block-Thread三级组织，warp为实际执行单位
2. **内核启动机制**：静态和动态并行，执行配置参数优化
3. **流与并发**：异步执行、多流并发、事件同步、CUDA Graph
4. **统一内存**：简化内存管理、页面迁移、超额订阅、跨GPU访问
5. **错误处理**：系统化的错误检测、调试工具链、容错设计

**性能优化要点**：
- 网格配置：平衡占用率与资源使用
- 流并发：重叠计算与传输，广度优先调度
- 统一内存：预取关键数据，设置访问提示
- 调试效率：分层调试策略，自动化错误检查

**最佳实践**：
- 使用occupancy calculator优化启动配置
- 实现全面的错误检查机制
- 利用Nsight工具链进行性能分析
- 设计容错和恢复机制

## 练习题

### 基础题

**练习2.1**：编写一个程序，测试不同块大小（32、64、128、256、512、1024）对简单向量加法内核的性能影响。记录每种配置的执行时间和占用率。
<details>
<summary>提示</summary>
使用cudaOccupancyMaxActiveBlocksPerMultiprocessor获取占用率信息，使用事件计时测量内核执行时间。
</details>

**练习2.2**：实现一个使用3个流的矩阵乘法程序，将输入矩阵分块，使数据传输和计算重叠。比较与单流版本的性能差异。
<details>
<summary>提示</summary>
将矩阵按行分成3部分，每个流处理一部分。使用广度优先的启动顺序以获得最佳并发。
</details>

**练习2.3**：使用统一内存实现一个简单的图像处理程序（如高斯模糊），通过cudaMemPrefetchAsync和cudaMemAdvise优化性能。
<details>
<summary>提示</summary>
预取输入图像到GPU，设置输出图像的首选位置为GPU，处理完成后预取结果到CPU。
</details>

**练习2.4**：编写一个包含完整错误处理的CUDA程序框架，包括同步和异步错误检查、设备端断言和恢复机制。
<details>
<summary>提示</summary>
创建错误检查宏，在每个CUDA API调用后使用，实现错误回调函数处理异步错误。
</details>

### 挑战题

**练习2.5**：实现一个自适应的网格配置系统，根据问题规模和GPU能力自动选择最优的网格和块大小。系统应考虑寄存器使用、共享内存需求和SM资源限制。
<details>
<summary>提示</summary>
使用cudaOccupancyMaxPotentialBlockSize作为起点，然后基于实际资源使用进行微调。考虑创建一个配置缓存以避免重复计算。
</details>

**练习2.6**：设计并实现一个使用动态并行的自适应四叉树构建算法，用于点云空间索引。根据点的密度动态决定是否继续细分。
<details>
<summary>提示</summary>
父内核检查点数，如果超过阈值则启动4个子内核处理子区域。注意控制递归深度和资源使用。
</details>

**练习2.7**：开发一个基于CUDA Graph的深度学习推理引擎，支持动态批大小和条件执行路径。实现图的动态更新而不需要完全重建。
<details>
<summary>提示</summary>
使用图捕获API记录不同批大小的执行路径，通过图更新API修改参数。考虑使用子图处理条件分支。
</details>

**练习2.8**：创建一个内存压力测试工具，测量统一内存在超额订阅情况下的性能特征。工具应能识别最优的工作集大小和页面迁移模式。
<details>
<summary>提示</summary>
逐步增加数据集大小，监控页面错误率和带宽利用率。使用不同的访问模式（顺序、随机、跨步）测试迁移行为。
</details>

## 常见陷阱与错误 (Gotchas)

1. **默认流的隐式同步**：默认流（0流）与所有其他流同步，可能破坏并发性。解决方案：使用非默认流或编译时添加`--default-stream per-thread`。

2. **整数溢出的索引计算**：`blockIdx.x * blockDim.x + threadIdx.x`在大规模问题时可能溢出。使用`size_t`或仔细检查范围。

3. **未检查的内核启动错误**：内核启动是异步的，错误可能延迟报告。始终在内核后调用`cudaGetLastError()`。

4. **统一内存的隐式同步**：在CPU访问统一内存前，GPU操作会隐式同步，可能导致意外的性能下降。

5. **流销毁时的隐式同步**：`cudaStreamDestroy()`会等待流中所有操作完成，可能导致阻塞。

6. **动态并行的资源耗尽**：嵌套启动消耗设备端启动池，深度递归可能导致失败。监控并限制嵌套深度。

7. **printf缓冲区溢出**：设备端printf输出过多会丢失信息。增加缓冲区大小或使用条件打印。

8. **调试版本的性能退化**：`-G`选项禁用优化，性能可能下降100倍。仅在必要时使用调试编译。

## 最佳实践检查清单

### 网格配置
- [ ] 块大小是32的倍数（warp大小）
- [ ] 使用occupancy calculator辅助决策
- [ ] 考虑寄存器和共享内存压力
- [ ] 避免过小的网格（无法充分利用GPU）
- [ ] 处理尾部情况（网格大小不整除问题规模）

### 流管理
- [ ] 使用非默认流实现并发
- [ ] 采用广度优先的操作提交顺序
- [ ] 合理设置流的数量（通常4-8个）
- [ ] 正确同步流间依赖关系
- [ ] 考虑使用CUDA Graph减少启动开销

### 内存管理
- [ ] 选择合适的内存类型（全局、统一、页锁定）
- [ ] 预取关键数据路径
- [ ] 设置统一内存访问提示
- [ ] 监控页面迁移开销
- [ ] 实现内存池减少分配开销

### 错误处理
- [ ] 检查所有CUDA API返回值
- [ ] 内核启动后检查错误
- [ ] 实现错误恢复机制
- [ ] 使用断言验证关键假设
- [ ] 定期运行内存检查工具

### 调试策略
- [ ] 保留发布和调试两个构建配置
- [ ] 使用分层调试方法（printf→断言→调试器）
- [ ] 自动化测试和错误检查
- [ ] 记录和分析性能指标
- [ ] 建立基准测试套件

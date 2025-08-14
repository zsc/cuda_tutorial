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

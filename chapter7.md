# 第7章：原子操作与同步原语

本章深入探讨CUDA中的原子操作和同步机制。我们将从硬件层面理解原子操作的实现原理，掌握各种同步原语的使用场景，并通过实现高并发哈希表来综合运用这些技术。对于自动驾驶和具身智能应用中的并发数据结构设计，这些知识至关重要。

## 7.1 原子操作的硬件实现

### 7.1.1 原子操作的本质

原子操作保证了在多线程环境下对共享内存位置的读-改-写操作的原子性。在GPU上，当多个线程同时访问同一内存地址时，原子操作确保每个操作都完整执行，不会被其他线程打断。

硬件层面，原子操作通过以下机制实现：

1. **L2缓存原子单元**：从Fermi架构开始，NVIDIA GPU在L2缓存中集成了专门的原子操作单元。这些单元能够直接在缓存行上执行原子操作，避免了数据在内存层次结构中的移动。

2. **内存控制器原子单元**：对于全局内存的原子操作，内存控制器包含专门的原子操作逻辑，可以在DRAM接口处直接执行原子操作。

3. **共享内存原子操作**：在SM内部，共享内存的原子操作通过bank锁定机制实现。当一个warp中的线程对同一bank执行原子操作时，硬件会串行化这些访问。

### 7.1.2 原子操作的性能特征

原子操作的延迟特征：
```
操作类型         延迟(cycles)   吞吐量
----------------------------------------
全局内存原子      200-600       低(串行化)
共享内存原子      20-40         中等
L2缓存原子       100-200       中等
寄存器原子       不支持         -
```

关键性能因素：

1. **地址冲突**：多个线程访问同一地址会导致串行化
2. **内存位置**：L2缓存中的原子操作比全局内存快3-5倍
3. **操作类型**：简单操作(add)比复杂操作(CAS)快
4. **访问模式**：分散的访问模式优于集中访问

### 7.1.3 原子操作的内存序

CUDA支持多种内存序模型：

```cuda
// 默认原子操作 - 宽松内存序
atomicAdd(&counter, 1);

// 使用内存序参数 (CUDA 11.0+)
atomicAdd_system(&counter, 1);  // 系统范围一致性
atomicAdd_block(&counter, 1);   // 块范围一致性
```

内存序层次：
```
       系统范围 (system)
           ↑
       设备范围 (device)  
           ↑
        块范围 (block)
           ↑
       warp范围 (默认)
```

### 7.1.4 原子操作优化策略

1. **批量化原子操作**
```cuda
// 低效：每个线程执行原子操作
atomicAdd(&global_counter, 1);

// 高效：先在warp内归约
int mask = __activemask();
int leader = __ffs(mask) - 1;
int warp_sum = __popc(mask);
if (lane_id() == leader) {
    atomicAdd(&global_counter, warp_sum);
}
```

2. **使用warp聚合函数**
```cuda
// CUDA 11.0+ 提供的warp级原子聚合
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

auto tile = cg::tiled_partition<32>(cg::this_thread_block());
if (tile.thread_rank() == 0) {
    atomicAdd(&counter, tile.ballot(predicate));
}
```

3. **原子操作合并**
```cuda
// 使用atomicCAS实现复杂原子操作
__device__ float atomicMax(float* addr, float value) {
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        float old_value = __uint_as_float(assumed);
        float new_value = fmaxf(old_value, value);
        old = atomicCAS(addr_as_uint, assumed, 
                       __float_as_uint(new_value));
    } while (assumed != old);
    
    return __uint_as_float(old);
}
```

## 7.2 自定义原子操作

### 7.2.1 基于CAS的自定义原子操作

Compare-And-Swap (CAS) 是构建自定义原子操作的基础：

```cuda
// 原子乘法实现
__device__ int atomicMul(int* addr, int val) {
    int old = *addr;
    int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, assumed * val);
    } while (assumed != old);
    
    return old;
}

// 原子除法实现（需要处理除零）
__device__ float atomicDiv(float* addr, float divisor) {
    if (divisor == 0.0f) return 0.0f;
    
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        float old_value = __uint_as_float(assumed);
        float new_value = old_value / divisor;
        old = atomicCAS(addr_as_uint, assumed,
                       __float_as_uint(new_value));
    } while (assumed != old);
    
    return __uint_as_float(old);
}
```

### 7.2.2 复杂数据结构的原子操作

对于复杂数据类型，可以使用128位原子操作：

```cuda
// 128位原子CAS (仅compute capability 7.0+)
struct Complex {
    float real;
    float imag;
    float magnitude;
    float phase;
};

__device__ void atomicUpdateComplex(Complex* addr, Complex new_val) {
    // 将128位数据作为两个64位整数处理
    unsigned long long* addr_as_ull = (unsigned long long*)addr;
    unsigned long long old_low = addr_as_ull[0];
    unsigned long long old_high = addr_as_ull[1];
    unsigned long long new_low, new_high;
    
    Complex old_complex, assumed_complex;
    
    do {
        // 重构旧值
        memcpy(&assumed_complex, &old_low, sizeof(Complex));
        
        // 计算新值
        Complex updated = computeUpdate(assumed_complex, new_val);
        memcpy(&new_low, &updated, sizeof(unsigned long long));
        memcpy(&new_high, ((char*)&updated) + 8, sizeof(unsigned long long));
        
        // 原子更新
        old_low = atomicCAS(&addr_as_ull[0], old_low, new_low);
        if (old_low == assumed_complex.low) {
            old_high = atomicCAS(&addr_as_ull[1], old_high, new_high);
        }
    } while (old_low != assumed_complex.low || old_high != assumed_complex.high);
}
```

### 7.2.3 无锁数据结构原语

实现无锁栈：

```cuda
struct Node {
    int data;
    Node* next;
};

struct LockFreeStack {
    Node* top;
    
    __device__ void push(Node* new_node) {
        Node* old_top;
        do {
            old_top = top;
            new_node->next = old_top;
        } while (atomicCAS((unsigned long long*)&top, 
                          (unsigned long long)old_top,
                          (unsigned long long)new_node) != 
                 (unsigned long long)old_top);
    }
    
    __device__ Node* pop() {
        Node* old_top;
        Node* new_top;
        do {
            old_top = top;
            if (old_top == nullptr) return nullptr;
            new_top = old_top->next;
        } while (atomicCAS((unsigned long long*)&top,
                          (unsigned long long)old_top,
                          (unsigned long long)new_top) !=
                 (unsigned long long)old_top);
        return old_top;
    }
};
```

## 7.3 内存栅栏与一致性模型

### 7.3.1 CUDA内存一致性模型

CUDA采用弱内存一致性模型，需要显式的同步来保证内存操作的顺序：

```
线程内顺序：程序顺序
线程间顺序：需要同步原语
  ↓
内存操作重排序规则：
1. Load-Load: 可重排序
2. Load-Store: 可重排序  
3. Store-Load: 可重排序
4. Store-Store: 可重排序
  ↓
需要栅栏指令来防止重排序
```

### 7.3.2 栅栏指令层次

```cuda
// 线程栅栏 - 保证单个线程内的内存操作顺序
__threadfence();        // 设备范围
__threadfence_block();  // 块范围
__threadfence_system(); // 系统范围

// 示例：生产者-消费者模式
__device__ void producer(int* data, int* flag) {
    data[threadIdx.x] = compute_value();
    __threadfence();  // 确保data写入在flag之前完成
    if (threadIdx.x == 0) {
        *flag = 1;  // 通知消费者数据已准备好
    }
}

__device__ void consumer(int* data, int* flag) {
    while (*flag == 0);  // 等待标志
    __threadfence();     // 确保读取flag后再读取data
    int value = data[threadIdx.x];
    process(value);
}
```

### 7.3.3 内存栅栏的性能影响

不同栅栏的开销：
```
栅栏类型                 延迟(cycles)   影响范围
------------------------------------------------
__threadfence_block()    ~30          块内所有线程
__threadfence()          ~200         设备所有线程
__threadfence_system()   ~500         系统所有设备
```

优化策略：
1. 尽量使用最小范围的栅栏
2. 批量操作后使用单个栅栏
3. 使用协作组的同步原语替代

### 7.3.4 发布-获取语义

CUDA 11.0+支持C++11风格的内存序：

```cuda
// 发布语义 - 之前的所有写操作对其他线程可见
__device__ void release_store(int* ptr, int value) {
    __threadfence();
    atomicExch(ptr, value);
}

// 获取语义 - 之后的所有读操作看到最新值
__device__ int acquire_load(int* ptr) {
    int value = atomicAdd(ptr, 0);  // 原子读
    __threadfence();
    return value;
}

// 顺序一致性
__device__ void seq_cst_operation(int* ptr, int value) {
    __threadfence();
    atomicAdd(ptr, value);
    __threadfence();
}
```

## 7.4 自旋锁与无锁算法

### 7.4.1 高效自旋锁实现

基础自旋锁：
```cuda
struct SpinLock {
    int lock;
    
    __device__ void acquire() {
        while (atomicCAS(&lock, 0, 1) != 0) {
            // 自旋等待
        }
    }
    
    __device__ void release() {
        atomicExch(&lock, 0);
    }
};
```

优化的自旋锁（减少原子操作）：
```cuda
struct OptimizedSpinLock {
    volatile int lock;
    
    __device__ void acquire() {
        while (true) {
            // 先用普通读检查
            if (lock == 0) {
                // 尝试获取锁
                if (atomicCAS((int*)&lock, 0, 1) == 0) {
                    break;
                }
            }
            // 指数退避
            __nanosleep(1 << (threadIdx.x & 7));
        }
    }
    
    __device__ void release() {
        __threadfence();  // 确保之前的操作完成
        lock = 0;         // 普通写即可
    }
};

### 7.4.2 票据锁（Ticket Lock）

票据锁提供公平性保证：

```cuda
struct TicketLock {
    unsigned int ticket;
    unsigned int serving;
    
    __device__ void acquire() {
        unsigned int my_ticket = atomicAdd(&ticket, 1);
        while (serving != my_ticket) {
            // 自旋等待自己的票号
            __threadfence_block();
        }
    }
    
    __device__ void release() {
        __threadfence();
        atomicAdd(&serving, 1);
    }
};
```

### 7.4.3 MCS锁（可扩展性更好）

```cuda
struct MCSNode {
    volatile int locked;
    MCSNode* volatile next;
};

struct MCSLock {
    MCSNode* volatile tail;
    
    __device__ void acquire(MCSNode* my_node) {
        my_node->locked = 1;
        my_node->next = nullptr;
        
        MCSNode* predecessor = (MCSNode*)atomicExch(
            (unsigned long long*)&tail, 
            (unsigned long long)my_node
        );
        
        if (predecessor != nullptr) {
            predecessor->next = my_node;
            while (my_node->locked) {
                // 自旋在自己的节点上
            }
        }
    }
    
    __device__ void release(MCSNode* my_node) {
        if (my_node->next == nullptr) {
            if (atomicCAS((unsigned long long*)&tail,
                         (unsigned long long)my_node,
                         0) == (unsigned long long)my_node) {
                return;
            }
            while (my_node->next == nullptr);
        }
        my_node->next->locked = 0;
    }
};
```

### 7.4.4 无锁算法设计原则

1. **ABA问题及解决方案**

```cuda
// 使用版本号解决ABA问题
struct VersionedPointer {
    void* ptr;
    unsigned int version;
};

__device__ bool compareAndSwap(VersionedPointer* addr,
                               VersionedPointer expected,
                               VersionedPointer desired) {
    unsigned long long* addr_as_ull = (unsigned long long*)addr;
    unsigned long long expected_ull = *(unsigned long long*)&expected;
    unsigned long long desired_ull = *(unsigned long long*)&desired;
    
    unsigned long long old = atomicCAS(addr_as_ull, expected_ull, desired_ull);
    return old == expected_ull;
}
```

2. **内存管理与危险指针**

```cuda
// 危险指针机制防止过早释放
template<int MAX_THREADS>
struct HazardPointers {
    void* hazard[MAX_THREADS];
    
    __device__ void* protect(int tid, void** ptr_location) {
        void* ptr;
        do {
            ptr = *ptr_location;
            hazard[tid] = ptr;
            __threadfence();
        } while (ptr != *ptr_location);
        return ptr;
    }
    
    __device__ void clear(int tid) {
        hazard[tid] = nullptr;
    }
    
    __device__ bool is_safe_to_delete(void* ptr) {
        for (int i = 0; i < MAX_THREADS; i++) {
            if (hazard[i] == ptr) return false;
        }
        return true;
    }
};
```

### 7.4.5 无锁队列实现

Michael & Scott无锁队列：

```cuda
template<typename T>
struct LockFreeQueue {
    struct Node {
        T data;
        Node* next;
    };
    
    Node* head;
    Node* tail;
    
    __device__ void enqueue(T value) {
        Node* new_node = allocate_node();
        new_node->data = value;
        new_node->next = nullptr;
        
        while (true) {
            Node* last = tail;
            Node* next = last->next;
            
            if (last == tail) {
                if (next == nullptr) {
                    if (atomicCAS((unsigned long long*)&last->next,
                                 0, (unsigned long long)new_node) == 0) {
                        atomicCAS((unsigned long long*)&tail,
                                 (unsigned long long)last,
                                 (unsigned long long)new_node);
                        break;
                    }
                } else {
                    atomicCAS((unsigned long long*)&tail,
                             (unsigned long long)last,
                             (unsigned long long)next);
                }
            }
        }
    }
    
    __device__ bool dequeue(T* value) {
        while (true) {
            Node* first = head;
            Node* last = tail;
            Node* next = first->next;
            
            if (first == head) {
                if (first == last) {
                    if (next == nullptr) {
                        return false;  // 队列为空
                    }
                    atomicCAS((unsigned long long*)&tail,
                             (unsigned long long)last,
                             (unsigned long long)next);
                } else {
                    *value = next->data;
                    if (atomicCAS((unsigned long long*)&head,
                                 (unsigned long long)first,
                                 (unsigned long long)next) ==
                        (unsigned long long)first) {
                        free_node(first);
                        return true;
                    }
                }
            }
        }
    }
};
```

## 7.5 案例：高并发哈希表实现

### 7.5.1 设计目标与挑战

在自动驾驶场景中，高并发哈希表用于：
- 实时点云特征匹配
- 动态对象跟踪
- 传感器数据关联

设计挑战：
1. 极高的并发度（数千线程同时访问）
2. 动态扩容需求
3. 负载均衡
4. 内存效率

### 7.5.2 分段锁哈希表

```cuda
template<typename Key, typename Value, int NUM_SEGMENTS = 256>
class ConcurrentHashMap {
private:
    struct Entry {
        Key key;
        Value value;
        Entry* next;
        volatile int valid;  // 0: empty, 1: valid, -1: deleted
    };
    
    struct Segment {
        Entry** buckets;
        int bucket_count;
        int size;
        SpinLock lock;
    };
    
    Segment segments[NUM_SEGMENTS];
    
    __device__ int hash1(Key key) {
        // MurmurHash3的简化版本
        unsigned int h = key;
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }
    
    __device__ int get_segment(Key key) {
        return hash1(key) & (NUM_SEGMENTS - 1);
    }
    
public:
    __device__ bool insert(Key key, Value value) {
        int seg_idx = get_segment(key);
        Segment& seg = segments[seg_idx];
        
        seg.lock.acquire();
        
        int bucket_idx = hash1(key) % seg.bucket_count;
        Entry* entry = seg.buckets[bucket_idx];
        
        // 查找是否已存在
        while (entry != nullptr) {
            if (entry->valid == 1 && entry->key == key) {
                seg.lock.release();
                return false;  // 键已存在
            }
            entry = entry->next;
        }
        
        // 插入新条目
        Entry* new_entry = allocate_entry();
        new_entry->key = key;
        new_entry->value = value;
        new_entry->next = seg.buckets[bucket_idx];
        new_entry->valid = 1;
        
        seg.buckets[bucket_idx] = new_entry;
        seg.size++;
        
        // 检查是否需要扩容
        if (seg.size > seg.bucket_count * 0.75) {
            resize_segment(seg_idx);
        }
        
        seg.lock.release();
        return true;
    }
    
    __device__ bool find(Key key, Value* value) {
        int seg_idx = get_segment(key);
        Segment& seg = segments[seg_idx];
        
        // 读操作可以更宽松
        int bucket_idx = hash1(key) % seg.bucket_count;
        Entry* entry = seg.buckets[bucket_idx];
        
        while (entry != nullptr) {
            if (entry->valid == 1 && entry->key == key) {
                *value = entry->value;
                return true;
            }
            entry = entry->next;
        }
        return false;
    }
    
    __device__ bool remove(Key key) {
        int seg_idx = get_segment(key);
        Segment& seg = segments[seg_idx];
        
        seg.lock.acquire();
        
        int bucket_idx = hash1(key) % seg.bucket_count;
        Entry** current = &seg.buckets[bucket_idx];
        
        while (*current != nullptr) {
            Entry* entry = *current;
            if (entry->valid == 1 && entry->key == key) {
                entry->valid = -1;  // 标记删除
                seg.size--;
                seg.lock.release();
                return true;
            }
            current = &entry->next;
        }
        
        seg.lock.release();
        return false;
    }
};
```

### 7.5.3 无锁哈希表实现

```cuda
template<typename Key, typename Value>
class LockFreeHashMap {
private:
    struct Node {
        Key key;
        Value value;
        Node* next;
        unsigned int version;
    };
    
    struct Bucket {
        Node* head;
        unsigned int version;
    };
    
    Bucket* buckets;
    int bucket_count;
    
    __device__ unsigned int hash(Key key) {
        // CityHash的GPU版本
        unsigned long long k = key;
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccd;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53;
        k ^= k >> 33;
        return (unsigned int)k;
    }
    
public:
    __device__ bool insert(Key key, Value value) {
        unsigned int h = hash(key);
        int bucket_idx = h % bucket_count;
        
        Node* new_node = allocate_node();
        new_node->key = key;
        new_node->value = value;
        
        while (true) {
            Node* head = buckets[bucket_idx].head;
            
            // 检查键是否已存在
            Node* current = head;
            while (current != nullptr) {
                if (current->key == key) {
                    free_node(new_node);
                    return false;
                }
                current = current->next;
            }
            
            // 尝试插入
            new_node->next = head;
            if (atomicCAS((unsigned long long*)&buckets[bucket_idx].head,
                         (unsigned long long)head,
                         (unsigned long long)new_node) ==
                (unsigned long long)head) {
                atomicAdd(&buckets[bucket_idx].version, 1);
                return true;
            }
        }
    }
    
    __device__ bool find(Key key, Value* value) {
        unsigned int h = hash(key);
        int bucket_idx = h % bucket_count;
        
        Node* current = buckets[bucket_idx].head;
        while (current != nullptr) {
            if (current->key == key) {
                *value = current->value;
                return true;
            }
            current = current->next;
        }
        return false;
    }
};
```

### 7.5.4 Cuckoo哈希表（高速查找）

```cuda
template<typename Key, typename Value>
class CuckooHashMap {
private:
    struct Entry {
        Key key;
        Value value;
        unsigned int version;
        volatile int occupied;
    };
    
    Entry* table1;
    Entry* table2;
    int table_size;
    const int MAX_EVICTIONS = 500;
    
    __device__ unsigned int hash1(Key key) {
        return key * 0x9e3779b9;  // 黄金比例哈希
    }
    
    __device__ unsigned int hash2(Key key) {
        return key * 0x517cc1b7;  // 另一个质数
    }
    
public:
    __device__ bool insert(Key key, Value value) {
        unsigned int h1 = hash1(key) % table_size;
        unsigned int h2 = hash2(key) % table_size;
        
        // 尝试插入第一个表
        if (atomicCAS(&table1[h1].occupied, 0, 1) == 0) {
            table1[h1].key = key;
            table1[h1].value = value;
            atomicAdd(&table1[h1].version, 1);
            return true;
        }
        
        // 尝试插入第二个表
        if (atomicCAS(&table2[h2].occupied, 0, 1) == 0) {
            table2[h2].key = key;
            table2[h2].value = value;
            atomicAdd(&table2[h2].version, 1);
            return true;
        }
        
        // Cuckoo路径驱逐
        Entry evicted = {key, value, 0, 1};
        Entry* current_table = table1;
        unsigned int pos = h1;
        
        for (int i = 0; i < MAX_EVICTIONS; i++) {
            Entry temp = current_table[pos];
            current_table[pos] = evicted;
            
            if (temp.occupied == 0) return true;
            
            evicted = temp;
            
            // 切换到另一个表
            if (current_table == table1) {
                current_table = table2;
                pos = hash2(evicted.key) % table_size;
            } else {
                current_table = table1;
                pos = hash1(evicted.key) % table_size;
            }
        }
        
        // 需要重新哈希
        return false;
    }
    
    __device__ bool find(Key key, Value* value) {
        unsigned int h1 = hash1(key) % table_size;
        unsigned int h2 = hash2(key) % table_size;
        
        // 最多两次内存访问
        if (table1[h1].occupied && table1[h1].key == key) {
            *value = table1[h1].value;
            return true;
        }
        
        if (table2[h2].occupied && table2[h2].key == key) {
            *value = table2[h2].value;
            return true;
        }
        
        return false;
    }
};
```

### 7.5.5 性能优化策略

1. **SIMD友好的探测**

```cuda
// 使用向量化load进行批量查找
__device__ bool batch_find(unsigned int* keys, Value* values, int count) {
    // 利用向量化指令一次加载多个键
    uint4* vec_keys = (uint4*)keys;
    
    for (int i = 0; i < count/4; i++) {
        uint4 batch = vec_keys[i];
        
        // 并行计算4个哈希值
        unsigned int h0 = hash(batch.x) % bucket_count;
        unsigned int h1 = hash(batch.y) % bucket_count;
        unsigned int h2 = hash(batch.z) % bucket_count;
        unsigned int h3 = hash(batch.w) % bucket_count;
        
        // 预取bucket数据
        __builtin_prefetch(&buckets[h0], 0, 3);
        __builtin_prefetch(&buckets[h1], 0, 3);
        __builtin_prefetch(&buckets[h2], 0, 3);
        __builtin_prefetch(&buckets[h3], 0, 3);
        
        // 并行查找
        find_single(batch.x, &values[i*4]);
        find_single(batch.y, &values[i*4+1]);
        find_single(batch.z, &values[i*4+2]);
        find_single(batch.w, &values[i*4+3]);
    }
}
```

2. **内存池管理**

```cuda
// 线程本地内存池减少分配开销
template<typename T, int POOL_SIZE = 1024>
struct ThreadLocalPool {
    T pool[POOL_SIZE];
    int free_list[POOL_SIZE];
    int free_count;
    
    __device__ void init() {
        for (int i = 0; i < POOL_SIZE; i++) {
            free_list[i] = i;
        }
        free_count = POOL_SIZE;
    }
    
    __device__ T* allocate() {
        if (free_count > 0) {
            return &pool[free_list[--free_count]];
        }
        return nullptr;  // 需要全局分配
    }
    
    __device__ void deallocate(T* ptr) {
        int idx = ptr - pool;
        if (idx >= 0 && idx < POOL_SIZE) {
            free_list[free_count++] = idx;
        }
    }
};
```

### 7.5.6 实际应用：点云特征匹配

```cuda
// 用于自动驾驶中的实时点云配准
struct PointFeature {
    float3 position;
    float descriptor[32];  // FPFH特征
};

__global__ void match_point_features(
    PointFeature* source_cloud,
    int source_count,
    PointFeature* target_cloud,
    int target_count,
    LockFreeHashMap<unsigned int, int>* feature_map,
    int* matches
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= source_count) return;
    
    PointFeature& point = source_cloud[tid];
    
    // 计算特征哈希
    unsigned int feature_hash = 0;
    for (int i = 0; i < 32; i++) {
        feature_hash ^= __float_as_uint(point.descriptor[i]) * (i + 1);
    }
    
    // 在哈希表中查找匹配
    int match_idx;
    if (feature_map->find(feature_hash, &match_idx)) {
        // 验证空间一致性
        float3 target_pos = target_cloud[match_idx].position;
        float dist = length(point.position - target_pos);
        
        if (dist < 0.1f) {  // 10cm阈值
            matches[tid] = match_idx;
        }
    }
}
```

## 本章小结

本章深入探讨了CUDA中的原子操作和同步机制：

**关键概念**：
1. 原子操作通过硬件保证多线程环境下的操作原子性
2. 内存栅栏提供了不同范围的内存一致性保证
3. 无锁算法通过CAS等原语避免锁的开销
4. 高并发数据结构需要权衡吞吐量、延迟和内存效率

**性能公式**：
- 原子操作吞吐量：`T = min(内存带宽 / 操作大小, 原子单元数量 × 频率)`
- 锁竞争开销：`O = 串行化延迟 × 冲突概率 × 线程数`
- 无锁算法复杂度：`重试次数 = O(线程数 × 竞争强度)`

**优化要点**：
1. 批量化原子操作减少竞争
2. 使用合适的同步粒度
3. 选择适合访问模式的数据结构
4. 利用warp级原语优化

## 练习题

### 基础题

1. **原子操作选择**
   实现一个计数器，统计数组中大于阈值的元素个数。比较使用原子操作和归约的性能差异。
   
   <details>
   <summary>答案</summary>
   
   对于大规模数据，先在warp或block内归约，再使用原子操作更新全局计数器。这样可以将原子操作次数从N减少到N/32或N/blockSize。
   </details>

2. **内存栅栏应用**
   设计一个双缓冲系统，保证写入缓冲区的数据对读取线程完全可见。
   
   <details>
   <summary>答案</summary>
   
   写入完成后使用__threadfence()，然后更新缓冲区索引。读取时先读索引，再使用__threadfence()确保后续读取看到完整数据。
   </details>

3. **自旋锁优化**
   改进基础自旋锁，添加退避机制减少总线竞争。
   
   <details>
   <summary>答案</summary>
   
   使用指数退避：初始等待时间短，每次失败后加倍等待时间，设置最大等待时间上限。可以根据线程ID添加随机性避免同步冲突。
   </details>

### 挑战题

4. **无锁内存分配器**
   设计一个适用于GPU的无锁内存分配器，支持可变大小的内存块分配。
   
   提示：考虑使用多个大小类的内存池
   
   <details>
   <summary>答案</summary>
   
   使用分级内存池，每个大小类维护一个无锁栈。分配时根据请求大小选择合适的池，使用CAS操作管理空闲列表。需要处理内存碎片和池间迁移。
   </details>

5. **读写锁实现**
   实现一个读者优先的读写锁，允许多个读者同时访问，但写者独占。
   
   提示：使用原子计数器跟踪读者数量
   
   <details>
   <summary>答案</summary>
   
   使用两个原子变量：readers_count和writer_flag。读者通过原子增加readers_count获取锁，写者需要等待readers_count为0并设置writer_flag。需要处理饥饿问题。
   </details>

6. **并发B+树**
   设计一个GPU友好的并发B+树，支持高效的范围查询。
   
   提示：考虑使用乐观并发控制
   
   <details>
   <summary>答案</summary>
   
   使用版本号实现乐观读，只在修改时加锁。叶节点使用链表连接支持范围扫描。内部节点使用原子CAS更新指针，叶节点使用细粒度锁。
   </details>

### 开放性思考题

7. **原子操作 vs 事务内存**
   如果GPU支持硬件事务内存（HTM），会如何改变并发编程模型？分析优缺点。
   
   <details>
   <summary>答案</summary>
   
   HTM可以简化编程模型，自动处理冲突和回滚。但GPU的大规模并行特性可能导致频繁冲突，事务大小限制也是挑战。混合使用HTM和传统同步可能是最佳方案。
   </details>

8. **跨设备原子操作**
   设计一个支持多GPU原子操作的系统，考虑PCIe延迟和带宽限制。
   
   <details>
   <summary>答案</summary>
   
   可以使用层次化设计：设备内原子操作直接执行，跨设备操作通过主机端协调或使用GPUDirect RDMA。批量化跨设备操作，使用版本向量保证一致性。
   </details>

## 常见陷阱与错误

1. **死锁**
   - 错误：多个锁的获取顺序不一致
   - 正确：始终按相同顺序获取多个锁

2. **ABA问题**
   - 错误：仅比较指针值
   - 正确：使用版本号或标记指针

3. **内存序错误**
   - 错误：忽略内存重排序
   - 正确：正确使用内存栅栏

4. **原子操作滥用**
   - 错误：对所有共享变量使用原子操作
   - 正确：分析访问模式，必要时才使用

5. **活锁**
   - 错误：无限重试失败的CAS
   - 正确：添加退避机制或重试限制

## 最佳实践检查清单

- [ ] 原子操作是否已批量化？
- [ ] 是否选择了正确的同步粒度？
- [ ] 内存栅栏范围是否最小化？
- [ ] 是否处理了ABA问题？
- [ ] 无锁算法是否有退避机制？
- [ ] 是否避免了false sharing？
- [ ] 内存分配是否优化？
- [ ] 是否使用了合适的哈希函数？
- [ ] 并发数据结构是否支持扩展？
- [ ] 是否有性能监控和调试机制？
```
# study-cuda
This repository is for studying NVIDIA CUDA C++


## Contents

- 1_hello_kernel: what is kernel. kernel that prints "hello world!" twice using two threads.
- 2_get_threadid: Kernel that prints the thread id.
- 3_grid_and_block: what is grid and block. prints "hello world!" using multiple blocks.
- 4_warp: what is warp and branching.
- 5_global_memory: what is global memory. kernel that (element-wise) multiplies two arrays.
- 6_shared_memory: what is shared memory. kernel that multiplies two arrays using shared memory.
- 7_constant_memory: what is constant memory. kernel that multiplies two arrays using constant memory.
- 8_register: what is register. kernel that multiplies an array and a value using register.
- 9_memory_coalescing: what is memory coalescing. kernel that multiplies two arrays making use of coalescing.
- 10_pinned_memory: what is pinned memory. kernel that loads data from pinned memory.
- 11_dynamic_parallelism: what is dynamic parallelism. kernel that calls another kernel.
- 12_stream: what is stream. run two kernels in parallel using one GPU.
- 13_multi_gpus: run two kernels in parallel using two GPUs.

# study-cuda
This repository is for studying NVIDIA CUDA C++


## Contents

- 1_hello_kernel: what is kernel. kernel that prints "hello world!" twice using two threads.
- 2_grid_and_block: what is grid and block. prints "hello world!" using multiple blocks.
- 3_dim3: what is dim3. launch a kernel using dim3.

## To be writen

- 4_threadid: kernel that prints the thread id.
- 5_warp: what is warp and branching.
- 6_global_memory: what is global memory. kernel that (element-wise) multiplies two arrays.
- 7_shared_memory: what is shared memory. kernel that multiplies two arrays using shared memory.
- 8_constant_memory: what is constant memory. kernel that multiplies two arrays using constant memory.
- 9_register: what is register. kernel that multiplies an array and a value using register.
- 10_memory_coalescing: what is memory coalescing. kernel that multiplies two arrays making use of coalescing.
- 11_pinned_memory: what is pinned memory. kernel that loads data from pinned memory.
- 12_dynamic_parallelism: what is dynamic parallelism. kernel that calls another kernel.
- 13_stream: what is stream. run two kernels in parallel using one GPU.
- 14_multi_gpus: run two kernels in parallel using two GPUs.

## Warp
Warp is a unit of execution of streaming multiprocessor.
A warp is consists of 32 consecutive threads, and all threads in a warp execute the same operation if they are not branched.
A warp can branch by the `if` clause and the loop (`for`, `while`) clause.
When threads in a warp branch due to such clauses, we say that there is a branch divergence in a warp.
Suppost that the set of threads A and the set of threads B in a warp follows different code path.
One set of threads, say A, are executed together first. At the moment threads in A are executed, threads in B are stalled.
After A, threads in B are executed together. Likewise, threads in A are stalled while threads in B are executed.
Therefore, when there is a branch divergence, some amount of streaming multiprocessor will not be utilized and will downgrade the performance of parallelism.

## Branch Example
In the following example, no_branch() function prints the block id and the thread id of the thread that executes the kernel.
branch() function checks thread id, and branches to two code blocks; one that prints even numbered threads and one that prints odd numbered threads.

```bash
#include <stdio.h>

__global__ void no_branch(int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  printf("[BLOCK %d] thread id %d\n", blockIdx.x, tid, n);
}

__global__ void branch(int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid % 2 == 0) {
    printf("[BLOCK %d] even numbered thread id %d\n", blockIdx.x, tid, n);
  }
  else {
    printf("[BLOCK %d] odd numbered thread id %d\n", blockIdx.x, tid, n);
  }
}


int main() {
  const int N = 16;

  int num_thread_per_block = 8;
  int num_block = (int)(N / num_thread_per_block);
  if(N % num_thread_per_block != 0)
    num_block += 1;

  no_branch<<<num_block, num_thread_per_block>>>(N);
  cudaDeviceSynchronize();

  branch<<<num_block, num_thread_per_block>>>(N);
  cudaDeviceSynchronize();
}
```

The results of the call no_branch<<<...>>>() is as follows. 
The 8 consecutive threads in BLOCK 0 are executed together within warp 0,
and the 8 consecutive threads in BLOCK 1 are executed together within warp 1.
After all threads in warp 1 are executed, threads in warp 0 are executed.

```bash
[BLOCK 1] thread id 8
[BLOCK 1] thread id 9
[BLOCK 1] thread id 10
[BLOCK 1] thread id 11
[BLOCK 1] thread id 12
[BLOCK 1] thread id 13
[BLOCK 1] thread id 14
[BLOCK 1] thread id 15
[BLOCK 0] thread id 0
[BLOCK 0] thread id 1
[BLOCK 0] thread id 2
[BLOCK 0] thread id 3
[BLOCK 0] thread id 4
[BLOCK 0] thread id 5
[BLOCK 0] thread id 6
[BLOCK 0] thread id 7
```

The results of the call branch<<<...>>>() is as follows.
There is a branch divergence in each warp because of the `if` clause.
When there is a branch divergence, we can see that only a part of the threads in a warp is executed at the same time.
The 4 odd numbered threads in warp 0 are executed together, while the 4 even numbered threads in warp 0 stalled during the execution.
After the excution of warp 0 for odd numbered threads, warp 1 for odd numbered threads is executed.
After the execution of warp 1 for odd numbered threads, warp 1 for even numbered threads is executed. Finally, warp 0 for even numbered threads is executed.

```bash
[BLOCK 0] odd numbered thread id 1
[BLOCK 0] odd numbered thread id 3
[BLOCK 0] odd numbered thread id 5
[BLOCK 0] odd numbered thread id 7
[BLOCK 1] odd numbered thread id 9
[BLOCK 1] odd numbered thread id 11
[BLOCK 1] odd numbered thread id 13
[BLOCK 1] odd numbered thread id 15
[BLOCK 1] even numbered thread id 8
[BLOCK 1] even numbered thread id 10
[BLOCK 1] even numbered thread id 12
[BLOCK 1] even numbered thread id 14
[BLOCK 0] even numbered thread id 0
[BLOCK 0] even numbered thread id 2
[BLOCK 0] even numbered thread id 4
[BLOCK 0] even numbered thread id 6
```


## Note
Unlinke blocks and threads, warp is set by warp scheduler.
We should keep in mind the following hierarchy.
- a grid is composed of blocks
- a block is composed of warps
- a warp is composed of 32 threads (if the number of threads in a block is less than 32, the rest of the threads are stalled)
- all (unstalled) threads in a warp execute the same code in parallel (single instruction multiple thread)
- a thread in a warp stalls if there is a branch divergence.
- more warps can be executed in parallel with fewer registers (and shared memory) used per thread. (the number of warps in a streaming processor depends on the amount of resource a warp use)

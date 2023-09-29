## What is grid and block
A kernel is executed N times in parallel by N threads. 
A grid contains N threads hierarchically using blocks. 
In other words, a grid consists of NB blocks, and a block consists of MT threads, where NB * NT = N.
A block is a unit that can be executed by one streaming multiprocessor (hardware).
A block size (i.e., the number of threads in the block; NT) cannot exceed 1024.

## Example: kernel that prints "hello kernel!" using three blocks and two threads
In the following example, we define a kernel that prints "hello kernel" using multiple blocks.
This kernel will be executed six times since we set the number of blocks to 3 and the number of threads per block to 2.

```bash
#include <stdio.h>

__global__ void hello_kernel() {
  printf("hello kernel!\n");
}

int main() {
  hello_kernel<<<3, 2>>>();
  cudaDeviceSynchronize();
}
```

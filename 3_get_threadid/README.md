## Thread ID
A kernel is a function that executed N times in parallel by N threads.
Each execution of a kernel (in a block) has distict thread id, which can be determined by a keyword 'threadIdx'.

In theory, a kenrel in CUDA is a C++ function that executed N times in paralell by N threads.
In reality, since there is a limited CUDA processors (say M), a GPU may execute M threads in parallel.

## How to define a kernel function
Just add \__global__ in front of a C++ function, which is a declaration specifier.
The return type of a kernel must be void.

## How to call a kernel function
A kernel is called using <<<NB, NT>>> syntax (execution configuration syntax).
NB is the number of blocks in a grid, and NT is the number of threads in a block,
where grid is a set of N threads that executes the kernel. We will see what is grid and block in the later study.
At this point you can think that NB and NT are integers, and the total number of threads is equal to NB * NT.
Later we will see a differnt type (rather than int) of NB and NT.

## Example: kernel that prints "hello kernel!" twice using two threads
In the following example, we define a kernel that prints "hello kernel".
This kernel will be executed two times since we set the number of blocks to 1 and the number of threads per block to 2 (and thus total number of threads is equal to 2).


```bash
#include <stdio.h>

__global__ void hello_kernel() {
  printf("hello kernel!\n");
}

int main() {
  hello_kernel<<<1, 2>>>();
}
```

When you run the program, you may not see any prints because the kernel execution and the CPU program execution are asyncronous,
and the CPU program does not wait for kernel execution to end.
To make CPU program to wait until all kernel execution ends, we should add cudaDeviceSynchronize().
The modified code is as follows.


```bash
#include <stdio.h>

__global__ void hello_kernel() {
  printf("hello kernel!\n");
}

int main() {
  hello_kernel<<<1, 2>>>();
  cudaDeviceSynchronize();
}
```


## How to compile a CUDA program
First of all, a cuda source ends with .cu.
Second of all, a cuda source code is compiled using nvcc (NVidia Cuda Compiler).

```
nvcc hello_world.cu -o program
./program
```

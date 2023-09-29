## What is dim3
dim3 is a type in CUDA C++ that consists of three integers: dim3.x, dim3.y, and dim3.z.
If you assign an integer i to a dim3 variable d3, then d3.x = i and d3.y = d3.z = 1.

## dim3 in execution configuration
In the execution configuration <<<NB, NT>>>, NB and NT are in fact dim3 variables.
So, setting NT = 2 actually sets NT.x = 2, NT.y = 1, NT.z = 1.
The total number of blocks (corr. threads) is equal to NB.x * NB.y * NB.z (corr. NT.x * NT.y * NT.z).
There are limit for x, y, z in the execution configuration, and the limits depends on the cuda comutability.

## Example: kernel that prints "hello kernel!" using dim3
In the following example, we define a kernel that prints "hello kernel" using dim3.

```bash
#include <stdio.h>

__global__ void hello_kernel() {
  printf("hello kernel!\n");
}

int main() {

  dim3 NB = 3;  // x = 3, y = 1, z = 1 -> 3 blocks
  dim3 NT = 2;  // x = 2, y = 1, z = 1 -> 2 threads per block
  printf("NB.x=%d, NB.y=%d, NB.z=%d\n", NB.x, NB.y, NB.z);
  printf("NT.x=%d, NT.y=%d, NT.z=%d\n", NT.x, NT.y, NT.z);

  printf("First call\n");
  hello_kernel<<<NB, NT>>>();
  cudaDeviceSynchronize();

  NB.y = 2;  // x = 3, y = 2, z = 1 -> 6 blocks
  NT.z = 2;  // x = 2, y = 1, z = 2 -> 4 threads per block
  printf("NB.x=%d, NB.y=%d, NB.z=%d\n", NB.x, NB.y, NB.z);
  printf("NT.x=%d, NT.y=%d, NT.z=%d\n", NT.x, NT.y, NT.z);
  printf("Second call\n");
  hello_kernel<<<NB, NT>>>();
  cudaDeviceSynchronize();
}
```

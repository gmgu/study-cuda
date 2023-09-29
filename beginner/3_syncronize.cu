#include <stdio.h>
#include <cuda.h>

__global__
void print()
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int dim_block = blockDim.x;
  printf("THREAD %d in BLOCK %d\n", tid + bid * dim_block, bid);
}


int main()
{
  dim3 dim_grid(3);
  dim3 dim_block(2);

  printf("Launching CUDA Kernel with %d blocks and %d threads per block\n", dim_grid.x, dim_block.x);
  print<<<dim_grid, dim_block>>>();
  cudaDeviceSynchronize();  // wait until all threads finish.

  return 0;
}

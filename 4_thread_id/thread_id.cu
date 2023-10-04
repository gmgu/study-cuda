#include <stdio.h>

__global__ void print_thread_id() {
  // gridDim = NB: number of blocks per each dimension of a grid
  // blockDim = NT: number of threads per each dimension of a block
  // blockIdx: current block index (x, y, z)
  // threadIdx: current thread index (x, y, z)

  // there are gridDim.x * gridDim.y * gridDim.z blocks -> make it [0, #blocks)
  int block_id = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  // there are blockDim.x * blockDim.y * blockDim.z threadss -> make it [0, #threads)
  int thread_id = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  printf("Thread %d in Block %d: gridDim=(%d, %d, %d) and blockDim=(%d, %d, %d) and blockIdx=(%d, %d, %d) and threadIdx=(%d, %d, %d)\n", 
          thread_id, block_id,
          gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 
          blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);

}

int main() {
  // 1D: 2 blocks and 3 threads per block
  printf("1D example\n");
  dim3 NB = dim3(2, 1, 1);
  dim3 NT = dim3(3, 1, 1);
  print_thread_id<<<NB, NT>>>();
  cudaDeviceSynchronize();

  // 2D: 4 blocks and 3 threads per block
  printf("2D example\n");
  NB = dim3(2, 2, 1);
  NT = dim3(1, 3, 1);
  print_thread_id<<<NB, NT>>>();
  cudaDeviceSynchronize();

  // 3D: 4 blocks and 12 threads per block
  printf("3D example\n");
  NB = dim3(2, 1, 2);
  NT = dim3(3, 2, 2);
  print_thread_id<<<NB, NT>>>();
  cudaDeviceSynchronize();
}

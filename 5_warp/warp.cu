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

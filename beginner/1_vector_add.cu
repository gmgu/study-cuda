#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define N 10000000
#define MAX_ERR 1e-6


// __global__ means the function is a CUDA kernal
__global__ void vector_add(float *C, float *A, float *B, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  // each of them as x, y, and z
                                                    // threadIdx: the index of a thread in a block
                                                    // blockIdx: the index of a block in a grid
                                                    // blockDim: the number of threads in a block
  if(tid < n)  // for the case when a thread (or block) does not divide n
    C[tid] = A[tid] + B[tid];  // each thread computes this single addition
}


// casual C++ main function
int main() {
  // h_A, h_B, h_C are stored in the CPU (host) memory.
  // d_A, d_B, d_C are stored in the GPU (device) memory
  float* h_A = NULL;
  float* h_B = NULL;
  float* h_C = NULL; 
  float* d_A = NULL;
  float* d_B = NULL;
  float* d_C = NULL;

  // allocate host memory
  h_A = (float*)malloc(sizeof(float) * N);
  h_B = (float*)malloc(sizeof(float) * N);
  h_C = (float*)malloc(sizeof(float) * N);

  // initialize host arrays
  for(int i = 0; i < N; i++) {
    h_A[i] = 4.0f;  // recall that 1.0f is a float number (4 bytes), and 1.0 is a double number (8 bytes)
    h_B[i] = 5.0f;
  }

  // allocate device memory
  cudaMalloc((void**)&d_A, sizeof(float) * N);  // __host__ __device__ cudaError_t cudaMalloc(void** devPtr, size_t size)
  cudaMalloc((void**)&d_B, sizeof(float) * N);  // void**: it is a pointer (devPtr) of a pointer (d_B)
  cudaMalloc((void**)&d_C, sizeof(float) * N);  // it will initialize d_C to point an array of size 'sizeof(float) * N'

  // transfer data from host to device memory
  cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);  // __host__ cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
  cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice);  // note that it (to, from ...) and not (from, to ...). (How non-intuitive!)

  int num_thread_per_block = 256;  // which is 16 x 16 (this is the most common number)
  int num_block = (int)(N / num_thread_per_block);  // note that blockDim.x = num_thread_per_block in a kernel
  if(N % num_thread_per_block != 0)
    num_block += 1;  // we just add one more block, and handle the edge case

  // executing kernel 
  vector_add<<<num_block, num_thread_per_block>>>(d_C, d_A, d_B, N);  // <<<,>>> is called an execution configuration.
  // there is actually a different (and more complicated) way to execute a kernel.
  // <<<,>>> is just another way of doing that, and the two ways are essentially the same.
  // anyway, after this line, d_C is computed and stored in the GPU memory

  // transfer data back to host memory
  cudaMemcpy(h_C, d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);  // we always need to specify the flow (in this case, GPU -> CPU)

  // verification
  for(int i = 0; i < N; i++) {
    assert(fabs(h_C[i] - h_A[i] - h_B[i]) < MAX_ERR);  // MAX_ERR is for the floating point error
  }
  printf("Correctly computed C (C[0] = %f)\n", h_C[0]);

  // deallocate device memory
  cudaFree(d_A);  // __host__ __device__ cudaError_t cudaFree(void* devPtr)
  cudaFree(d_B);
  cudaFree(d_C);

  // deallocate host memory
  free(h_A); 
  free(h_B); 
  free(h_C);
}

## Global memory
The global memory of GPU can be accessed by any grid, any block, and any threads.
It is much slower than other types of memories (such as shared memory, constant memory, etc) because 
(1) the hardware itself is a slow but large memory, and (2) each thread can both read and write data to the memory.

## Example: multiplying two vectors
In the following example script, we element-wise multiply two vectors A and B, and puts the results to the vector C.
Arrays d_A, d_B, d_C are allocated in the global memory of the GPU when we call `cudaMalloc()`.
We use `cudaMemcpy()` to initialize the arrays allocated in GPU with values of the arrays reside in CPU. (data communication occurs from CPU to GPU)
Given the pointers of A and B in GPU, computing the multiplication is straight forward.
The results stored in C in GPU is handed over to CPU via `cudaMemcpy()`.
GPU memories are cleaned by `cudaFree()` and CPU memories are cleaned by `free()`.

```bash
#include <stdio.h>
#include <cuda.h>

#define N 10000000


__global__ void vector_mul(float *A, float *B, float *C, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n)
        C[tid] = A[tid] * B[tid];
}


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
    h_A[i] = 3.0f;
    h_B[i] = 3.0f;
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

  vector_mul<<<num_block, num_thread_per_block>>>(d_A, d_B, d_C, N);

  // transfer data back to host memory
  // cudaDeviceSynchronize is called inside of cudaMemcpy
  cudaMemcpy(h_C, d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);  // we always need to specify the flow (in this case, GPU -> CPU)

  printf("C[0] = %f\n", h_C[0]);

  // deallocate device memory
  cudaFree(d_A);  // __host__ __device__ cudaError_t cudaFree(void* devPtr)
  cudaFree(d_B);
  cudaFree(d_C);

  // deallocate host memory
  free(h_A); 
  free(h_B); 
  free(h_C);

  return 0;
}
```

#include <stdio.h>  // printf
#include <cuda_runtime.h>  // cudaDeviceProp


int main() {
  cudaDeviceProp prop;  // a structure contains information of GPUS.
  // a complete member variables are in https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp
  cudaGetDeviceProperties(&prop, 0);  // __host__ __cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device)
  // in case you have multiple GPUs set device parameter to the GPU number you want to see. (or loop through the number of GPUs)


  // below, we print some information
  printf("Device Name: %s\n", prop.name);  // char[256]
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);  // int. major and minor compute capability number, respectively
  printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);  // int
  printf("Total Global Memory: %zu bytes (%.2f GiB)\n", prop.totalGlobalMem, prop.totalGlobalMem / 1024.0 / 1024 / 1024);  // size_t
  printf("Shared Memory per Block: %zu bytes (%.2f KiB)\n", prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);  // size_t
  printf("Total Const Memory: %zu bytes (%.2f KiB)\n", prop.totalConstMem, prop.totalConstMem / 1024.0);  // size_t
  printf("Number of Registers Per Block: %d\n", prop.regsPerBlock);  // int
  printf("Warp Size (the number of threads in a warp): %d\n", prop.warpSize);  // int
  printf("Maximum Threads per Block: %d\n", prop.maxThreadsPerBlock);  // int
  printf("Maximum Threads for BlockDim 0: %d\n", prop.maxThreadsDim[0]);  // int[3]
  printf("Maximum Threads for BlockDim 1: %d\n", prop.maxThreadsDim[1]);  // int[3]
  printf("Maximum Threads for BlockDim 2: %d\n", prop.maxThreadsDim[2]);  // int[3]
  printf("Maximum Blocks for Grid 0: %d\n", prop.maxGridSize[0]);  // int[3]
  printf("Maximum Blocks for Grid 1: %d\n", prop.maxGridSize[1]);  // int[3]
  printf("Maximum Blocks for Grid 2: %d\n", prop.maxGridSize[2]);  // int[3]

  return 0;
}


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

#include <stdio.h>

__global__ void hello_kernel() {
  printf("hello kernel!\n");
}

int main() {
  hello_kernel<<<3, 2>>>();
  cudaDeviceSynchronize();
}

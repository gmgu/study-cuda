
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Max grid: %d\n", prop.maxGridSize[0]);
  printf("Max grid: %d\n", prop.maxGridSize[1]);
    printf("Max grid: %d\n", prop.maxGridSize[2]);
    printf("Max Thread: %d %d %d %d\n", prop.maxThreadsPerBlock, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("#multiprocessors: %d\n", prop.multiProcessorCount);

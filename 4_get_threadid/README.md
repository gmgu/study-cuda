## Thread ID
A kernel is a function that executed N times in parallel by N threads.
Each execution of a kernel (in a block) has distict thread id, which can be determined by a keyword 'threadIdx'.

## Example: kernel that prints thread id
In the following example, we define a kernel that prints "hello kernel".
This kernel will be executed two times since we set the number of blocks to 1 and the number of threads per block to 2 (and thus total number of threads is equal to 2).


```bash
#include <stdio.h>

  cudaDeviceSynchronize();
}
```

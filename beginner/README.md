# Beginner
Here, we implement some basic operations using CUDA


## Prerequisite
You need nvcc (nvidia cuda compiler) to compile CUDA C++ code (so install it).

You need NVIDIA GPU. I am using a GTX 3060 12GB GPU

## Operations
- Vector Addition:
-- Input = two vectors A and B of length n.
-- Ouput = a vector C of length n, where C[i] = A[i] + B[i] for 0 <= i < n.
```bash
nvcc 1_vector_add.cu -o 1_vector_add
./1_vector_add
```

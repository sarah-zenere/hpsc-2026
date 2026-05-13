#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

//count values into buckets
//wryyyy
__global__ void count_kernel(int *key, int *bucket, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    atomicAdd(&bucket[key[idx]], 1);
  }
}

int main() {
  int n = 50;
  int range = 5;

  std::vector<int> key(n);

  //random values
  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  //bucket array
  std::vector<int> bucket(range, 0);
  
  int *d_key, *d_bucket;

  cudaMalloc(&d_key, n * sizeof(int));
  cudaMalloc(&d_bucket, range * sizeof(int));
  cudaMemcpy(d_key, key.data(),
             n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemset(d_bucket, 0, range * sizeof(int));

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_key,
      d_bucket,
      n
  );

  cudaMemcpy(bucket.data(),
             d_bucket,
             range * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0, j = 0; i < range; i++) {
    for (; bucket[i] > 0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i = 0; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");
  cudaFree(d_key);
  cudaFree(d_bucket);

  return 0;
}

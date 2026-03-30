#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
    float* l, float* m, float* O){
        int tid = threadIdx.x;
        int bx = blockIdx.x; int by = blockIdx.y; //batch and head index

        // Offset into Q,K,V,O,l,m-different for each batch and head
}  

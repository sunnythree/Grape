#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "grape/util/dropout_util.h"

namespace Grape
{

    __global__ void dropout_kernel(float *input, int size, float *rand, float prob, float scale)
    {
        int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
    }

#ifdef GPU
    void forward_dropout_gpu(int batch, int in_dim, float * input, float *rand_data, float probability, float scale)
    {
        int size = batch*in_dim;
        cuda_random(layer.rand_gpu, size);
        dropout_kernel(input，size,rand_data,probability,scale);
        cuda_check_error(cudaPeekAtLastError());
    }
    void backward_dropout_gpu(int batch, int in_dim, float * input, float *rand_data, float probability, float scale)
    {
        dropout_kernel(input，size,rand_data,probability,scale);
        cuda_check_error(cudaPeekAtLastError());
    }
#endif
}
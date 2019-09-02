#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "grape/util/dropout_util.h"
#include "grape/util/cuda.h"

namespace Grape
{

#ifdef GPU
    __global__ void dropout_kernel(float *input, float *output,int size, float *rand, float prob, float scale)
    {
        int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(id < size) output[id] = (rand[id] < prob) ? 0 : input[id]*scale;
    }

    void forward_dropout_gpu(int batch, int in_dim, float * input,float *output, float *rand_data, float probability, float scale)
    {
        int size = batch*in_dim;
        cuda_random(rand_data, size);
        dropout_kernel<<<cuda_gridsize(size), BLOCK>>>(input,output,size,rand_data,probability,scale);
        cuda_check_error(cudaPeekAtLastError());
    }
    void backward_dropout_gpu(int batch, int in_dim, float * input, float * output, float *rand_data, float probability, float scale)
    {
        int size = batch*in_dim;
        dropout_kernel<<<cuda_gridsize(size), BLOCK>>>(input,output,size,rand_data,probability,scale);
        cuda_check_error(cudaPeekAtLastError());
    }
#endif
}
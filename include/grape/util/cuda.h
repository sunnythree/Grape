#ifndef __GRAPE_CUDA_H__
#define __GRAPE_CUDA_H__

#ifdef GPU
    #define BLOCK 512
    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"
#endif

namespace Grape
{
#ifdef GPU
    extern "C"{
    void cuda_set_device(int n);
    int  cuda_get_device();
    void cuda_check_error(cudaError_t status);
    dim3 cuda_gridsize(size_t n);
    void cuda_random(float *x_gpu, size_t n);
    void cuda_malloc(void** ptr, int size);
    void cuda_free(void *x_gpu);
    void cuda_memset(void *x_gpu, float alpha, size_t n);
    void cuda_pull_array(void *x_gpu, void *x, size_t n);
    void cuda_push_array(void *x_gpu, void *x, size_t n);
    void cuda_async_push_array(void *x_gpu, void *x, size_t n,const cudaStream_t& stream);
    void cuda_pointer_get_attributes(cudaPointerAttributes* attr,void* x_gpu);
    #ifdef CUDNN
    cudnnHandle_t cudnn_handle();
    #endif
    cublasHandle_t cuda_blas_handle();
    }
#endif
} // namespace Grape


#endif
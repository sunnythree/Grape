
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#ifdef GPU
    #include "grape/util/cuda.h"
    #include "grape/util/blas.h"
    #include "grape/log.h"
#endif

namespace Grape
{

#ifdef GPU
    void cuda_set_device(int n)
    {
        cuda_check_error(cudaSetDevice(n));
    }

    int cuda_get_device()
    {
        int n = 0;
        cuda_check_error(cudaGetDevice(&n));
        return n;
    }

    void cuda_check_error(cudaError_t status)
    {
        cudaError_t status2 = cudaGetLastError();
        if (status != cudaSuccess){   
            const char *s = cudaGetErrorString(status);
            char buffer[256];
            printf("CUDA Error: %s\n", s);
            assert(0);
            snprintf(buffer, 256, "CUDA Error: %s", s);
            printf(buffer);
        } 
        if (status2 != cudaSuccess){   
            const char *s = cudaGetErrorString(status);
            char buffer[256];
            printf("CUDA Error Prev: %s\n", s);
            assert(0);
            snprintf(buffer, 256, "CUDA Error Prev: %s", s);
            printf(buffer);
        } 
    }

    dim3 cuda_gridsize(size_t n)
    {
        size_t k = (n-1) / BLOCK + 1;
        size_t x = k;
        size_t y = 1;
        if(x > 65535){
            x = ceil(sqrt(k));
            y = (n-1)/(x*BLOCK) + 1;
        }
        dim3 d = {x, y, 1};
        //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
        return d;
    }


    void cuda_random(float* x_gpu, size_t n)
    {
        static curandGenerator_t gen[16];
        static int init[16] = {0};
        int i = cuda_get_device();
        if(!init[i]){
            curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
            init[i] = 1;
        }
        curandGenerateUniform(gen[i], x_gpu, n);
        cuda_check_error(cudaPeekAtLastError());
    }

    void cuda_malloc(void** ptr,int size)
    {
        cuda_check_error(cudaMalloc((void**)ptr,size));
    }

    void cuda_free(void* x_gpu)
    {
        cuda_check_error(cudaFree((void*)x_gpu));
    }

    void cuda_memset(void *x_gpu, float alpha, size_t n)
    {
        cuda_check_error(cudaMemset(x_gpu,alpha,n));
    }

    void cuda_push_array(void *x_gpu, void *x, size_t n)
    {
        cuda_check_error(cudaMemcpy(x_gpu, x, n, cudaMemcpyHostToDevice));
    }

    void cuda_async_push_array(void *x_gpu, void *x, size_t n,const cudaStream_t& stream)
    {
        cuda_check_error(cudaMemcpyAsync(x_gpu, x, n, cudaMemcpyHostToDevice, stream));
    }

    void cuda_pull_array(void *x_gpu, void *x, size_t n)
    {
        cuda_check_error(cudaMemcpy(x, x_gpu, n, cudaMemcpyDeviceToHost));
    }

    void cuda_pointer_get_attributes(cudaPointerAttributes* attr,void* x_gpu)
    {
        cuda_check_error(cudaPointerGetAttributes(attr, x_gpu));
    }

    #ifdef CUDNN
    cudnnHandle_t cudnn_handle()
    {
        static int init[16] = {0};
        static cudnnHandle_t handle[16];
        int i = cuda_get_device();
        if(!init[i]) {
            cudnnCreate(&handle[i]);
            init[i] = 1;
        }
        return handle[i];
    }
    #endif

    cublasHandle_t cuda_blas_handle()
    {
        static int init[16] = {0};
        static cublasHandle_t handle[16];
        int i = cuda_get_device();
        if(!init[i]) {
            cublasCreate(&handle[i]);
            init[i] = 1;
        }
        return handle[i];
    }
#endif
} // namespace Grape


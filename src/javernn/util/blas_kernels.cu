#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <assert.h>

extern "C" {
#include "javernn/util/blas.h"
#include "javernn/util/cuda.h"
}

namespace javernn
{

    __global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
    }

    extern "C" void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
    {
        axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
    }

    extern "C" void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
    {
        axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
    }

    extern "C" void copy_gpu(int N, float * X, int INCX, float * Y, int INCY)
    {
        copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
    }

    extern "C" void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
    {
        copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
    }

    extern "C" void pow_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
    {
        pow_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void mul_kernel(int N, float *X, int INCX, float *Y, int INCY)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < N) Y[i*INCY] *= X[i*INCX];
    }

    extern "C" void mul_gpu(int N, float * X, int INCX, float * Y, int INCY)
    {
        mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void mask_kernel(int n,  float *x, float mask_num, float *mask, float val)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < n && mask[i] == mask_num) x[i] = val;
    }

    extern "C" void mask_gpu(int N, float * X, float mask_num, float * mask, float val)
    {
        mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, val);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void scale_mask_kernel(int n,  float *x, float mask_num, float *mask, float scale)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < n && mask[i] == mask_num) x[i] *= scale;
    }

    extern "C" void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale)
    {
        scale_mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, scale);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void const_kernel(int N, float ALPHA, float *X, int INCX)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < N) X[i*INCX] = ALPHA;
    }

    extern "C" void const_gpu(int N, float ALPHA, float * X, int INCX)
    {
        const_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void add_kernel(int N, float ALPHA, float *X, int INCX)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < N) X[i*INCX] += ALPHA;
    }


    extern "C" void add_gpu(int N, float ALPHA, float * X, int INCX)
    {
        add_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void scal_kernel(int N, float ALPHA, float *X, int INCX)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < N) X[i*INCX] *= ALPHA;
    }

    extern "C" void scal_gpu(int N, float ALPHA, float * X, int INCX)
    {
        scal_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < N) X[i*INCX] = ALPHA;
    }

    extern "C" void fill_gpu(int N, float ALPHA, float * X, int INCX)
    {
        fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
        cuda_check_error(cudaPeekAtLastError());
    }


    __global__ void mult_add_into_kernel(int n, float *a, float *b, float *c)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < n){
            c[i] += a[i]*b[i];
        }
    }

    extern "C" void mult_add_into_gpu(int num, float *a, float *b, float *c)
    {
        mult_add_into_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, c);
        cuda_check_error(cudaPeekAtLastError());
    }
    

    __device__ void softmax_device(float *input, int n, float temp, int stride, float *output)
    {
        int i;
        float sum = 0;
        float largest = -INFINITY;
        for(i = 0; i < n; ++i){
            int val = input[i*stride];
            largest = (val>largest) ? val : largest;
        }
        for(i = 0; i < n; ++i){
            float e = expf(input[i*stride]/temp - largest/temp);
            sum += e;
            output[i*stride] = e;
        }
        for(i = 0; i < n; ++i){
            output[i*stride] /= sum;
        }
    }

    __global__ void softmax_kernel(float *input, int n, float temp, int stride, float *output)
    {
        int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if (id >= n) return;
        softmax_device(input, n, temp, stride, output);
    }

    extern "C" void softmax_gpu(float *input, int n, float temp, int stride, float *output)
    {
        softmax_kernel<<<cuda_gridsize(n), BLOCK>>>(input, n, temp, stride ,output);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void softmax_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error)
    {
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i < n){
            float t = truth[i];
            float p = pred[i];
            error[i] = (t) ? -log(p) : 0;
            delta[i] = t-p;
        }
    }

    extern "C" void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
    {
        softmax_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
        cuda_check_error(cudaPeekAtLastError());
    }
}








#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <assert.h>

extern "C" {
#include "grape/util/blas.h"
#include "grape/util/cuda.h"
}

namespace Grape
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

    __global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
    {
        int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if (index >= n*size*batch) return;
        int i = index % size;
        index /= size;
        int j = index % n;
        index /= n;
        int k = index;

        output[(k*n+j)*size + i] += biases[j];
    }

    void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
    {
        int num = n*size*batch;

        add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
        cuda_check_error(cudaPeekAtLastError());
    }

    __global__ void backward_bias_conn_kernel(float *bias_updates, float *delta, int batch, int n)
    {
        int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if (index >= n) return;
        int b;
        float sum = 0;
        for(b = 0; b < batch; ++b){
            int i = b*n + index;
            sum += delta[i];
        }
        bias_updates[index] += sum;
    }

    __global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
    {
        __shared__ float part[BLOCK];
        int i,b;
        int filter = blockIdx.x;
        int p = threadIdx.x;
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; i += BLOCK){
                int index = p + i + size*(filter + n*b);
                sum += (p+i < size) ? delta[index] : 0;
            }
        }
        part[p] = sum;
        __syncthreads();
        if (p == 0) {
            for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
        }
    }

    void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
    {
        if(size == 1){
            backward_bias_conn_kernel<<<cuda_gridsize(n), BLOCK>>>(bias_updates, delta, batch, n);
        }else{
            backward_bias_kernel<<<n, BLOCK>>>(bias_updates, delta, batch, n, size);
        }
        cuda_check_error(cudaPeekAtLastError());
    }
}








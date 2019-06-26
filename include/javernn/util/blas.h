#ifndef __JAVERNN_BLAS_H__
#define __JAVERNN_BLAS_H__

#ifdef GPU
#include "javernn/util/cuda.h"
#endif

namespace javernn
{
    void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);
    void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
    void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
    void scal_cpu(int N, float ALPHA, float *X, int INCX);
    void fill_cpu(int N, float ALPHA, float * X, int INCX);
    void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
    void mult_add_into_cpu(int N, float *X, float *Y, float *Z);
    void add_cpu(int N, float *X, int INCX, float *Y, int INCY);

    #ifdef GPU
    void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
    void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
    void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
    void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
    void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
    void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);
    void mask_gpu(int N, float * X, float mask_num, float * mask, float val);
    void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale);
    void const_gpu(int N, float ALPHA, float *X, int INCX);
    void add_gpu(int N, float ALPHA, float * X, int INCX);
    void scal_cpu(int N, float ALPHA, float *X, int INCX);
    void fill_gpu(int N, float ALPHA, float * X, int INCX);
    void mult_add_into_gpu(int num, float *a, float *b, float *c);
    #endif
}

#endif

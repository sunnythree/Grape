#ifndef __GRAPE_BLAS_H__
#define __GRAPE_BLAS_H__


namespace Grape
{
    extern "C"{
    void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);
    void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
    void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
    void scal_cpu(int N, float ALPHA, float *X, int INCX);
    void fill_cpu(int N, float ALPHA, float * X, int INCX);
    void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
    void mult_add_into_cpu(int N, float *X, float *Y, float *Z);
    void add_cpu(int N, float *X, int INCX, float *Y, int INCY);
    void softmax(float *input, int n, float temp, int stride, float *output);
    void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups,
                     int group_offset, int stride, float temp, float *output);
    void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
    void add_bias(float *output, float *biases, int batch, int n, int size);
    void scale_bias(float *output, float *scales, int batch, int n, int size);
    void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

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
    void scal_gpu(int N, float ALPHA, float *X, int INCX);
    void fill_gpu(int N, float ALPHA, float * X, int INCX);
    void mult_add_into_gpu(int num, float *a, float *b, float *c);
    void softmax_gpu(float *input, int n, float temp, int stride, float *output);
    void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);

    void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
    void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
    void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
    void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
    #endif
    }
}

#endif

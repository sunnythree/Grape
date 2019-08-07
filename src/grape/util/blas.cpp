#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "grape/util/blas.h"
#include "grape/util/util.h"




namespace Grape
{
    void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
    {
        int i;
        for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
    }

    void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
    {
        int i;
        for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
    }

    void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
    {
        int i;
        for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
    }

    void scal_cpu(int N, float ALPHA, float *X, int INCX)
    {
        int i;
        for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
    }

    void fill_cpu(int N, float ALPHA, float *X, int INCX)
    {
        int i;
        for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
    }

    void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
    {
        int i;
        for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
    }

    void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
    {
        int i;
        for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
    }

    void add_cpu(int N, float *X, int INCX, float *Y, int INCY)
    {
        int i;
        for(i = 0; i < N; ++i) Y[i*INCY] += X[i*INCX];
    }

    void softmax(float *input, int n, float temp, int stride, float *output)
    {
        int i;
        float sum = 0;
        float largest = -FLT_MAX;
        for(i = 0; i < n; ++i){
            if(input[i*stride] > largest) largest = input[i*stride];
        }
        for(i = 0; i < n; ++i){
            float e = exp(input[i*stride]/temp - largest/temp);
            sum += e;
            output[i*stride] = e;
        }
        for(i = 0; i < n; ++i){
            output[i*stride] /= sum;
        }
    }
    

   void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
    {
        int i;
        for(i = 0; i < n; ++i){
            float t = truth[i];
            float p = pred[i];
            error[i] = (t>0)/*&&(p!=0)*/ ? -log(p) : 0;
            delta[i] = t-p;
            // if((t>0)&&(p!=0)){
            //     printf("t=%f,p=%f\n",t,p);
            // }
        }
    }


    void add_bias(float *output, float *biases, int batch, int n, int size)
    {
        int i,j,b;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < n; ++i){
                for(j = 0; j < size; ++j){
                    output[(b*n + i)*size + j] += biases[i];
                }
            }
        }
    }

    void scale_bias(float *output, float *scales, int batch, int n, int size)
    {
        int i,j,b;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < n; ++i){
                for(j = 0; j < size; ++j){
                    output[(b*n + i)*size + j] *= scales[i];
                }
            }
        }
    }

    void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
    {
        int i,b;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < n; ++i){
                bias_updates[i] += sum_array(delta+size*(i+b*n), size);
            }
        }
    }

} // namespace Grape






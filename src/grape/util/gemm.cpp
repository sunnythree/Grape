#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include "grape/util/gemm.h"
#include "grape/util/cuda.h"
#include "grape/log.h"

static std::string TAG = "gemm";

namespace Grape
{

    void gemm_bin(int M, int N, int K, float ALPHA, 
            char  *A, int lda, 
            float *B, int ldb,
            float *C, int ldc)
    {
        int i,j,k;
        for(i = 0; i < M; ++i){
            for(k = 0; k < K; ++k){
                char A_PART = A[i*lda+k];
                if(A_PART){
                    for(j = 0; j < N; ++j){
                        C[i*ldc+j] += B[k*ldb+j];
                    }
                } else {
                    for(j = 0; j < N; ++j){
                        C[i*ldc+j] -= B[k*ldb+j];
                    }
                }
            }
        }
    }

    float *random_matrix(int rows, int cols)
    {
        int i;
        float *m = (float *) calloc(rows*cols, sizeof(float));
        for(i = 0; i < rows*cols; ++i){
            m[i] = (float)rand()/RAND_MAX;
        }
        return m;
    }

    void time_random_matrix(int TA, int TB, int m, int k, int n)
    {
        float *a;
        if(!TA) a = random_matrix(m,k);
        else a = random_matrix(k,m);
        int lda = (!TA)?k:m;
        float *b;
        if(!TB) b = random_matrix(k,n);
        else b = random_matrix(n,k);
        int ldb = (!TB)?n:k;

        float *c = random_matrix(m,n);
        for(int i = 0; i<10; ++i){
            gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
        }
        free(b);
        free(c);
    }


    void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
            float *A, int lda, 
            float *B, int ldb,
            float BETA,
            float *C, int ldc)
    {
        gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
    }

    void gemm_nn(int M, int N, int K, float ALPHA, 
            float *A, int lda, 
            float *B, int ldb,
            float *C, int ldc)
    {
        int i,j,k;
        #pragma omp parallel for
        for(i = 0; i < M; ++i){
            for(k = 0; k < K; ++k){
                register float A_PART = ALPHA*A[i*lda+k];
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += A_PART*B[k*ldb+j];
                }
            }
        }
    }

    void gemm_nt(int M, int N, int K, float ALPHA, 
            float *A, int lda, 
            float *B, int ldb,
            float *C, int ldc)
    {
        int i,j,k;
        #pragma omp parallel for
        for(i = 0; i < M; ++i){
            for(j = 0; j < N; ++j){
                register float sum = 0;
                for(k = 0; k < K; ++k){
                    sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
                }
                C[i*ldc+j] += sum;
            }
        }
    }

    void gemm_tn(int M, int N, int K, float ALPHA, 
            float *A, int lda, 
            float *B, int ldb,
            float *C, int ldc)
    {
        int i,j,k;
        #pragma omp parallel for
        for(i = 0; i < M; ++i){
            for(k = 0; k < K; ++k){
                register float A_PART = ALPHA*A[k*lda+i];
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += A_PART*B[k*ldb+j];
                }
            }
        }
    }

    void gemm_tt(int M, int N, int K, float ALPHA, 
            float *A, int lda, 
            float *B, int ldb,
            float *C, int ldc)
    {
        int i,j,k;
        #pragma omp parallel for
        for(i = 0; i < M; ++i){
            for(j = 0; j < N; ++j){
                register float sum = 0;
                for(k = 0; k < K; ++k){
                    sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
                }
                C[i*ldc+j] += sum;
            }
        }
    }


    void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
            float *A, int lda, 
            float *B, int ldb,
            float BETA,
            float *C, int ldc)
    {
        for(int i = 0; i < M; ++i){
            for(int j = 0; j < N; ++j){
                C[i*ldc + j] *= BETA;
            }
        }
        if(!TA && !TB)
            gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
        else if(TA && !TB)
            gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
        else if(!TA && TB)
            gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
        else
            gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    }

    #ifdef GPU

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
            float *A_gpu, int lda, 
            float *B_gpu, int ldb,
            float BETA,
            float *C_gpu, int ldc)
    {
        cublasHandle_t handle = cuda_blas_handle();
        cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
                (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
        if(status != CUBLAS_STATUS_SUCCESS){
            //Log::e(TAG,"cublasSgemm errors");
        }
    }
    #endif
} // namespace Grape



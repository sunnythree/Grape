#ifndef __GRAPE_POOL_H__
#define __GRAPE_POOL_H__

namespace Grape
{
    void forward_maxpool_cpu(int batch,int in_w,int in_h, int out_w,int out_h,int c,int stride, 
        int size, int pad, float *in,float *out, int *indexes);
    void backward_maxpool_cpu(int n,float *in_diff,float *out_diff, int *indexes);
    void forward_avgpool_cpu(int batch, int w, int h, int c, float *in, float *out);
    void backward_avgpool_cpu(int batch, int w, int h, int c, float *in_diff, float *out_diff);
#ifdef GPU
    void forward_maxpool_gpu(int n,int w,int h,int c,int stride, 
        int size, int pad, float *in,float *out, int *indexes);
    void backward_maxpool_gpu(int n,int w,int h,int c,int stride, 
        int size, int pad, float *in_diff,float *out_diff, int *indexes);
    void forward_avgpool_gpu(int n, int w, int h, int c, float *in, float *out);
    void backward_avgpool_gpu(int n, int w, int h, int c, float *in_diff, float *out_diff);
#endif
}

#endif

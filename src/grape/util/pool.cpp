#include <stdio.h>
#include <float.h>
#include "grape/util/pool.h"

namespace Grape
{
    void forward_maxpool_cpu(int batch,int in_w,int in_h, int out_w,int out_h,int c,int stride, 
        int size, int pad, float *in,float *out, int *indexes)
    {
        int w_offset = -pad/2;
        int h_offset = -pad/2;
        for(int b = 0; b < batch; ++b){
            for(int k = 0; k < c; ++k){
                for(int i = 0; i < out_h; ++i){
                    for(int j = 0; j < out_w; ++j){
                        int out_index = j + out_w*(i + out_h*(k + c*b));
                        float max = -FLT_MAX;
                        int max_i = -1;
                        for(int n = 0; n < size; ++n){
                            for(int m = 0; m < size; ++m){
                                int cur_h = h_offset + i*stride + n;
                                int cur_w = w_offset + j*stride + m;
                                int index = cur_w + in_w*(cur_h + in_h*(k + b*c));
                                int valid = (cur_h >= 0 && cur_h < in_h &&
                                            cur_w >= 0 && cur_w < in_w);
                                float val = (valid != 0) ? in[index] : -FLT_MAX;
                                max_i = (val > max) ? index : max_i;
                                max   = (val > max) ? val   : max;
                            }
                        }
                        out[out_index] = max;
                        indexes[out_index] = max_i;
                    }
                }
            }
        }
    }

    void backward_maxpool_cpu(int n,float *in_diff,float *out_diff, int *indexes)
    {
        for(int i = 0; i < n; ++i){
            int index = indexes[i];
            out_diff[index] = in_diff[i];
        }
    }

    void forward_avgpool_cpu(int batch, int w, int h, int c, float *in, float *out)
    {
        for(int b = 0; b < batch; ++b){
            for(int k = 0; k < c; ++k){
                int out_index = k + b*c;
                out[out_index] = 0;
                for(int i = 0; i < w*h; ++i){
                    int in_index = i + w*h*(k + b*c);
                    out[out_index] += in[in_index];
                }
                out[out_index] /= w*h;
            }
        }
    }

    void backward_avgpool_cpu(int batch, int w, int h, int c, float *in_diff, float *out_diff)
    {
        for(int b = 0; b < batch; ++b){
            for(int k = 0; k < c; ++k){
                int out_index = k + b*c;
                for(int i = 0; i < w*h; ++i){
                    int in_index = i + w*h*(k + b*c);
                    out_diff[in_index] += in_diff[out_index] / (w*h);
                }
            }
        }
    }

} // namespace Grape




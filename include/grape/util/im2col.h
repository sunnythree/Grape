#ifndef __GRAPE_IM2COL_H__
#define __GRAPE_IM2COL_H__

namespace Grape
{
    extern "C"{
    void im2col_cpu(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col);

    #ifdef GPU

    void im2col_gpu(float *im,
            int channels, int height, int width,
            int ksize, int stride, int pad,float *data_col);

    #endif
    }
} // namespace Grape


#endif

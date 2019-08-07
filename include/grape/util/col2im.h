#ifndef __GRAPE_COL2IM_H__
#define __GRAPE_COL2IM_H__


namespace Grape
{
    extern "C"{

    void col2im_cpu(float* data_col,
            int channels, int height, int width,
            int ksize, int stride, int pad, float* data_im);

    #ifdef GPU
    void col2im_gpu(float *data_col,
            int channels, int height, int width,
            int ksize, int stride, int pad, float *data_im);
    #endif
            
    }
} // namespace Grape


#endif

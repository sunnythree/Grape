#ifndef COL2IM_H
#define COL2IM_H


namespace javernn
{
    void col2im_cpu(float* data_col,
            int channels, int height, int width,
            int ksize, int stride, int pad, float* data_im);

    #ifdef GPU
    void col2im_gpu(float *data_col,
            int channels, int height, int width,
            int ksize, int stride, int pad, float *data_im);
    #endif
} // namespace javernn


#endif

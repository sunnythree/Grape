#ifndef __GRAPE_DROPOUT_UTIL_H__
#define __GRAPE_DROPOUT_UTIL_H__

namespace Grape
{
    void forward_dropout_cpu(int batch, int in_dim, float *input, float *output, float *rand_data, float probability, float scale);
    void backward_dropout_cpu(int batch, int in_dim, float *input, float *output, float *rand_data, float probability, float scale);
#ifdef GPU
    void forward_dropout_gpu(int batch, int in_dim, float *input, float *output, float *rand_data, float probability, float scale);
    void backward_dropout_gpu(int batch, int in_dim, float *input, float *output, float *rand_data, float probability, float scale);
#endif
}

#endif

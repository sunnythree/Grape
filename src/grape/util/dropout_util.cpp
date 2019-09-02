#include "grape/util/dropout_util.h"
#include "grape/util/random.h"

namespace Grape
{
    void forward_dropout_cpu(int batch,int in_dim,float * input, float *rand_data,float probability, float scale)
    {
        for(int i = 0; i < batch * in_dim; ++i){
            float r = Random::GetInstance().GetUniformFloat(0.,1.);
            rand_data[i] = r;
            if(r < probability) input[i] = 0;
            else input[i] *= scale;
        }
    }
    void backward_dropout_cpu(int batch, int in_dim, float * diff, float *rand_data, float probability, float scale)
    {
        for(int i = 0; i < batch * in_dim; ++i){
            float r = rand_data[i];
            if(r < probability) diff[i] = 0;
            else diff[i] *= scale;
        }
    }

} // namespace Grape
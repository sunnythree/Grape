#include "javernn/optimizer/sgd.h"

namespace javernn
{
    SGD::SGD(float lr):alpha_(float(0.01)), lambda_(float(0))
    {

    }

    SGD::~SGD()
    {

    }

    void SGD::reset() 
    {

    }

    void SGD::UpdateCpu(Tensor *weights)
    {
        float *W = (float *)weights->mutable_cpu_data();
        float *dW = (float *)weights->mutable_cpu_diff(); 
        for(int i=0;i<weights->shape().count();i++){
            W[i] = W[i] - alpha_ * (dW[i] + lambda_ * W[i]);
        }   
    }

#ifdef GPU
    void SGD::UpdateGpu(Tensor *weights)
    {

    }
#endif
} // namespace javernn



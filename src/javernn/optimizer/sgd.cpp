#include "javernn/optimizer/sgd.h"

namespace javernn
{
    SGDOptimizer::SGDOptimizer(float lr):alpha_(float(0.01)), lambda_(float(0))
    {

    }

    SGDOptimizer::~SGDOptimizer()
    {

    }

    void SGDOptimizer::reset() 
    {

    }

    void SGDOptimizer::UpdateCpu(Tensor *weights)
    {
        float *W = (float *)weights->mutable_cpu_data();
        float *dW = (float *)weights->mutable_cpu_diff(); 
        for(int i=0;i<weights->shape().count();i++){
            W[i] = W[i] - alpha_ * (dW[i] + lambda_ * W[i]);
        }   
    }

#ifdef GPU
    void SGDOptimizer::UpdateGpu(Tensor *weights)
    {

    }
#endif
} // namespace javernn



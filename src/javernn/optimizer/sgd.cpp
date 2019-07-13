#include "javernn/optimizer/sgd.h"
#include "javernn/util/blas.h"

namespace javernn
{
    SGDOptimizer::SGDOptimizer(float lr):alpha_(lr), lambda_(float(0))
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
        uint32_t n = weights->shape().count(); 
        if(weights->vtype() == WEIGHTS){
            axpy_cpu(n, -lambda_, W, 1, dW, 1);
            axpy_cpu(n, alpha_, dW, 1, W, 1);
            fill_cpu(n,0,dW,1);
        }else{
            axpy_cpu(n, alpha_, dW, 1, W, 1);
            fill_cpu(n,0,dW,1);
        }
        update_count_++;
        // if(update_count_%200==0){
        //     alpha_*=0.8;
        // }
    }

#ifdef GPU
    void SGDOptimizer::UpdateGpu(Tensor *weights)
    {

    }
#endif
} // namespace javernn



#ifndef __JAVERNN_OPTIMIZER_H__
#define __JAVERNN_OPTIMIZER_H__
#include <unordered_map>
#include "javernn/tensor.h"
namespace javernn
{
    /**
     * base class of Optimizer
     * usesHessian : true if an Optimizer uses hessian (2nd order derivative of loss
     *function)
    **/
    class Optimizer {
        Optimizer()                  = default;
        Optimizer(const Optimizer &) = default;
        Optimizer(Optimizer &&)      = default;
        Optimizer &operator=(const Optimizer &) = default;
        Optimizer &operator=(Optimizer &&) = default;
        virtual ~Optimizer()               = default;
        virtual void reset() {}  // override to implement pre-learning action
        virtual void UpdateCpu(const Tensor &dW, Tensor &W) = 0;
    #ifdef GPU
        virtual void UpdateGpu(const Tensor &dW, Tensor &W) = 0;
    #endif
    };
}

#endif

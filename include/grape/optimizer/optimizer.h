#ifndef __GRAPE_OPTIMIZER_H__
#define __GRAPE_OPTIMIZER_H__
#include <unordered_map>
#include "grape/tensor.h"
namespace Grape
{
    enum LR_POLICY{
        POLICY_FIXED,
        POLICY_STEP,
        POLICY_MUTISTEP
    };
    /**
     * base class of Optimizer
     * usesHessian : true if an Optimizer uses hessian (2nd order derivative of loss
     *function)
    **/
    class Optimizer {
    public:
        Optimizer()                  = default;
        Optimizer(const Optimizer &) = default;
        Optimizer(Optimizer &&)      = default;
        Optimizer &operator=(const Optimizer &) = default;
        Optimizer &operator=(Optimizer &&) = default;
        virtual ~Optimizer()               = default;
        virtual void reset() = 0; // override to implement pre-learning action
        virtual void UpdateCpu(Tensor *weights, uint32_t batch) = 0;
    #ifdef GPU
        virtual void UpdateGpu(Tensor *weights, uint32_t batch) = 0;
    #endif
        virtual void CheckLrUpdate(uint32_t iter_cout){};
    };
}

#endif

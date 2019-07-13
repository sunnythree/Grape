
#ifndef __JAVERNN_SGD_H__
#define __JAVERNN_SGD_H__

#include "javernn/optimizer/optimizer.h"
#include "javernn/tensor.h"
namespace javernn
{
    class SGDOptimizer:public Optimizer{
    public:
        SGDOptimizer(float lr);
        virtual ~SGDOptimizer();
        virtual void reset(); // override to implement pre-learning action
        virtual void UpdateCpu(Tensor *weights);
    #ifdef GPU
        virtual void UpdateGpu(Tensor *weights);
    #endif
    private:
    float alpha_ = 0;   // learning rate
    float lambda_ = 0;  // weight decay
    uint32_t update_count_ = 0;
    };
} // namespace javernn

#endif
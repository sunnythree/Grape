
#ifndef __JAVERNN_SGD_H__
#define __JAVERNN_SGD_H__

#include "javernn/optimizer/optimizer.h"
#include "javernn/tensor.h"
namespace javernn
{
    class SGD:public Optimizer{
    public:
        SGD(float lr);
        virtual ~SGD();
        virtual void reset(); // override to implement pre-learning action
        virtual void UpdateCpu(Tensor *weights) = 0;
    #ifdef GPU
        virtual void UpdateGpu(Tensor *weights) = 0;
    #endif
    private:
    float alpha_;   // learning rate
    float lambda_;  // weight decay
    };
} // namespace javernn

#endif
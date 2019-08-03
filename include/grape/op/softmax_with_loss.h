#ifndef __GRAPE_SOFTMAX_WITH_LOSS_H__
#define __GRAPE_SOFTMAX_WITH_LOSS_H__

#include "grape/op.h"

namespace Grape{
    class SoftmaxWithLoss :public Op{
    public:
        SoftmaxWithLoss(std::string name, uint32_t batch_size, uint32_t in_dim);
        virtual ~SoftmaxWithLoss();
        virtual void Setup();
        virtual void ForwardCpu(); 
        virtual void BackwardCpu();
        virtual void UpdateWeightsCpu(Optimizer &opt);
        virtual void Display();

#ifdef GPU
        virtual void ForwardGpu(); 
        virtual void BackwardGpu();
        virtual void UpdateWeightsGpu(Optimizer &opt);
#endif
    private:
        uint32_t batch_size_;
        uint32_t in_dim_;
        uint32_t temperature_;
        std::shared_ptr<Tensor> cost_;
    };
}
#endif
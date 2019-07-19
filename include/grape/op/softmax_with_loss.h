#ifndef __GRAPE_SOFTMAX_WITH_LOSS_H__
#define __GRAPE_SOFTMAX_WITH_LOSS_H__

#include "grape/op.h"

namespace Grape{
    class SoftmaxWithLoss :public Op{
    public:
        SoftmaxWithLoss(std::string name, uint32_t batch_size, uint32_t in_dim);
        ~SoftmaxWithLoss();
        void Setup();
        void ForwardCpu(); 
        void BackwardCpu();
        void UpdateWeightsCpu(Optimizer &opt);

#ifdef GPU
        void ForwardGpu(); 
        void BackwardGpu();
        void UpdateWeightsGpu(Optimizer &opt);
#endif
    private:
        uint32_t batch_size_;
        uint32_t in_dim_;
        uint32_t temperature_;
        std::shared_ptr<Tensor> cost_;
    };
}
#endif
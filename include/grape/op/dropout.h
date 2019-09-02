#ifndef __GRAPE_DROPOUT_H__
#define __GRAPE_DROPOUT_H__

#include "grape/op.h"

namespace Grape{
    class Dropout : public Op{
    public:
        Dropout() = delete;
        Dropout(std::string name, uint32_t batch_size, uint32_t in_dim, float probability);
        ~Dropout();
        virtual void Setup();
        virtual void ForwardCpu(); 
        virtual void BackwardCpu();
        virtual void UpdateWeightsCpu(Optimizer &opt);
        virtual void OnTestBegin();
        virtual void OnTrainBegin();

#ifdef GPU
        virtual void ForwardGpu(); 
        virtual void BackwardGpu();
        virtual void UpdateWeightsGpu(Optimizer &opt);
#endif

    private:
        uint32_t batch_size_ = 0;
        uint32_t in_dim_ = 0;
        bool is_train_ = false;
        float probability_ = 0;
        float scale_ = 0;
        std::shared_ptr<Tensor> rand_;
    };
}
#endif
#ifndef __GRAPE_BATCH_NORM_H__
#define __GRAPE_BATCH_NORM_H__

#include "grape/op.h"

namespace Grape
{
    class BatchNorm : public Op
    {
    public:
        BatchNorm(
            std::string name,
            uint32_t batch_size_,
            uint32_t in_c_,
            uint32_t in_h_,
            uint32_t in_w_
        );
        virtual ~BatchNorm();

        virtual void Setup();
    
        virtual void ForwardCpu(); 
        virtual void BackwardCpu();
        virtual void UpdateWeightsCpu(Optimizer &opt);

#ifdef GPU
        virtual void ForwardGpu(); 
        virtual void BackwardGpu();
        virtual void UpdateWeightsGpu(Optimizer &opt);
#endif
    private:
        uint32_t batch_size_;
        uint32_t in_c_;
        uint32_t in_h_;
        uint32_t in_w_;
    };
    
} // namespace Grape

#endif
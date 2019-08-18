#ifndef __GRAPE_L2NORM_H__
#define __GRAPE_L2NORM_H__

#include "grape/op.h"

namespace Grape
{
    class L2Norm : public Op
    {
    public:
        L2Norm(
            std::string name,
            uint32_t batch_size_,
            uint32_t inputs_
        );
        virtual ~L2Norm();

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
        uint32_t inputs_;
    };
    
} // namespace Grape

#endif
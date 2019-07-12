#ifndef __JAVERNN_OUTPUT_H__
#define __JAVERNN_OUTPUT_H__

#include "javernn/op.h"

namespace javernn
{
    class Output: public Op
    {
    public:
        Output(uint32_t batch_size, uint32_t in_dim);
        virtual ~Output();

        virtual void Setup();
    
        virtual void ForwardCpu(); 
        virtual void BackwardCpu();
        virtual void UpdateWeightsCpu(Optimizer &opt);

#ifdef GPU
        virtual void ForwardGpu(); 
        virtual void BackwardGpu();
        virtual void UpdateWeightsGpu(Optimizer &opt);
#endif
        virtual Tensor* GetLossTensor();
        virtual float GetLoss();
        
    protected:
        uint32_t batch_size_;
        uint32_t out_dim_;
    };
    
} // namespace javernn

#endif
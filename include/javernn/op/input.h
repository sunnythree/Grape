#ifndef __JAVERNN_INPUT_H__
#define __JAVERNN_INPUT_H__

#include "javernn/op.h"

namespace javernn
{
    class Input: public Op
    {
    public:
        Input(uint32_t batch_size, uint32_t data_dim_, uint32_t label_dim_);
        virtual ~Input();

        virtual void Setup();
    
        virtual void ForwardCpu(); 
        virtual void BackwardCpu();
        virtual void UpdateWeightsCpu(Optimizer &opt);

#ifdef GPU
        virtual void ForwardGpu(); 
        virtual void BackwardGpu();
        virtual void UpdateWeightsGpu(Optimizer &opt);
#endif
        virtual std::vector<tensorptr_t> GetOutputTensor();
    protected:
        uint32_t batch_size_;
        uint32_t data_dim_;
        uint32_t label_dim_;
    };
    
} // namespace javernn

#endif
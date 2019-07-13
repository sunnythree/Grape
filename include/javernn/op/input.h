#ifndef __JAVERNN_INPUT_H__
#define __JAVERNN_INPUT_H__

#include "javernn/op.h"

namespace javernn
{
    class Input: public Op
    {
    public:
        Input(std::string name, uint32_t batch_size, uint32_t out_dim);
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
        virtual Tensor* GetOutputTensor();
    protected:
        uint32_t batch_size_;
        uint32_t out_dim_;
    };
    
} // namespace javernn

#endif
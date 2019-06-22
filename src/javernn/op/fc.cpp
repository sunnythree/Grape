#include "javernn/op/fc.h"


namespace javernn{
    Fc::Fc(uint32_t in_dim,uint32_t out_dim,bool has_bias)
    {

    }

    Fc::~Fc()
    {

    }

    void SetupCpu(bool reset_weight)
    {

    }

    std::vector<Tensor> ForwardCpu()
    {
        std::vector<Tensor> out;
        return out;
    } 

    void BackwardCpu()
    {

    }

    void UpdateWeightsCpu(Optimizer *opt)
    {

    }

#ifdef GPU
    void SetupGpu(bool reset_weight)
    {

    }

    std::vector<Tensor> ForwardGpu()
    {
        std::vector<Tensor> out;
        return out;
    }

    void BackwardGpu()
    {

    }

    void UpdateWeightsGpu(Optimizer *opt)
    {

    }
#endif
}
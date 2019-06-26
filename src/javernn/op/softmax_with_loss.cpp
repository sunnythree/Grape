#include "javernn/op/softmax_with_loss.h"



namespace javernn
{
    void SoftmaxWithLoss::Setup()
    {

    }

    void SoftmaxWithLoss::ForwardCpu()
    {

    }

    void SoftmaxWithLoss::BackwardCpu()
    {

    }

    void SoftmaxWithLoss::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void SoftmaxWithLoss::ForwardGpu()
    {

    } 

    void SoftmaxWithLoss::BackwardGpu()
    {

    }

    void SoftmaxWithLoss::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
} // namespace javernn

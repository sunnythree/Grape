#ifndef __JAVERNN_SOFTMAX_WITH_LOSS_H__
#define __JAVERNN_SOFTMAX_WITH_LOSS_H__

#include "javernn/op.h"

namespace javernn{
    class SoftmaxWithLoss : Op{
        void Setup();
        void ForwardCpu(); 
        void BackwardCpu();
        void UpdateWeightsCpu(Optimizer &opt);

#ifdef GPU
        void ForwardGpu(); 
        void BackwardGpu();
        void UpdateWeightsGpu(Optimizer &opt);
#endif
    };
}
#endif
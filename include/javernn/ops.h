#ifndef __javernn_nodes_h__
#define __javernn_nodes_h__
#include <vector>
#include "javernn/op.h"

namespace javernn{
    class Ops{
    public:
        virtual ~Ops(){};
        virtual void BackwardCpu() = 0;
        virtual void BackwardGpu() = 0;
        virtual std::vector<Tensor> ForwardCpu() = 0;  // NOLINT
        virtual std::vector<Tensor> ForwardGpu() = 0;  // NOLINT
        virtual void UpdateWeights(Optimizer *opt) = 0;
        virtual void Setup(bool reset_weight)= 0;

    };
}

#endif
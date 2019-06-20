#ifndef __javernn_nodes_h__
#define __javernn_nodes_h__
#include <vector>
#include "javernn/op.h"

namespace javernn{
    class Ops{
    public:
        virtual void Backward() = 0;
        virtual std::vector<Tensor> Forward() = 0;  // NOLINT
        virtual void UpdateWeights(Optimizer *opt) = 0;
        virtual void Setup(bool reset_weight)= 0;

    };
}

#endif
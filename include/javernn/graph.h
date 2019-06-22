#ifndef __JAVERNN_GRAPH_H__
#define __JAVERNN_GRAPH_H__

#include "javernn/ops.h"

namespace javernn{
    class Graph:public Ops{
    public:
        void Backward(const std::vector<Tensor> &cost);
        std::vector<Tensor> Forward(const std::vector<Tensor> &inputs);  
        void UpdateWeights(Optimizer &opt);
        void Setup(bool reset_weight);
        void Construct(const std::vector<Op *> &input,
                 const std::vector<Op *> &output);
        int32_t FindIndex(const std::vector<Op *> &ops, Op *target);
    private:
        std::vector<Op *> ops_;
        std::vector<Op *> input_layers_;
        std::vector<Op *> output_layers_;
    };
}

#endif
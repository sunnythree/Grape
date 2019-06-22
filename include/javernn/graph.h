#ifndef __javernn_graph_h__
#define __javernn_graph_h__

#include "javernn/ops.h"

namespace javernn{
    class Graph:public Ops{
    public:
        void Backward();
        std::vector<Tensor> Forward();  
        void UpdateWeights(Optimizer *opt);
        void Setup(bool reset_weight);
        void Construct(const std::vector<Op *> &input,
                 const std::vector<Op *> &output);
        size_t FindIndex(const std::vector<Op *> &ops, Op *target);
    private:
        std::vector<Op *> ops_;
        std::vector<Op *> input_layers_;
        std::vector<Op *> output_layers_;
    };
}

#endif
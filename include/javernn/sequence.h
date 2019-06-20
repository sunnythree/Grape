#ifndef __javernn_sequence_h__
#define __javernn_sequence_h__

#include "javernn/ops.h"

namespace javernn{
    class Sequence:public Ops{
    public:
        void Backward();
        std::vector<Tensor> Forward();  
        void UpdateWeights(Optimizer *opt);
        void Setup(bool reset_weight);
        void Add(Op* op);
    private:
        std::vector<Op *> ops_;
    };
}

#endif
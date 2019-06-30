#ifndef __JAVERNN_SEQUENCE_H__
#define __JAVERNN_SEQUENCE_H__

#include "javernn/ops.h"
#include "javernn/op/input.h"

namespace javernn{
    class Sequence:public Ops{
    public:
        void Backward();
        void Forward();  
        void UpdateWeights(Optimizer &opt);
        void Setup();
        void Add(Op* op);
        void Construct();
    private:
        std::vector<Op *> ops_;
        std::shared_ptr<Input> label_;
    };
}

#endif
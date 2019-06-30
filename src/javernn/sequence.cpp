#include <memory>
#include "javernn/sequence.h"
#include "javernn/log.h"

namespace javernn{
    static std::string TAG = "Sequence";
    void Sequence::Backward()
    {
        for(int i=ops_.size()-1;i>=0;--i){
            if(gNetMode == CPU_MODE){
                ops_[i]->BackwardCpu();
            }else{
#ifdef GPU
                ops_[i]->BackwardGpu();
#endif
            }
        }
    }
    void Sequence::Forward()
    {

        for(auto o:ops_){
            if(gNetMode == CPU_MODE){
                o->ForwardCpu();
            }else{
#ifdef GPU
                o->ForwardGpu();
#endif
            }
        }
    } 
    void Sequence::UpdateWeights(Optimizer &opt)
    {
        for(auto o:ops_){
            if(gNetMode == CPU_MODE){
                o->UpdateWeightsCpu(opt);
            }else{
#ifdef GPU
                o->UpdateWeightsGpu(opt);
#endif
            }
        }
    }
    void Sequence::Setup()
    {
        for(auto o:ops_){
             o->Setup();
        }
        auto out_tensors = ops_[ops_.size()-1]->prev();
        Shape out_shape = out_tensors[0]->shape();
        label_ = std::make_shared<Input>(out_shape.index(0), out_shape.count(1));
        connect_op(label_.get(), ops_[ops_.size()-1], 0, 1);
    }
    void Sequence::Add(Op* op)
    {
        ops_.push_back(op);
        if (ops_.size() != 1) {
            auto head = ops_[ops_.size() - 2];
            auto tail = ops_[ops_.size() - 1];
            connect_op(head, tail, 0, 0);
        }
    }

    void Sequence::Construct()
    {
        Setup();
    }
}

#include "javernn/sequence.h"

namespace javernn{
    void Sequence::Backward()
    {
        for(auto o:ops_){
            o->Backward();
        }
    }
    std::vector<Tensor> Sequence::Forward()
    {
        for(auto o:ops_){
            o->Forward();
        }
    } 
    void Sequence::UpdateWeights(Optimizer *opt)
    {
        for(auto o:ops_){
            o->UpdateWeights(opt);
        }
    }
    void Sequence::Setup(bool reset_weight)
    {
        for(auto o:ops_){
            o->Setup(reset_weight);
        }
    }
    void Sequence::Add(Op* op)
    {
        ops_.push_back(op);
    }
}

#include "javernn/sequence.h"

namespace javernn{
    void Sequence::Backward(const std::vector<Tensor> &cost)
    {
        for(auto o:ops_){
            if(gNetMode == CPU_MODE){
                o->BackwardCpu();
            }else{
#ifdef GPU
                o->BackwardGpu();
#endif
            }
        }
    }
    std::vector<Tensor> Sequence::Forward(const std::vector<Tensor> &inputs)
    {
        std::vector<Tensor> cost;
        for(auto o:ops_){
            if(gNetMode == CPU_MODE){
                o->ForwardCpu();
            }else{
#ifdef GPU
                o->ForwardGpu();
#endif
            }
        }
        return cost;
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
    }
    void Sequence::Add(Op* op)
    {
        ops_.push_back(op);
    }

    void Sequence::Construct()
    {
        Setup();
    }
}

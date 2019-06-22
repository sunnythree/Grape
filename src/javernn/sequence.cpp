#include "javernn/sequence.h"

namespace javernn{
    void Sequence::Backward()
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
    std::vector<Tensor> Sequence::Forward()
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
    void Sequence::UpdateWeights(Optimizer *opt)
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
    void Sequence::Setup(bool reset_weight)
    {
        for(auto o:ops_){
            if(gNetMode == CPU_MODE){
                o->SetupCpu(reset_weight);
            }else{
#ifdef GPU
                o->SetupCpu(reset_weight);
#endif
            }
        }
    }
    void Sequence::Add(Op* op)
    {
        ops_.push_back(op);
    }
}

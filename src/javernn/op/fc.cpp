#include "javernn/op/fc.h"
#include "javernn/log.h"
#include <memory>

namespace javernn{
    static std::string TAG = "Fc";
    Fc::Fc(uint32_t batch_size,uint32_t in_dim,uint32_t out_dim,bool has_bias)
    : Op({DATA,WEIGHTS,BIAS}, {DATA}),
    batch_size_(batch_size),
    in_dim_(in_dim),
    out_dim_(out_dim),
    has_bias_(has_bias)
    {
        //create input tensor,only weights and bias
        for(int i=0;i<prev_.size();i++){
            if(in_type_[i] == WEIGHTS){
                prev_[i] = std::make_shared<Tensor>(static_cast<Op *>(this),
                Shape({in_dim_,out_dim_}),DATA,gNetMode);
            }else if(in_type_[i] == BIAS && has_bias_){
                prev_[i] = std::make_shared<Tensor>(static_cast<Op *>(this),
                Shape({out_dim_}),DATA,gNetMode);
            }
        }
        //create output tensor,only data
        for(int i=0;i<next_.size();i++){
            if(out_type_[i] == DATA){
                next_[i] = std::make_shared<Tensor>(static_cast<Op *>(this),
                Shape({batch_size_,out_dim_}),DATA,gNetMode);
            }
        }
    }

    Fc::~Fc()
    {

    }

    void Fc::SetupCpu(bool reset_weight)
    {
        
    }

    std::vector<Tensor> Fc::ForwardCpu()
    {
        std::vector<Tensor> out;
        Log::v(TAG," Fc ForwardCpu");
        return out;
    } 

    void Fc::BackwardCpu()
    {
        Log::v(TAG," Fc BackwardCpu");
    }

    void Fc::UpdateWeightsCpu(Optimizer &opt)
    {
        Log::v(TAG," Fc UpdateWeightsCpu");
    }

#ifdef GPU
    void Fc::SetupGpu(bool reset_weight)
    {

    }

    std::vector<Tensor> Fc::ForwardGpu()
    {
        std::vector<Tensor> out;
        return out;
    }

    void Fc::BackwardGpu()
    {

    }

    void Fc::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
}
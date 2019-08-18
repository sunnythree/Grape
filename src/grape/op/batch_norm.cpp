#include <string>
#include "grape/op/batch_norm.h"
#include "grape/log.h"
#include "grape/global_config.h"

namespace Grape
{
    const static std::string TAG = "BatchNorm";
    

    BatchNorm::BatchNorm(std::string name, uint32_t batch_size, uint32_t in_c, uint32_t in_h, uint32_t in_w):
    Op({},{DATA}),
    batch_size_(batch_size),
    in_c_(in_c),
    in_h_(in_h),
    in_w_(in_w_)
    {
        type_ = STRING_INPUT_TYPE;
        name_ = name;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({batch_size_,in_c_,in_h_,in_w_}),DATA,sizeof(float));
    }
    
    BatchNorm::~BatchNorm()
    {
    }

    void BatchNorm::Setup()
    {
    }

    void BatchNorm::ForwardCpu()
    {
    } 

    void BatchNorm::BackwardCpu()
    {
    }

    void BatchNorm::UpdateWeightsCpu(Optimizer &opt)
    {

    }


#ifdef GPU
    void BatchNorm::ForwardGpu()
    {

    } 

    void BatchNorm::BackwardGpu()
    {

    }

    void BatchNorm::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
}

#include <string>
#include "grape/op/l2_norm.h"
#include "grape/log.h"
#include "grape/global_config.h"

namespace Grape
{
    const static std::string TAG = "L2Norm";
    

    L2Norm::L2Norm(std::string name, uint32_t batch_size, uint32_t inputs):
    Op({},{DATA}),
    batch_size_(batch_size),
    inputs_(inputs)
    {
        type_ = STRING_INPUT_TYPE;
        name_ = name;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({batch_size_,inputs}),DATA,sizeof(float));
    }
    
    L2Norm::~L2Norm()
    {
    }

    void L2Norm::Setup()
    {
    }

    void L2Norm::ForwardCpu()
    {
    } 

    void L2Norm::BackwardCpu()
    {
    }

    void L2Norm::UpdateWeightsCpu(Optimizer &opt)
    {

    }


#ifdef GPU
    void L2Norm::ForwardGpu()
    {

    } 

    void L2Norm::BackwardGpu()
    {

    }

    void L2Norm::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
}

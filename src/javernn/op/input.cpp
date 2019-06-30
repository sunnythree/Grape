#include "javernn/op/input.h"
#include "javernn/log.h"

static std::string TAG = "Input";

namespace javernn
{
    Input::Input(uint32_t batch_size, uint32_t in_dim, uint32_t label_dim):
    Op({},{DATA,DATA}),
    batch_size_(batch_size),
    in_dim_(in_dim)
    {
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,in_dim_}),DATA,gNetMode);
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,in_dim_}),DATA,gNetMode);
    }
    
    Input::~Input()
    {
    }

    void Input::Setup()
    {
        Log::v(TAG,"Setup");
    }

    void Input::ForwardCpu()
    {
        Log::v(TAG,"ForwardCpu");
    } 

    void Input::BackwardCpu()
    {
        Log::v(TAG,"BackwardCpu");
    }

    void Input::UpdateWeightsCpu(Optimizer &opt)
    {

    }

    std::vector<tensorptr_t> Input::GetOutputTensor()
    {
        return next_;
    }

#ifdef GPU
    void Input::ForwardGpu()
    {

    } 

    void Input::BackwardGpu()
    {

    }

    void Input::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
}

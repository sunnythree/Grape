#include <string>
#include "grape/op/input.h"
#include "grape/log.h"

namespace Grape
{
    static std::string TAG = "Input";

    Input::Input(std::string name, uint32_t batch_size, uint32_t out_dim):
    Op({},{DATA}),
    batch_size_(batch_size),
    out_dim_(out_dim)
    {
        type_ = "Input";
        name_ = name;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,out_dim_}),DATA,cal_mode_);
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

    Tensor* Input::GetOutputTensor()
    {
        return next_[0].get();
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

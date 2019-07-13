#include <algorithm>
#include "javernn/op/softmax.h"
#include "javernn/util/blas.h"
#include "javernn/util/util.h"
#include "javernn/log.h"

static std::string TAG = "Softmax";

namespace javernn
{
    Softmax::Softmax(std::string name, uint32_t batch_size, uint32_t in_dim):
    Op({DATA},{DATA}),
    batch_size_(batch_size),
    in_dim_(in_dim),
    temperature_(1.)
    {
        type_ = "SoftmaxWithLoss";
        name_ = name;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,in_dim_}),DATA,gNetMode);
    }

    Softmax::~Softmax()
    {
    }

    void Softmax::Setup()
    {
        Log::v(TAG,"Setup");
    }

    void Softmax::ForwardCpu()
    {
        Log::v(TAG,"SoftmaxWithLoss");
        Tensor *intput_tensor = prev_[0].get();
        Tensor *output_tensor = next_[0].get();
        float *intput_data = (float *)intput_tensor->mutable_cpu_data();
        float *output_data = (float *)output_tensor->mutable_cpu_data();
        for(int i=0;i<batch_size_;i++){
            softmax(intput_data+i*in_dim_, in_dim_, temperature_, 1, output_data+i*in_dim_);
        }
    }

    void Softmax::BackwardCpu()
    {
        Log::v(TAG,"BackwardCpu");
    }

    void Softmax::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void Softmax::ForwardGpu()
    {

    } 

    void Softmax::BackwardGpu()
    {

    }

    void Softmax::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
} // namespace javernn

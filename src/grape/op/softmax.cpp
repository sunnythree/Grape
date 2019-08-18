#include <algorithm>
#include "grape/op/softmax.h"
#include "grape/util/blas.h"
#include "grape/util/util.h"
#include "grape/log.h"
#include "grape/global_config.h"

namespace Grape
{
    const static std::string TAG = "Softmax";
    

    Softmax::Softmax(std::string name, uint32_t batch_size, uint32_t in_dim):
    Op({DATA},{DATA}),
    batch_size_(batch_size),
    in_dim_(in_dim),
    temperature_(1.)
    {
        type_ = STRING_SOFTMAX_TYPE;
        name_ = name;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,in_dim_}),DATA,sizeof(float));
    }

    Softmax::~Softmax()
    {
    }

    void Softmax::Setup()
    {

    }

    void Softmax::ForwardCpu()
    {
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

    }

    void Softmax::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void Softmax::ForwardGpu()
    {
        Tensor *intput_tensor = prev_[0].get();
        Tensor *output_tensor = next_[0].get();
        float *intput_data = (float *)intput_tensor->gpu_data();
        float *output_data = (float *)output_tensor->mutable_gpu_data();
        for(int i=0;i<batch_size_;i++){
            softmax_gpu(intput_data+i*in_dim_, in_dim_, temperature_, 1, output_data+i*in_dim_);
        }
    } 

    void Softmax::BackwardGpu()
    {

    }

    void Softmax::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
} // namespace Grape

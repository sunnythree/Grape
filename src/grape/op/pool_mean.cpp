#include "grape/op/pool_mean.h"
#include "grape/util/pool.h"
#include "grape/global_config.h"

namespace Grape
{
    const static std::string TAG = "PoolMean";
    
    PoolMean::PoolMean(
            std::string name, 
            uint32_t batch_size,
            uint32_t in_c,
            uint32_t in_h,
            uint32_t in_w
            ):
        Op({DATA,WEIGHTS,BIAS}, {DATA}),
        batch_size_(batch_size),
        in_w_(in_w),
        in_h_(in_h),
        in_c_(in_c)
    {
        name_ = name;
        type_ = STRING_POOL_MEAN_TYPE;
        int output_size = in_c * batch_size;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({batch_size_,in_c}),DATA,sizeof(float));
    }
    PoolMean::~PoolMean()
    {
        
    }
    void PoolMean::Setup()
    {

    }

    void PoolMean::ForwardCpu()
    {
        Tensor* in_data_tensor = prev_[0].get();
        Tensor* out_data_tensor = next_[0].get();
        float* in_data = (float *)in_data_tensor->cpu_data();
        float* out_data = (float *)out_data_tensor->mutable_cpu_data();
        forward_meanpool_cpu(batch_size_,in_w_,in_h_,in_c_,in_data,out_data);
    } 

    void PoolMean::BackwardCpu()
    {
        Tensor* in_data_tensor = prev_[0].get();
        Tensor* out_data_tensor = next_[0].get();
        float* out_diff = (float *)in_data_tensor->mutable_cpu_diff();
        float* in_diff = (float *)out_data_tensor->cpu_diff();
        backward_meanpool_cpu(batch_size_,in_w_,in_h_,in_c_,in_diff,out_diff);
    }

    void PoolMean::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void PoolMean::ForwardGpu()
    {
        Tensor* in_data_tensor = prev_[0].get();
        Tensor* out_data_tensor = next_[0].get();
        float* in_data = (float *)in_data_tensor->gpu_data();
        float* out_data = (float *)out_data_tensor->mutable_gpu_data();
        forward_meanpool_gpu(in_data_tensor->shape().count(),in_w_,in_h_,in_c_,in_data,out_data);
    } 

    void PoolMean::BackwardGpu()
    {
        Tensor* in_data_tensor = prev_[0].get();
        Tensor* out_data_tensor = next_[0].get();
        float* out_diff = (float *)in_data_tensor->mutable_gpu_diff();
        float* in_diff = (float *)out_data_tensor->gpu_diff();
        backward_meanpool_gpu(in_data_tensor->shape().count(),in_w_,in_h_,in_c_,in_diff,out_diff);
    }

    void PoolMean::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
} // namespace Grape


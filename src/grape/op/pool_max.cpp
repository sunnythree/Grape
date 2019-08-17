#include "grape/op/pool_max.h"
#include "grape/util/pool.h"

namespace Grape
{
    const static std::string TAG = "PoolMax";
    const static std::string POOL_MAX_TYPE = "PoolMax";
    PoolMax::PoolMax(
            std::string name, 
            uint32_t batch_size,
            uint32_t in_w,
            uint32_t in_h,
            uint32_t in_c,
            uint32_t ksize,
            uint32_t stride,
            uint32_t padding
            ):
        Op({DATA,WEIGHTS}, {DATA}),
        batch_size_(batch_size),
        in_w_(in_w),
        in_h_(in_h),
        in_c_(in_c),
        ksize_(ksize),
        stride_(stride),
        padding_(padding)
    {
        out_w_ = (in_w_ + padding - ksize_)/stride + 1;
        out_h_ = (in_h_ + padding - ksize_)/stride + 1;
        out_c_ = in_c_;
        unsigned int output_size = out_w_ * out_h_ * out_c_ * batch_size_;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({(unsigned int)(output_size*sizeof(float))}),DATA,sizeof(float));
        prev_[1] = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({(unsigned int)(output_size*sizeof(int))}),INDEX,sizeof(float));
    }
    PoolMax::~PoolMax()
    {
        
    }
    void PoolMax::Setup()
    {

    }

    void PoolMax::ForwardCpu()
    {
        Tensor* in_data_tensor = prev_[0].get();
        Tensor* out_data_tensor = next_[0].get();
        Tensor* index_tensor = prev_[1].get();
        float* in_data = (float *)in_data_tensor->cpu_data();
        float* out_data = (float *)out_data_tensor->mutable_cpu_data();
        int* indexes = (int *)index_tensor->mutable_cpu_data();
        forward_maxpool_cpu(
            batch_size_,in_w_,in_h_,out_w_,
            out_h_,in_c_,stride_,ksize_,
            padding_,in_data,out_data,indexes);
    } 

    void PoolMax::BackwardCpu()
    {
        Tensor* in_data_tensor = prev_[0].get();
        Tensor* out_data_tensor = next_[0].get();
        Tensor* index_tensor = prev_[1].get();
        float* out_diff = (float *)in_data_tensor->mutable_cpu_diff();
        float* in_diff = (float *)out_data_tensor->cpu_diff();
        int* indexes = (int *)index_tensor->cpu_data();
        backward_maxpool_cpu(index_tensor->shape().count(),in_diff,out_diff,indexes);
    }

    void PoolMax::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void PoolMax::ForwardGpu()
    {
        Tensor* in_data_tensor = prev_[0].get();
        Tensor* out_data_tensor = next_[0].get();
        Tensor* index_tensor = prev_[1].get();
        float* in_data = (float *)in_data_tensor->gpu_data();
        float* out_data = (float *)out_data_tensor->mutable_gpu_data();
        int* indexes = (int *)index_tensor->mutable_gpu_data();
        int n = index_tensor->shape().count();
        forward_maxpool_gpu(n,in_w_,in_h_,in_c_,stride_,ksize_,padding_,in_data,out_data,indexes);
    } 

    void PoolMax::BackwardGpu()
    {
        Tensor* in_data_tensor = prev_[0].get();
        Tensor* out_data_tensor = next_[0].get();
        Tensor* index_tensor = prev_[1].get();
        float* out_diff = (float *)in_data_tensor->mutable_gpu_diff();
        float* in_diff = (float *)out_data_tensor->gpu_diff();
        int* indexes = (int *)index_tensor->gpu_data();
        int n = index_tensor->shape().count();
        backward_maxpool_gpu(n,in_w_,in_h_,in_c_,stride_,ksize_,padding_,in_diff,out_diff,indexes);
    }

    void PoolMax::UpdateWeightsGpu(Optimizer &opt)
    {

    }

#endif
} // namespace Grape


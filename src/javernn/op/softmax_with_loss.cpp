#include <algorithm>
#include "javernn/op/softmax_with_loss.h"
#include "javernn/util/blas.h"
#include "javernn/util/util.h"
#include "javernn/log.h"

static std::string TAG = "SoftmaxWithLoss";

namespace javernn
{
    SoftmaxWithLoss::SoftmaxWithLoss(uint32_t batch_size, uint32_t in_dim):
    Op({DATA,DATA},{}),
    batch_size_(batch_size),
    in_dim_(in_dim)
    {
        inner_ = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,in_dim_}),DATA,gNetMode);
        cost_ = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,in_dim_}),DATA,gNetMode);
    }

    SoftmaxWithLoss::~SoftmaxWithLoss()
    {
    }

    void SoftmaxWithLoss::Setup()
    {
        Log::v(TAG,"Setup");
    }

    void SoftmaxWithLoss::ForwardCpu()
    {
        Log::v(TAG,"SoftmaxWithLoss");
        Tensor *intput_tensor = prev_[0].get();
        Tensor *label_tensor = prev_[1].get();
        Tensor *output_tensor = inner_.get();
        float *intput_data = (float *)intput_tensor->mutable_cpu_data();
        float *label_data = (float *)label_tensor->mutable_cpu_data();
        float *output_data = (float *)output_tensor->mutable_cpu_data();
        float *output_diff = (float *)output_tensor->mutable_cpu_diff();
        float *cost_data = (float *)cost_->mutable_cpu_data();
        int32_t n = intput_tensor->shape().count();
        softmax(intput_data, n, temperature_, 1, output_data);
        softmax_x_ent_cpu(n, output_data, label_data, output_diff, cost_data);
        float cost_sum = sum_array(cost_data,n);
        Log::v(TAG,"cost_sum is "+ std::to_string(cost_sum));
    }

    void SoftmaxWithLoss::BackwardCpu()
    {
        Log::v(TAG,"BackwardCpu");
        Tensor *intput_tensor = prev_[0].get();
        Tensor *output_tensor = inner_.get();
        float *intput_diff = (float *)output_tensor->mutable_cpu_diff();
        float *output_diff = (float *)intput_tensor->mutable_cpu_diff();
        int32_t n = intput_tensor->shape().count();
        axpy_cpu(n, 1, intput_diff, 1, output_diff, 1);
    }

    void SoftmaxWithLoss::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void SoftmaxWithLoss::ForwardGpu()
    {

    } 

    void SoftmaxWithLoss::BackwardGpu()
    {

    }

    void SoftmaxWithLoss::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
} // namespace javernn

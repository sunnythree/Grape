#include <algorithm>
#include "grape/op/softmax_with_loss.h"
#include "grape/util/blas.h"
#include "grape/util/util.h"
#include "grape/log.h"
#include "grape/global_config.h"


namespace Grape
{
    const static std::string TAG = "SoftmaxWithLoss";
    
    SoftmaxWithLoss::SoftmaxWithLoss(std::string name, uint32_t batch_size, uint32_t in_dim):
    Op({DATA,DATA},{DATA}),
    batch_size_(batch_size),
    in_dim_(in_dim),
    temperature_(1.)
    {
        type_ = STRING_SOFTMAX_WITH_LOSS_TYPE;
        name_ = name;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,in_dim_}),DATA,sizeof(float));
        cost_ = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,in_dim_}),DATA,sizeof(float));
    }

    SoftmaxWithLoss::~SoftmaxWithLoss()
    {

    }

    void SoftmaxWithLoss::Setup()
    {

    }

    void SoftmaxWithLoss::ForwardCpu()
    {
        Tensor *intput_tensor = prev_[0].get();
        Tensor *label_tensor = prev_[1].get();
        Tensor *output_tensor = next_[0].get();
        float *intput_data = (float *)intput_tensor->cpu_data();
        float *intput_diff = (float *)intput_tensor->mutable_cpu_diff();
        float *label_data = (float *)label_tensor->cpu_data();
        float *output_data = (float *)output_tensor->mutable_cpu_data();
        float *cost_data = (float *)cost_->mutable_cpu_data();
        int32_t n = intput_tensor->shape().count();
        // for(int i=0;i<n;i++){
        //     std::cout<<intput_data[i]<<" ";
        // }
        // std::cout<<std::endl;
        // for(int i=0;i<n;i++){
        //     std::cout<<label_data[i]<<" ";
        // }
        // std::cout<<std::endl;
        for(int i=0;i<batch_size_;++i){
            softmax(intput_data+i*in_dim_, in_dim_, temperature_, 1, output_data+i*in_dim_);
        }

        softmax_x_ent_cpu(in_dim_*batch_size_, output_data, label_data, intput_diff, cost_data);

    }

    void SoftmaxWithLoss::BackwardCpu()
    {

    }

    void SoftmaxWithLoss::UpdateWeightsCpu(Optimizer &opt)
    {

    }

    void SoftmaxWithLoss::Display()
    {
        float *cost_data = (float *)cost_->cpu_data();

        float cost_sum = sum_array(cost_data,cost_->shape().count())/batch_size_;
        Log::v(TAG,"cost_sum is "+ std::to_string(cost_sum));
    }

#ifdef GPU
    void SoftmaxWithLoss::ForwardGpu()
    {
        Tensor *intput_tensor = prev_[0].get();
        Tensor *label_tensor = prev_[1].get();
        Tensor *output_tensor = next_[0].get();
        float *intput_data = (float *)intput_tensor->gpu_data();
        float *intput_diff = (float *)intput_tensor->mutable_gpu_diff();
        float *label_data = (float *)label_tensor->gpu_data();
        float *output_data = (float *)output_tensor->mutable_gpu_data();
        float *cost_data = (float *)cost_->mutable_gpu_data();
        int32_t n = intput_tensor->shape().count();
        for(int i=0;i<batch_size_;++i){
            softmax_gpu(intput_data+i*in_dim_, in_dim_, temperature_, 1, output_data+i*in_dim_);
        }

        softmax_x_ent_gpu(in_dim_*batch_size_, output_data, label_data, intput_diff, cost_data);
    } 

    void SoftmaxWithLoss::BackwardGpu()
    {

    }

    void SoftmaxWithLoss::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
} // namespace Grape

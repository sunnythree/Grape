#include <algorithm>
#include "grape/op/softmax_with_loss.h"
#include "grape/util/blas.h"
#include "grape/util/util.h"
#include "grape/log.h"

static std::string TAG = "SoftmaxWithLoss";

namespace Grape
{
    SoftmaxWithLoss::SoftmaxWithLoss(std::string name, uint32_t batch_size, uint32_t in_dim):
    Op({DATA,DATA},{DATA}),
    batch_size_(batch_size),
    in_dim_(in_dim),
    temperature_(1.)
    {
        type_ = "SoftmaxWithLoss";
        name_ = name;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,in_dim_}),DATA,cal_mode_);
        cost_ = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,in_dim_}),DATA,cal_mode_);
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
        float *intput_data = (float *)intput_tensor->mutable_cpu_data();
        float *intput_diff = (float *)intput_tensor->mutable_cpu_diff();
        float *label_data = (float *)label_tensor->mutable_cpu_data();
        float *output_data = (float *)output_tensor->mutable_cpu_data();
        float *output_diff = (float *)output_tensor->mutable_cpu_diff();
        float *cost_data = (float *)cost_->mutable_cpu_data();
        int32_t n = intput_tensor->shape().count();
        // printf("n is %d\n",n);
        // std::cout<<std::endl;
        // for(int i=0;i<in_dim_*batch_size_;i++){
        //     std::cout<<intput_data[i]<<" ";
        // }
        // std::cout<<std::endl;
        // for(int i=0;i<in_dim_*batch_size_;i++){
        //     std::cout<<label_data[i]<<" ";
        // }
        // std::cout<<std::endl;
        for(int i=0;i<batch_size_;i++){
            softmax(intput_data+i*in_dim_, in_dim_, temperature_, 1, output_data+i*in_dim_);
        }
        // for(int i=0;i<in_dim_*batch_size_;i++){
        //     std::cout<<output_data[i]<<" ";
        // }
        // std::cout<<std::endl;
        softmax_x_ent_cpu(in_dim_*batch_size_, output_data, label_data, intput_diff, cost_data);
        // for(int i=0;i<in_dim_*batch_size_;i++){
        //     std::cout<<intput_diff[i]<<" ";
        // }
        // std::cout<<std::endl;
        
    }

    void SoftmaxWithLoss::BackwardCpu()
    {
        // Tensor *intput_tensor = prev_[0].get();
        // Tensor *output_tensor = inner_.get();
        // float *intput_diff = (float *)output_tensor->mutable_cpu_diff();
        // float *output_diff = (float *)intput_tensor->mutable_cpu_diff();
        // int32_t n = intput_tensor->shape().count();
        // axpy_cpu(n, 1, intput_diff, 1, output_diff, 1);
        // std::cout<<"SoftmaxWithLoss diff"<<std::endl;
        // for(int i=0;i<in_dim_;i++){
        //     std::cout<<intput_diff[i]<<" ";
        // }
        // std::cout<<std::endl;
    }

    void SoftmaxWithLoss::UpdateWeightsCpu(Optimizer &opt)
    {

    }

    void SoftmaxWithLoss::Display()
    {
        float *cost_data = (float *)cost_->mutable_cpu_data();
        float cost_sum = sum_array(cost_data,cost_->shape().count())/batch_size_;
        Log::v(TAG,"cost_sum is "+ std::to_string(cost_sum));
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
} // namespace Grape

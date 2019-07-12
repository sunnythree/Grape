#include <algorithm>
#include "javernn/op/softmax_with_loss.h"
#include "javernn/util/blas.h"
#include "javernn/util/util.h"
#include "javernn/log.h"

static std::string TAG = "SoftmaxWithLoss";

namespace javernn
{
    SoftmaxWithLoss::SoftmaxWithLoss(uint32_t batch_size, uint32_t in_dim):
    Op({DATA,DATA},{DATA}),
    batch_size_(batch_size),
    in_dim_(in_dim),
    temperature_(1.)
    {
        type_ = "SoftmaxWithLoss";
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
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
        float cost_sum = sum_array(cost_data,n)/batch_size_;
        Log::v(TAG,"cost_sum is "+ std::to_string(cost_sum));
    }

    void SoftmaxWithLoss::BackwardCpu()
    {
        Log::v(TAG,"BackwardCpu");
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

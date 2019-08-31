
#include <memory>
#include <assert.h>
#include <chrono>
#include <fstream>
#include "grape/op/fc.h"
#include "grape/log.h"
#include "grape/util/random.h"
#include "grape/util/gemm.h"

#include "grape/util/cuda.h"
#include "grape/util/blas.h"

namespace Grape{
    const static std::string TAG = "Fc";
    
    Fc::Fc(
        std::string name, 
        uint32_t batch_size,
        uint32_t in_dim,
        uint32_t out_dim,
        bool has_bias,
        ACTIVATION activation): 
    Op({DATA,WEIGHTS,BIAS}, {DATA}),
    batch_size_(batch_size),
    in_dim_(in_dim),
    out_dim_(out_dim),
    has_bias_(has_bias),
    activation_(activation),
    setuped_(false)
    {
        type_ = STRING_FC_TYPE;
        name_ = name;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({batch_size_,out_dim_}),DATA,sizeof(float));
        //reverve size
        if(has_bias_){
            prev_.reserve(3);
        }else{
            prev_.reserve(2);
        }

        prev_[1] = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({out_dim_,in_dim_}),WEIGHTS,sizeof(float));

        if(has_bias_){
            prev_[2] = std::make_shared<Tensor>(static_cast<Op *>(this),
                Shape({out_dim_}),BIAS,sizeof(float));
        }
    }

    Fc::~Fc()
    {
    }

    void Fc::Setup()
    {
        if(setuped_){
            return;
        }
        //create input tensor,only weights and bias
        if(prev_.size()==0){
            Log::v(TAG,"skip init weights");
        }else{
            //Log::v(TAG,"create weights");
            //assert(prev_[0].get() != nullptr);

            unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
            Random::GetInstance().SetSeed(seed);
            Random::GetInstance().SetNormalFloat((float *)prev_[1]->mutable_cpu_data(),
            prev_[1]->shape().count(),0,0.1);
    
            fill_cpu(prev_[1]->shape().count(),0,(float *)prev_[1]->mutable_cpu_diff(),1);

            if(has_bias_){
                //Log::v(TAG,"create bias");
                fill_cpu(prev_[2]->shape().count(),0,(float *)prev_[2]->mutable_cpu_data(),1);
                fill_cpu(prev_[2]->shape().count(),0,(float *)prev_[2]->mutable_cpu_diff(),1);
            }
            setuped_ = true;
        }
    }

    void Fc::ForwardCpu()
    {
        //get data
        Tensor* data_tensor = prev_[0].get();
        Tensor* weight_tensor = prev_[1].get();
        Tensor* bias_tensor = prev_[2].get();
        Tensor* out_data_tensor = next_[0].get();
        int m = batch_size_;
        int k = in_dim_;
        int n = out_dim_;
        assert(data_tensor != nullptr);
        assert(weight_tensor != nullptr);
        assert(out_data_tensor != nullptr);
        float *a = (float *)data_tensor->cpu_data();
        float *b = (float *)weight_tensor->cpu_data();
        float *c = (float *)out_data_tensor->mutable_cpu_data();
        fill_cpu(batch_size_*out_dim_,0,c,1);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
        if(has_bias_){
            float *bias_data = (float *)bias_tensor->cpu_data();
            add_bias(c, bias_data, batch_size_, out_dim_, 1);
        }
        
        activate_array(c,batch_size_*out_dim_,activation_);
    } 

    void Fc::BackwardCpu()
    {
        //get data
        Tensor* data_tensor = prev_[0].get();
        Tensor* weight_tensor = prev_[1].get();
        Tensor* out_data_tensor = next_[0].get();
        Tensor* bias_tensor = prev_[2].get();

        float *output_data = (float *)out_data_tensor->cpu_data();
        float *input_diff = (float *)out_data_tensor->mutable_cpu_diff();

        gradient_array(output_data,batch_size_*out_dim_,activation_,input_diff);

        if(has_bias_) {
            float *bias_diff = (float *)bias_tensor->cpu_diff();
            backward_bias(bias_diff,input_diff, batch_size_, out_dim_, 1);
        }

        int m = out_dim_;
        int k = batch_size_;
        int n = in_dim_;
        float *a = (float *)input_diff;
        float *b = (float *)data_tensor->cpu_data();
        float *c = (float *)weight_tensor->mutable_cpu_diff();
        gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);


        m = batch_size_;
        k = out_dim_;
        n = in_dim_;

        a = (float *)input_diff;
        b = (float *)weight_tensor->cpu_data();
        c = (float *)data_tensor->mutable_cpu_diff();
        fill_cpu(batch_size_*in_dim_,0,c,1);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }

    void Fc::UpdateWeightsCpu(Optimizer &opt)
    {
        opt.UpdateCpu( prev_[1].get(),batch_size_);
        if(has_bias_){
            opt.UpdateCpu( prev_[2].get(),batch_size_);
        }
    }

#ifdef GPU

    void Fc::ForwardGpu()
    {
       //get data
        Tensor* data_tensor = prev_[0].get();
        Tensor* weight_tensor = prev_[1].get();
        Tensor* bias_tensor = prev_[2].get();
        Tensor* out_data_tensor = next_[0].get();
        int m = batch_size_;
        int k = in_dim_;
        int n = out_dim_;
        assert(data_tensor != nullptr);
        assert(weight_tensor != nullptr);
        assert(out_data_tensor != nullptr);
        float *a = (float *)data_tensor->gpu_data();
        float *b = (float *)weight_tensor->gpu_data();
        float *c = (float *)out_data_tensor->mutable_gpu_data();
        fill_gpu(batch_size_*out_dim_,0,c,1);
        gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
        if(has_bias_){
            float *bias_data = (float *)bias_tensor->gpu_data();
            add_bias_gpu(c, bias_data, batch_size_, out_dim_, 1);
        }
        
        activate_array_gpu(c,batch_size_*out_dim_,activation_);
    }

    void Fc::BackwardGpu()
    {
        //get data
        Tensor* data_tensor = prev_[0].get();
        Tensor* weight_tensor = prev_[1].get();
        Tensor* out_data_tensor = next_[0].get();
        Tensor* bias_tensor = prev_[2].get();

        float *output_data = (float *)out_data_tensor->gpu_data();
        float *input_diff = (float *)out_data_tensor->mutable_gpu_diff();

        gradient_array_gpu(output_data,batch_size_*out_dim_,activation_,input_diff);

        if(has_bias_) {
            float *bias_diff = (float *)bias_tensor->gpu_diff();
            backward_bias_gpu(bias_diff,input_diff, batch_size_, out_dim_, 1);
        }

        int m = out_dim_;
        int k = batch_size_;
        int n = in_dim_;
        float *a = (float *)input_diff;
        float *b = (float *)data_tensor->gpu_data();
        float *c = (float *)weight_tensor->mutable_gpu_diff();
        gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);


        m = batch_size_;
        k = out_dim_;
        n = in_dim_;

        a = (float *)input_diff;
        b = (float *)weight_tensor->gpu_data();
        c = (float *)data_tensor->mutable_gpu_diff();
        fill_gpu(batch_size_*in_dim_,0,c,1);
        gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }

    void Fc::UpdateWeightsGpu(Optimizer &opt)
    {
        opt.UpdateGpu( prev_[1].get(),batch_size_);
        if(has_bias_){
            opt.UpdateGpu( prev_[2].get(),batch_size_);
        }
    }
#endif

    void Fc::Load(cereal::BinaryInputArchive &archive)
    {
        archive(cereal::make_nvp("weight", *prev_[1].get()));
        if(has_bias_){
            archive(cereal::make_nvp("bias", *prev_[2].get()));
        }
    }

    void Fc::Load(cereal::JSONInputArchive &archive)
    {
        archive(cereal::make_nvp("weight", *prev_[1].get()));
        if(has_bias_){
            archive(cereal::make_nvp("bias", *prev_[2].get()));
        }
    }

    void Fc::Load(cereal::XMLInputArchive &archive)
    {
        archive(cereal::make_nvp("weight", *prev_[1].get()));
        if(has_bias_){
            archive(cereal::make_nvp("bias", *prev_[2].get()));
        }
    }

    void Fc::Save(cereal::BinaryOutputArchive &archive)
    {
        archive(cereal::make_nvp("weight", *prev_[1].get()));
        if(has_bias_){
            archive(cereal::make_nvp("bias", *prev_[2].get()));
        }
    }

    void Fc::Save(cereal::JSONOutputArchive &archive)
    {
        archive(cereal::make_nvp("weight", *prev_[1].get()));
        if(has_bias_){
            archive(cereal::make_nvp("bias", *prev_[2].get()));
        }
    }

    void Fc::Save(cereal::XMLOutputArchive &archive)
    {
        archive(cereal::make_nvp("weight", *prev_[1].get()));
        if(has_bias_){
            archive(cereal::make_nvp("bias", *prev_[2].get()));
        }
    }
}
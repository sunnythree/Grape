
#include <memory>
#include "javernn/op/fc.h"
#include "javernn/log.h"
#include "javernn/util/random.h"
#include "javernn/util/blas.h"
#include "javernn/util/gemm.h"

namespace javernn{
    static std::string TAG = "Fc";
    Fc::Fc(uint32_t batch_size,uint32_t in_dim,uint32_t out_dim,bool has_bias)
    : Op({DATA,WEIGHTS,BIAS}, {DATA}),
    batch_size_(batch_size),
    in_dim_(in_dim),
    out_dim_(out_dim),
    has_bias_(has_bias)
    {
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,out_dim_}),DATA,gNetMode);
    }

    Fc::~Fc()
    {

    }

    void Fc::Setup()
    {
        //create input tensor,only weights and bias
        if(prev_.size()>0){
            Log::i(TAG,"skip init weights");
        }else{
            for(int i=0;i<prev_.size();i++){
                if(in_type_[i] == WEIGHTS){
                    prev_[i] = std::make_shared<Tensor>(static_cast<Op *>(this),
                    Shape({in_dim_,out_dim_}),DATA,gNetMode);
                    Random::GetInstance().SetNormalFloat((float *)prev_[i]->cpu_data(),
                    prev_[i]->shape().count(),0,1);
                    if(gNetMode == GPU_MODE){
                        prev_[i]->data_to_gpu();
                    }
                }else if(in_type_[i] == BIAS && has_bias_){
                    prev_[i] = std::make_shared<Tensor>(static_cast<Op *>(this),
                    Shape({out_dim_}),DATA,gNetMode);
                    fill_cpu(prev_[i]->shape().count(),0,(float *)prev_[i]->cpu_data(),1);
                    if(gNetMode == GPU_MODE){
                        prev_[i]->data_to_gpu();
                    }
                }
            }
        }
    }

    void Fc::ForwardCpu()
    {
        Log::v(TAG," Fc ForwardCpu");
        //get data
        Tensor* data_tensor = prev_[0].get();
        Tensor* weight_tensor = prev_[1].get();
        Tensor* out_data_tensor = next_[0].get();
        int m = batch_size_;
        int k = in_dim_;
        int n = out_dim_;
        float *a = (float *)data_tensor->mutable_cpu_data();
        float *b = (float *)weight_tensor->mutable_cpu_data();
        float *c = (float *)out_data_tensor->mutable_cpu_data();
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
        if(has_bias_){
            Tensor* bias_tensor = prev_[2].get();
            add_cpu(out_dim_,(float *)bias_tensor,1,(float *)out_data_tensor,1);
        }
    } 

    void Fc::BackwardCpu()
    {
        Log::v(TAG," Fc BackwardCpu");
        //get data
        Tensor* data_tensor = prev_[0].get();
        Tensor* weight_tensor = prev_[1].get();
        Tensor* out_data_tensor = next_[0].get();
        int m = out_dim_;
        int k = batch_size_;
        int n = in_dim_;
        float *a = (float *)data_tensor->mutable_cpu_data();
        float *b = (float *)out_data_tensor->mutable_cpu_diff();
        float *c = (float *)weight_tensor->mutable_cpu_diff();
        gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

        m = batch_size_;
        k = out_dim_;
        n = in_dim_;

        a = (float *)out_data_tensor->mutable_cpu_diff();
        b = (float *)weight_tensor->mutable_cpu_diff();
        c = (float *)data_tensor->mutable_cpu_diff();

        if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }

    void Fc::UpdateWeightsCpu(Optimizer &opt)
    {
        Log::v(TAG," Fc UpdateWeightsCpu");
        opt.UpdateCpu( prev_[1].get());
        if(has_bias_){
            opt.UpdateCpu( prev_[2].get());
        }
    }

#ifdef GPU

    void Fc::ForwardGpu()
    {

    }

    void Fc::BackwardGpu()
    {

    }

    void Fc::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
}
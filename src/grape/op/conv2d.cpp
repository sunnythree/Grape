#include <chrono>
#include "grape/log.h"
#include "grape/op/conv2d.h"
#include "grape/util/gemm.h"
#include "grape/util/blas.h"
#include "grape/util/im2col.h"
#include "grape/util/col2im.h"
#include "grape/util/col2im.h"
#include "grape/util/random.h"
#include "grape/global_config.h"

namespace Grape
{
    const static std::string TAG = "Conv2d";
    
    Conv2d::Conv2d(
            std::string name, 
            uint32_t batch_size,
            uint32_t in_c,
            uint32_t in_h,
            uint32_t in_w,
            uint32_t out_c,
            uint32_t group,
            uint32_t ksize,
            uint32_t stride,
            uint32_t padding,
            bool has_bias = true,
            ACTIVATION activation = LEAKY
            ):
        Op({DATA,WEIGHTS,BIAS}, {DATA}),
        batch_size_(batch_size),
        in_w_(in_w),
        in_h_(in_h),
        in_c_(in_c),
        out_c_(out_c),
        group_(group),
        ksize_(ksize),
        stride_(stride),
        padding_(padding),
        has_bias_(has_bias),
        activation_(activation)
    {
        type_ = STRING_CONV2D_TYPE;
        name_ = name;
        out_w_ = (in_w_+ 2*padding_-ksize_)/stride_ + 1;
        out_h_ = (in_h_+ 2*padding_-ksize_)/stride_ + 1;
        noutputs_ = out_c_*out_h_*out_w_;
        nweights_ = in_c/group_*out_c*ksize*ksize;;
        
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({batch_size_,out_c_,out_h_,out_w_}),DATA,sizeof(float));
        //reverve size
        if(has_bias_){
            prev_.reserve(3);
        }else{
            prev_.reserve(2);
        }

        prev_[1] = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({in_c_/group_*out_c_,ksize,ksize}),WEIGHTS,sizeof(float));

        if(has_bias_){
            prev_[2] = std::make_shared<Tensor>(static_cast<Op *>(this),
                Shape({out_c_}),BIAS,sizeof(float));
        }

        im_col_tensor_ = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({in_c_/group_*ksize*ksize,out_w_*out_h_}),AUX,sizeof(float));
    }
    Conv2d::~Conv2d()
    {
        
    }
    void Conv2d::Setup()
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

            float scale = sqrt(2./(ksize_*ksize_*in_c_/group_));
            int n = prev_[1]->shape().count();
            unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
            Random::GetInstance().SetSeed(seed);
            Random::GetInstance().SetNormalFloat((float *)prev_[1]->mutable_cpu_data(),n,0,scale);
    
            fill_cpu(n,0,(float *)prev_[1]->mutable_cpu_diff(),1);

            if(has_bias_){
                //Log::v(TAG,"create bias");
                fill_cpu(prev_[2]->shape().count(),0,(float *)prev_[2]->mutable_cpu_data(),1);
                fill_cpu(prev_[2]->shape().count(),0,(float *)prev_[2]->mutable_cpu_diff(),1);
            }
            setuped_ = true;
        }
    }

    void Conv2d::ForwardCpu()
    {
        Tensor* in_tensor = prev_[0].get();
        Tensor* weight_tensor = prev_[1].get();
        Tensor* out_tensor = next_[0].get();
        Tensor* im_col_tensor = im_col_tensor_.get();

        float *in_data = (float *)in_tensor->cpu_data();
        float *out_data = (float *)out_tensor->mutable_cpu_data();
        float *weight_data = (float *)weight_tensor->cpu_data();
        float *im_col_data = (float *)im_col_tensor->mutable_cpu_data();

        fill_cpu(out_tensor->shape().count(), 0, out_data, 1);
        int m = out_c_/group_;
        int k = ksize_*ksize_*in_c_/group_;
        int n = out_w_*out_h_;
        for(int i = 0; i < batch_size_; ++i){
            for(int j = 0; j < group_; ++j){
                float *a = weight_data + j*nweights_/group_;
                float *b = im_col_data;
                float *c = out_data + (i*group_ + j)*n*m;
                float *im =  in_data + (i*group_ + j)*in_c_/group_*in_h_*in_w_;

                if (ksize_ == 1) {
                    b = im;
                } else {
                    im2col_cpu(im,in_c_/group_, in_h_, in_w_, ksize_, stride_, padding_, b);
                }
                gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
            }
        }

        if(has_bias_) {
            Tensor* bias_tensor = prev_[2].get();
            float *bias_data = (float *)bias_tensor->cpu_data();
            add_bias(out_data, bias_data, batch_size_, out_c_, out_w_*out_h_);
        }

        activate_array(out_data, noutputs_*batch_size_, activation_);

    } 

    void Conv2d::BackwardCpu()
    {
        int m = out_c_/group_;
        int n = ksize_*ksize_*in_c_/group_;
        int k = out_w_*out_h_;

        Tensor* in_tensor = prev_[0].get();
        Tensor* weight_tensor = prev_[1].get();
        Tensor* out_tensor = next_[0].get();
        Tensor* im_col_tensor = im_col_tensor_.get();

        float *in_data = (float *)in_tensor->cpu_data();
        float *out_diff = (float *)in_tensor->mutable_cpu_diff();
        float *out_data = (float *)out_tensor->cpu_data();
        float *in_diff = (float *)out_tensor->cpu_diff();
        float *weight_data = (float *)weight_tensor->cpu_data();
        float *weight_diff = (float *)weight_tensor->mutable_cpu_diff();
        float *im_col_data = (float *)im_col_tensor->cpu_data();
        fill_cpu(in_tensor->shape().count(), 0, out_diff, 1);
        gradient_array(out_data, noutputs_*batch_size_, activation_, in_diff);

        if(has_bias_){
            Tensor* bias_tensor = prev_[2].get();
            float *bias_diff = (float *)bias_tensor->mutable_cpu_diff();
            backward_bias(bias_diff, in_diff, batch_size_, out_c_, k);
        }

        for(int i = 0; i < batch_size_; ++i){
            for(int j = 0; j < group_; ++j){
                float *a = in_diff + (i*group_ + j)*m*k;
                float *b = im_col_data;
                float *c = weight_diff +  j*nweights_/group_;

                float *im  = in_data + (i*group_ + j)*in_c_/group_*in_h_*in_w_;
                float *imd = out_diff + (i*group_ + j)*in_c_/group_*in_h_*in_w_;

                if(ksize_ == 1){
                    b = im;
                } else {
                    im2col_cpu(im,in_c_/group_, in_h_, in_w_, 
                            ksize_, stride_, padding_, b);
                }

                gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);


                a = weight_data + j*nweights_/group_;
                b = in_diff + (i*group_ + j)*m*k;
                c = im_col_data;
                if (ksize_ == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (ksize_ != 1) {
                    col2im_cpu(im_col_data, in_c_/group_, in_h_, in_w_, ksize_, stride_, padding_, imd);
                }
            }
        }
    }

    void Conv2d::UpdateWeightsCpu(Optimizer &opt)
    {
        opt.UpdateCpu( prev_[1].get(),batch_size_);
        if(has_bias_){
            opt.UpdateCpu( prev_[2].get(),batch_size_);
        }
    }

#ifdef GPU
    void Conv2d::ForwardGpu()
    {
        Tensor* in_tensor = prev_[0].get();
        Tensor* weight_tensor = prev_[1].get();
        Tensor* out_tensor = next_[0].get();
        Tensor* im_col_tensor = im_col_tensor_.get();

        float *in_data = (float *)in_tensor->gpu_data();
        float *out_data = (float *)out_tensor->mutable_gpu_data();
        float *weight_data = (float *)weight_tensor->gpu_data();
        float *im_col_data = (float *)im_col_tensor->mutable_gpu_data();

        fill_gpu(noutputs_*batch_size_, 0, out_data, 1);
        int m = out_c_/group_;
        int k = ksize_*ksize_*in_c_/group_;
        int n = out_w_*out_h_;
        for(int i = 0; i < batch_size_; ++i){
            for(int j = 0; j < group_; ++j){
                float *a = weight_data + j*nweights_/group_;
                float *b = im_col_data;
                float *c = out_data + (i*group_ + j)*n*m;
                float *im =  in_data + (i*group_ + j)*in_c_/group_*in_h_*in_w_;

                if (ksize_ == 1) {
                    b = im;
                } else {
                    im2col_gpu(im,in_c_/group_, in_h_, in_w_, ksize_, stride_, padding_, b);
                }
                gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
            }
        }

        if(has_bias_) {
            Tensor* bias_tensor = prev_[2].get();
            float *bias_data = (float *)bias_tensor->gpu_data();
            add_bias_gpu(out_data, bias_data, batch_size_, out_c_, out_w_*out_h_);
        }

        activate_array_gpu(out_data, noutputs_*batch_size_, activation_);
    } 

    void Conv2d::BackwardGpu()
    {
        int m = out_c_/group_;
        int n = ksize_*ksize_*in_c_/group_;
        int k = out_w_*out_h_;

        Tensor* in_tensor = prev_[0].get();
        Tensor* weight_tensor = prev_[1].get();
        Tensor* out_tensor = next_[0].get();
        Tensor* im_col_tensor = im_col_tensor_.get();

        float *in_data = (float *)in_tensor->gpu_data();
        float *out_diff = (float *)in_tensor->mutable_gpu_diff();
        float *out_data = (float *)out_tensor->gpu_data();
        float *in_diff = (float *)out_tensor->gpu_diff();
        float *weight_data = (float *)weight_tensor->gpu_data();
        float *weight_diff = (float *)weight_tensor->mutable_gpu_diff();
        float *im_col_data = (float *)im_col_tensor->gpu_data();
        fill_gpu(in_tensor->shape().count(), 0, out_diff, 1);
        gradient_array_gpu(out_data, noutputs_*batch_size_, activation_, in_diff);

        if(has_bias_){
            Tensor* bias_tensor = prev_[2].get();
            float *bias_diff = (float *)bias_tensor->mutable_gpu_diff();
            backward_bias_gpu(bias_diff, in_diff, batch_size_, out_c_, k);
        }

        for(int i = 0; i < batch_size_; ++i){
            for(int j = 0; j < group_; ++j){
                float *a = in_diff + (i*group_ + j)*m*k;
                float *b = im_col_data;
                float *c = weight_diff +  j*nweights_/group_;

                float *im  = in_data + (i*group_ + j)*in_c_/group_*in_h_*in_w_;
                float *imd = out_diff + (i*group_ + j)*in_c_/group_*in_h_*in_w_;

                if(ksize_ == 1){
                    b = im;
                } else {
                    im2col_gpu(im,in_c_/group_, in_h_, in_w_, 
                            ksize_, stride_, padding_, b);
                }

                gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);


                a = weight_data + j*nweights_/group_;
                b = in_diff + (i*group_ + j)*m*k;
                c = im_col_data;
                if (ksize_ == 1) {
                    c = imd;
                }

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (ksize_ != 1) {
                    col2im_gpu(im_col_data, in_c_/group_, in_h_, in_w_, ksize_, stride_, padding_, imd);
                }
            }
        }
    }

    void Conv2d::UpdateWeightsGpu(Optimizer &opt)
    {
        opt.UpdateGpu( prev_[1].get(),batch_size_);
        if(has_bias_){
            opt.UpdateGpu( prev_[2].get(),batch_size_);
        }
    }

#endif

    void Conv2d::Load(cereal::BinaryInputArchive &archive)
    {

    }

    void Conv2d::Load(cereal::JSONInputArchive &archive)
    {

    }

    void Conv2d::Load(cereal::XMLInputArchive &archive)
    {

    }

    void Conv2d::Save(cereal::BinaryOutputArchive &archive)
    {

    }

    void Conv2d::Save(cereal::JSONOutputArchive &archive)
    {

    }

    void Conv2d::Save(cereal::XMLOutputArchive &archive)
    {

    }
} // namespace Grape


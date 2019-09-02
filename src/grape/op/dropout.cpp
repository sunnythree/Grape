 #include "grape/op/dropout.h"
 #include "grape/log.h"
 #include "grape/util/util.h"
 #include "grape/global_config.h"
 #include "grape/util/dropout_util.h"
 #include "grape/util/blas.h"

 namespace Grape{
    const static std::string TAG = "Dropout";

    Dropout::Dropout(std::string name, uint32_t batch_size, uint32_t in_dim, float probability):
    Op({DATA},{DATA}),
    batch_size_(batch_size),
    in_dim_(in_dim)
    {
        type_ = STRING_DROPOUT_TYPE;
        name_ = name;
        batch_size_ = batch_size;
        in_dim_ = in_dim;
        probability_ = probability;
        scale_ = 1./(1.-probability);
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({batch_size_,in_dim}),DATA,sizeof(float));

        rand_ = std::make_shared<Tensor>(static_cast<Op *>(this),
            Shape({batch_size_,in_dim}),AUX,sizeof(float));
    }
    Dropout::~Dropout()
    {

    }
    void Dropout::Setup()
    {

    }
    void Dropout::ForwardCpu()
    {
        Tensor *intput_tensor = prev_[0].get();
        Tensor *output_tensor = next_[0].get();
        
        float *intput_data = (float *)intput_tensor->cpu_data();
        float *output_data = (float *)output_tensor->mutable_cpu_data();
        if(!is_train_){
            int size = batch_size_*in_dim_;
            copy_cpu(size,intput_data,1,output_data,1);
            return;
        }
        Tensor *rand_tensor = rand_.get();
        float *rand_data = (float *)rand_tensor->mutable_cpu_data();
        forward_dropout_cpu(batch_size_,in_dim_,intput_data,output_data,rand_data,probability_,scale_);
    }
    void Dropout::BackwardCpu()
    {
        if(!is_train_){
            return;
        }
        Tensor *intput_tensor = prev_[0].get();
        Tensor *output_tensor = next_[0].get();
        Tensor *rand_tensor = rand_.get();
        float *output_diff = (float *)intput_tensor->cpu_diff();
        float *input_diff = (float *)output_tensor->mutable_cpu_diff();
        float *rand_data = (float *)rand_tensor->mutable_cpu_data();
        backward_dropout_cpu(batch_size_,in_dim_,input_diff,output_diff,rand_data,probability_,scale_);
    }
    void Dropout::UpdateWeightsCpu(Optimizer &opt)
    {

    }
    void Dropout::OnTrainBegin()
    {
        is_train_ = true;
    }
    void Dropout::OnTestBegin()
    {
        is_train_ = false;
    }
    
#ifdef GPU
    void Dropout::ForwardGpu()
    {
        Tensor *intput_tensor = prev_[0].get();
        Tensor *output_tensor = next_[0].get();
        
        float *intput_data = (float *)intput_tensor->gpu_data();
        float *output_data = (float *)output_tensor->mutable_gpu_data();

        if(!is_train_){
            int size = batch_size_*in_dim_;
            copy_gpu(size,intput_data,1,output_data,1);
            return;
        }

        Tensor *rand_tensor = rand_.get();
        float *rand_data = (float *)rand_tensor->mutable_gpu_data();
        forward_dropout_gpu(batch_size_,in_dim_,intput_data,output_data,rand_data,probability_,scale_);
    } 
    void Dropout::BackwardGpu()
    {
        if(!is_train_){
            return;
        }
        Tensor *intput_tensor = prev_[0].get();
        Tensor *output_tensor = next_[0].get();
        Tensor *rand_tensor = rand_.get();
        float *output_diff = (float *)intput_tensor->gpu_diff();
        float *input_diff = (float *)output_tensor->mutable_gpu_diff();
        float *rand_data = (float *)rand_tensor->mutable_gpu_data();
        backward_dropout_gpu(batch_size_,in_dim_,input_diff,output_diff,rand_data,probability_,scale_);
    }
    void Dropout::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
 }
 
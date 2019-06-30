#include "javernn/op/binary_data.h"
#include "javernn/util/blas.h"
#include "javernn/log.h"

namespace javernn
{
    static std::string TAG = "BinaryData";
    BinaryData::BinaryData(std::string file_path, uint32_t batch_size, uint32_t data_dim_,uint32_t label_dim):
    Input(batch_size,data_dim_,label_dim),
    file_path_(file_path)
    {
        tmp_data_ = std::make_shared<char>(batch_size*data_dim_);
        tmp_label_ = std::make_shared<char>(batch_size*label_dim);
        file_in_.open(file_path_,std::ios::binary);
        file_in_.seekg(0,std::ios::end);
        file_size_ = file_in_.tellg();
        file_in_.seekg(0,std::ios::beg);
    }

    BinaryData::~BinaryData()
    {
        file_in_.close();
    }

    void BinaryData::Setup()
    {
        Log::v(TAG,"Setup");

    }

    void BinaryData::ForwardCpu()
    {
        Log::v(TAG,"ForwardCpu");
        auto tensors = GetOutputTensor();
        float *cpu_data = (float *)tensors[0]->mutable_cpu_data();
        float *cpu_label = (float *)tensors[1]->mutable_cpu_data();
        fill_cpu(batch_size_*data_dim_,0,cpu_data,1);
        fill_cpu(batch_size_*label_dim_,0,cpu_label,1);
        for(int i=0;i<batch_size_;i++){
            file_in_.read(tmp_label_.get(), label_dim_);
            for(int i=0;i<data_dim_;i++){
                cpu_data[i] = (float)tmp_data_.get()[i];
            }
            file_in_.read(tmp_data_.get(), data_dim_);
            for(int i=0;i<label_dim_;i++){
                cpu_label[i] = (float)tmp_data_.get()[i];
            }
        }
        
    } 

    void BinaryData::BackwardCpu()
    {
        Log::v(TAG,"BackwardCpu");
    }

    void BinaryData::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void BinaryData::ForwardGpu()
    {
        auto tensors = GetOutputTensor();
        float *cpu_data = (float *)tensors[0]->mutable_cpu_data();
        float *cpu_label = (float *)tensors[1]->mutable_cpu_data();
        fill_cpu(batch_size_*data_dim_,0,cpu_data,1);
        fill_cpu(batch_size_*label_dim_,0,cpu_label,1);
        for(int i=0;i<batch_size_;i++){
            file_in_.read(tmp_label_.get(), label_dim_);
            for(int i=0;i<data_dim_;i++){
                cpu_data[i] = (float)tmp_data_.get()[i];
            }
            file_in_.read(tmp_data_.get(), data_dim_);
            for(int i=0;i<label_dim_;i++){
                cpu_label[i] = (float)tmp_data_.get()[i];
            }
        }
        tensors[0]->data_to_gpu();
        tensors[1]->data_to_gpu();
    } 

    void BinaryData::BackwardGpu()
    {

    }

    void BinaryData::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
} // namespace javernnvoid Input::Setup()
  
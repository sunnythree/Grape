#include "javernn/op/binary_data.h"
#include "javernn/util/blas.h"
#include "javernn/log.h"

namespace javernn
{
    static std::string TAG = "BinaryData";
    BinaryData::BinaryData(std::string file_path, uint32_t batch_size, uint32_t in_dim,uint32_t data_offset):
    Input(batch_size,in_dim),
    file_path_(file_path),
    data_offset_(data_offset)
    {
        tmp_data_ = std::make_shared<char>(batch_size*in_dim);
        file_in_.open(file_path_,std::ios::binary);
        file_in_.seekg(0,std::ios::end);
        file_size_ = file_in_.tellg();
        file_in_.seekg(data_offset_,std::ios::beg);
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
        auto out_tensor = GetOutputTensor();
        float *cpu_data = (float *)out_tensor->mutable_cpu_data();
        fill_cpu(batch_size_*in_dim_,0,cpu_data,1);
        for(int i=0;i<batch_size_;i++){
            file_in_.read(tmp_data_.get(), in_dim_);
            for(int i=0;i<in_dim_;i++){
                cpu_data[i] = (float)tmp_data_.get()[i];
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
        auto out_tensor = GetOutputTensor();
        float *cpu_data = (float *)out_tensor->mutable_cpu_data();
        fill_cpu(batch_size_*in_dim_,0,cpu_data,1);
        for(int i=0;i<batch_size_;i++){
            file_in_.read(tmp_data_.get(), in_dim_);
            for(int i=0;i<in_dim_;i++){
                cpu_data[i] = (float)tmp_data_.get()[i];
            }
        }
        out_tensor->data_to_gpu();
    } 

    void BinaryData::BackwardGpu()
    {

    }

    void BinaryData::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
} // namespace javernnvoid Input::Setup()
  
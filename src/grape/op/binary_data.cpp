#include <chrono>
#include "grape/op/binary_data.h"
#include "grape/util/blas.h"
#include "grape/util/random.h"
#include "grape/log.h"
#include "grape/global_config.h"

namespace Grape
{
    const static std::string TAG = "BinaryData";

    BinaryData::BinaryData(std::string name, std::string file_path, uint32_t batch_size, uint32_t in_dim,
    uint32_t out_dim, uint32_t data_offset, bool one_hot):
    Op({},{DATA,DATA}),
    file_path_(file_path),
    data_offset_(data_offset),
    one_hot_(one_hot),
    in_dim_(in_dim),
    out_dim_(out_dim)
    {
        type_ = STRING_BINARY_DATA_TYPE;
        name_ = name;
        tmp_data_.reserve(in_dim);
        file_in_.open(file_path_,std::ios::binary);
        if(!file_in_.is_open()){
            Log::v(TAG,"file open error, path is "+file_path_);
        }
        Log::v(TAG,"file open sucess");
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
        auto out_tensor = next_[0];
//         float *cpu_data = (float *)out_tensor->mutable_cpu_data();
//         fill_cpu(batch_size_*out_dim_,0,cpu_data,1);
//         for(int i=0;i<batch_size_;i++){
//             unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
//             Random::GetInstance().SetSeed(seed);
//             int rand_point = Random::GetInstance().GetUniformInt(0,5000);
//             file_in_.seekg(data_offset_+rand_point*in_dim_,std::ios::beg);
//             if(one_hot_){
//                 file_in_.read(tmp_data_.data(), in_dim_);
//                 for(int ii=0;ii<in_dim_;ii++){
//                     int tmp = (int)(tmp_data_.data()[ii]&0xff);
// //                    Log::v(TAG,std::to_string(tmp));
//                     for(int j=0;j<out_dim_;j++){
//                         if(tmp == j)
//                             cpu_data[i*in_dim_*out_dim_+ii*out_dim_+j] = 1;
//                         else 
//                             cpu_data[i*in_dim_*out_dim_+ii*out_dim_+j] = 0;
//                         //std::cout<<cpu_data[i*in_dim_*out_dim_+ii*out_dim_+j]<<" ";
//                     }
//                     //std::cout<<std::endl;
//                 }
//             }else{
//                 file_in_.read(tmp_data_.data(), out_dim_);
//                 for(int i=0;i<out_dim_;i++){
//                     cpu_data[i] = ((float)(tmp_data_.data()[i]&0xff))/255.0;
//                     // std::cout<<(cpu_data[i]?1:0)<<" ";
//                     // if((i+1)%28==0){
//                     //     std::cout<<std::endl;
//                     // }
//                 }
//                 // std::cout<<std::endl;
//             }
//         }

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

    } 

    void BinaryData::BackwardGpu()
    {

    }

    void BinaryData::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
} // namespace Grapevoid Input::Setup()
  
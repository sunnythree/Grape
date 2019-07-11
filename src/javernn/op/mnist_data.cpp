#include <chrono>
#include "javernn/op/mnist_data.h"
#include "javernn/util/blas.h"
#include "javernn/util/random.h"
#include "javernn/log.h"

namespace javernn
{
    static std::string TAG = "MnistData";
    static uint32_t mnist_data_size = 784;
    static uint32_t mnist_label_size = 1;

    MnistData::MnistData(std::string data_path, std::string label_path, uint32_t batch_size):
    Op({},{DATA,DATA}),
    data_path_(data_path),
    label_path_(label_path),
    batch_size_(batch_size)
    {
        type_ = "MnistData";
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,784}),DATA,gNetMode);
        next_[1] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,10}),DATA,gNetMode);

        data_in_.open(data_path_,std::ios::binary);
        if(!data_in_.is_open()){
            Log::v(TAG,"file open error, path is "+data_path_);
        }
        Log::v(TAG,"data file open sucess");
        data_in_.seekg(0,std::ios::beg);
        label_in_.open(data_path_,std::ios::binary);
        if(!label_in_.is_open()){
            Log::v(TAG,"file open error, path is "+label_path_);
        }
        Log::v(TAG,"label file open sucess");
        label_in_.seekg(0,std::ios::beg);

        tmp_data_.reserve(mnist_data_size);
    }

    MnistData::~MnistData()
    {
        data_in_.close();
        label_in_.close();
    }

    void MnistData::Setup()
    {
        Log::v(TAG,"Setup");

    }

    void MnistData::ForwardCpu()
    {
        Log::v(TAG,"ForwardCpu");
        Tensor *data_tensor = next_[0].get();
        Tensor *label_tensor = next_[1].get();
        float *data = (float *)data_tensor->mutable_cpu_data();
        float *label = (float *)label_tensor->mutable_cpu_data();

        for(int i=0;i<batch_size_;i++){
            unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
            Random::GetInstance().SetSeed(seed);
            int rand_point = Random::GetInstance().GetUniformInt(0,5000);
            data_in_.seekg(rand_point*mnist_data_size,std::ios::beg);
            data_in_.read(tmp_data_.data(), mnist_data_size);
            for(int ii=0;ii<mnist_data_size;ii++){
                data[ii+i*mnist_data_size] = ((float)(tmp_data_.data()[i]&0xff))/255.0;
            }
            data_in_.seekg(rand_point*mnist_label_size,std::ios::beg);
            data_in_.read(tmp_data_.data(), mnist_label_size);
            int tmp = (int)(tmp_data_.data()[0]&0xff);
            for(int ii=0;ii<10;ii++){
                if(tmp == ii){
                    label[i*10+ii] = 1;
                }
                else {
                    label[i*10+ii] = 0;
                }
            }
        }

    } 

    void MnistData::BackwardCpu()
    {
        Log::v(TAG,"BackwardCpu");
    }

    void MnistData::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void MnistData::ForwardGpu()
    {
        auto out_tensor = GetOutputTensor();
        float *cpu_data = (float *)out_tensor->mutable_cpu_data();
        fill_cpu(batch_size_*out_dim_,0,cpu_data,1);
        for(int i=0;i<batch_size_;i++){
            file_in_.read(tmp_data_.data(), out_dim_);
            for(int i=0;i<out_dim_;i++){
                cpu_data[i] = (float)(tmp_data_.data()[i]&0xff);
                //Log::v("",std::to_string(cpu_data[i]));
            }
        }
        out_tensor->data_to_gpu();
    } 

    void MnistData::BackwardGpu()
    {

    }

    void MnistData::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
} // namespace javernnvoid Input::Setup()
  
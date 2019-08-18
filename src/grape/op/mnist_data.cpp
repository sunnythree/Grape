#include <chrono>
#include "grape/op/mnist_data.h"
#include "grape/util/blas.h"
#include "grape/util/random.h"
#include "grape/log.h"
#include "grape/global_config.h"

namespace Grape
{
    const static std::string TAG = "MnistData";
    
    static uint32_t mnist_data_size = 784;
    static uint32_t mnist_label_size = 1;

    MnistData::MnistData(std::string name, std::string data_path, 
        std::string label_path, uint32_t batch_size, bool random, uint32_t sample_count):
    Op({},{DATA,DATA}),
    data_path_(data_path),
    label_path_(label_path),
    batch_size_(batch_size),
    random_(random),
    sample_count_(sample_count)
    {
        type_ = STRING_MNIST_DATA_TYPE;
        name_ = name;
        next_[0] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,784}),DATA,sizeof(float));
        next_[1] = std::make_shared<Tensor>(static_cast<Op *>(this),
        Shape({batch_size_,10}),DATA,sizeof(float));

        data_in_.open(data_path_,std::ios::binary);
        if(!data_in_.is_open()){
            Log::v(TAG,"file open error, path is "+data_path_);
        }
        Log::v(TAG,"data file open sucess");
        data_in_.seekg(0,std::ios::beg);
        label_in_.open(label_path_,std::ios::binary);
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

    }



    void MnistData::ForwardCpu()
    {
        Tensor *data_tensor = next_[0].get();
        Tensor *label_tensor = next_[1].get();
        float *data = (float *)data_tensor->mutable_cpu_data();
        float *label = (float *)label_tensor->mutable_cpu_data();
        std::cout.setf(std::ios::fixed);//固定小数位数
        std::cout.setf(std::ios::showpoint);//显示小数点
        std::cout.precision(2);//设置输出精度为0位小数
        for(int i=0;i<batch_size_;i++){
            read_point_ = read_sequence[iter_cout*batch_size_+i];
            if(read_point_>=sample_count_){
                read_point_ = 0;
            }
            data_in_.seekg(read_point_*mnist_data_size+16,std::ios::beg);
            // std::cout<<"rand_point: "<<rand_point<<std::endl;
            data_in_.read(tmp_data_.data(), mnist_data_size);
            for(int ii=0;ii<mnist_data_size;ii++){
                data[ii+i*mnist_data_size] = ((float)(tmp_data_[ii]&0xff))/256.0;
                // std::cout<<(tmp_data_[ii]?1:0)<<" ";
                // if((ii+1)%28==0){
                //     std::cout<<std::endl;
                // }
            }
            // std::cout<<std::endl;
            char label_read = 0;
            label_in_.seekg(read_point_*mnist_label_size+8,std::ios::beg);
            label_in_.read(&label_read, mnist_label_size);
            // std::cout<<"label: "<<std::to_string(label_read)<<std::endl;
            // std::cout<<std::endl;
            for(int ii=0;ii<10;ii++){
                if(label_read == ii){
                    label[i*10+ii] = 1;
                }
                else {
                    label[i*10+ii] = 0;
                }
            }
        }
        iter_cout++;
    } 

    void MnistData::BackwardCpu()
    {
    }

    void MnistData::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void MnistData::ForwardGpu()
    {
        ForwardCpu();
    } 

    void MnistData::BackwardGpu()
    {

    }

    void MnistData::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif

    void MnistData::gen_sequence()
    {
        read_sequence.resize(sample_count_);
        read_sequence.shrink_to_fit();
        for(int i=0;i<sample_count_;i++){
            read_sequence[i] = i;
        }
        if(random_){
            std::random_shuffle(read_sequence.begin(),read_sequence.end());
        }
    }

    void MnistData::OnTrainBegin()
    {
        gen_sequence();
        iter_cout = 0;
    }

    void MnistData::OnTestBegin()
    {
        gen_sequence();
        iter_cout = 0;
    }
} // namespace Grapevoid Input::Setup()
  
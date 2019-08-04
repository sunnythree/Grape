#ifndef __GRAPE_MNIST_DATA_H__
#define __GRAPE_MNIST_DATA_H__

#include <fstream>
#include "grape/op/input.h"

namespace Grape
{
    class MnistData: public Op
    {
    public:
        MnistData(std::string name, std::string data_path, 
            std::string label_path, uint32_t batch_size,
            bool random, uint32_t sample_count);
        virtual ~MnistData();

        virtual void Setup();
    
        virtual void ForwardCpu(); 
        virtual void BackwardCpu();
        virtual void UpdateWeightsCpu(Optimizer &opt);

#ifdef GPU
        virtual void ForwardGpu(); 
        virtual void BackwardGpu();
        virtual void UpdateWeightsGpu(Optimizer &opt);
#endif
        virtual void OnTrainBegin();
        virtual void OnTestBegin();
    private:
        void gen_sequence();
        std::string data_path_ = "";
        std::string label_path_ = "";
        std::ifstream data_in_;
        std::ifstream label_in_;
        std::vector<char> tmp_data_;
        uint32_t batch_size_ = 0;
        bool random_ = true;
        uint32_t sample_count_ = 0;
        uint32_t read_point_ = 0;
        uint32_t iter_cout = 0;
        std::vector<uint32_t> read_sequence;
    };
    
} // namespace Grape

#endif
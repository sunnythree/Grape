#ifndef __JAVERNN_BINARY_DATA_H__
#define __JAVERNN_BINARY_DATA_H__

#include <fstream>
#include "javernn/op/input.h"

namespace javernn
{
    class BinaryData: public Input
    {
    public:
        BinaryData(std::string file_path, uint32_t batch_size, uint32_t data_dim_,uint32_t label_dim);
        virtual ~BinaryData();

        virtual void Setup();
    
        virtual void ForwardCpu(); 
        virtual void BackwardCpu();
        virtual void UpdateWeightsCpu(Optimizer &opt);

#ifdef GPU
        virtual void ForwardGpu(); 
        virtual void BackwardGpu();
        virtual void UpdateWeightsGpu(Optimizer &opt);
#endif
    private:
        std::string file_path_;
        std::string file_size_;
        std::ifstream file_in_;
        std::shared_ptr<char> tmp_data_;
        std::shared_ptr<char> tmp_label_;
    };
    
} // namespace javernn

#endif
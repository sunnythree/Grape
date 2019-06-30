#ifndef __JAVERNN_BINARY_DATA_H__
#define __JAVERNN_BINARY_DATA_H__

#include <fstream>
#include "javernn/op/input.h"

namespace javernn
{
    class BinaryData: public Input
    {
    public:
        BinaryData(std::string file_path, uint32_t batch_size, uint32_t in_dim, uint32_t data_offset);
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
        std::vector<char> tmp_data_;
        uint32_t data_offset_;
    };
    
} // namespace javernn

#endif
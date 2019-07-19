#ifndef __GRAPE_BINARY_DATA_H__
#define __GRAPE_BINARY_DATA_H__

#include <fstream>
#include "grape/op.h"

namespace Grape
{
    class BinaryData : public Op
    {
    public:
        BinaryData(std::string name, std::string file_path, uint32_t batch_size, uint32_t in_dim,uint32_t out_dim, 
        uint32_t data_offset, bool one_hot = false);
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
        bool one_hot_;
        uint32_t out_dim_;
        uint32_t in_dim_;
    };
    
} // namespace Grape

#endif
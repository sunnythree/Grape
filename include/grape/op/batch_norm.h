#ifndef __GRAPE_BATCH_NORM_H__
#define __GRAPE_BATCH_NORM_H__

#include "grape/op.h"

namespace Grape
{
    class BatchNorm : public Op
    {
    public:
        BatchNorm(std::string name, std::string file_path, uint32_t batch_size, uint32_t in_dim,uint32_t out_dim, 
        uint32_t data_offset, bool one_hot = false);
        virtual ~BatchNorm();

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
        uint32_t data_offset_;
        bool one_hot_;
        uint32_t out_dim_;
        uint32_t in_dim_;
    };
    
} // namespace Grape

#endif
#ifndef __GRAPE_POOL_MEAN_H__
#define __GRAPE_POOL_MEAN_H__

#include "grape/op.h"


namespace Grape{
    class PoolMean : public Op{
    public:
        PoolMean() = delete;
        explicit PoolMean(
            std::string name, 
            uint32_t batch_size,
            uint32_t in_w,
            uint32_t in_h,
            uint32_t in_c
            );
        virtual ~PoolMean();
        void Setup();
        void ForwardCpu(); 
        void BackwardCpu();
        void UpdateWeightsCpu(Optimizer &opt);

#ifdef GPU
        void ForwardGpu(); 
        void BackwardGpu();
        void UpdateWeightsGpu(Optimizer &opt);
#endif


    private:
        uint32_t batch_size_;
        uint32_t in_w_;
        uint32_t in_h_;
        uint32_t in_c_;
    };
}

#endif
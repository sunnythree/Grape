#ifndef __GRAPE_POOL_MEAN_H__
#define __GRAPE_POOL_MEAN_H__

#include "grape/op.h"


namespace Grape{
    class PoolAvg : public Op{
    public:
        PoolAvg() = delete;
        explicit PoolAvg(
            std::string name, 
            uint32_t batch_size,
            uint32_t in_c,
            uint32_t in_h,
            uint32_t in_w
            );
        virtual ~PoolAvg();
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
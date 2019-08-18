#ifndef __GRAPE_POOL_MAX_H__
#define __GRAPE_POOL_MAX_H__

#include "grape/op.h"


namespace Grape{
    class PoolMax : public Op{
    public:
        PoolMax() = delete;
        explicit PoolMax(
            std::string name, 
            uint32_t batch_size,
            uint32_t in_c,
            uint32_t in_h,
            uint32_t in_w,
            uint32_t ksize,
            uint32_t stride,
            uint32_t padding
            );
        virtual ~PoolMax();
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
        uint32_t out_w_;
        uint32_t out_h_;
        uint32_t out_c_;
        uint32_t ksize_;
        uint32_t stride_;
        uint32_t padding_;
    };
}

#endif
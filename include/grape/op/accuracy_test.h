#ifndef __GRAPE_ACCURACY_TEST_H__
#define __GRAPE_ACCURACY_TEST_H__

#include "grape/op.h"

namespace Grape{
    class AccuracyTest : public Op{
    public:
        AccuracyTest() = delete;
        AccuracyTest(std::string name, uint32_t batch_size, uint32_t in_dim);
        ~AccuracyTest();
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
        uint32_t batch_size_ = 0;
        uint32_t in_dim_ = 0;
        float accuracy_ = 0.f;
        uint32_t accuracy_count_ = 0;
        uint32_t all_count = 0;
    };
}
#endif
#ifndef __GRAPE_ACCURACY_TEST_H__
#define __GRAPE_ACCURACY_TEST_H__

#include "grape/op.h"

namespace Grape{
    class AccuracyTest : public Op{
    public:
        AccuracyTest() = delete;
        AccuracyTest(std::string name, uint32_t batch_size, uint32_t in_dim);
        ~AccuracyTest();
        virtual void Setup();
        virtual void ForwardCpu(); 
        virtual void BackwardCpu();
        virtual void UpdateWeightsCpu(Optimizer &opt);
        virtual void OnTestBegin();
        virtual void Display();

#ifdef GPU
        virtual void ForwardGpu(); 
        virtual void BackwardGpu();
        virtual void UpdateWeightsGpu(Optimizer &opt);
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
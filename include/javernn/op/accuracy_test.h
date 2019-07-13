#ifndef __JAVERNN_ACCURACY_TEST_H__
#define __JAVERNN_ACCURACY_TEST_H__

#include "javernn/op.h"

namespace javernn{
    class AccuracyTest :protected Op{
    public:
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
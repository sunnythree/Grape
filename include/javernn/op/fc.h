#ifndef __JAVERNN_FC_H__
#define __JAVERNN_FC_H__
#include "javernn/op.h"
namespace javernn{
    class Fc:public Op{
    public:
        Fc() = delete;
        explicit Fc(uint32_t batch_size,uint32_t in_dim,uint32_t out_dim,bool has_bias = true);
        virtual ~Fc();
        void SetupCpu(bool reset_weight);
        std::vector<Tensor> ForwardCpu(); 
        void BackwardCpu();
        void UpdateWeightsCpu(Optimizer &opt);

#ifdef GPU
        void SetupGpu(bool reset_weight);
        std::vector<Tensor> ForwardGpu(); 
        void BackwardGpu();
        void UpdateWeightsGpu(Optimizer &opt);
#endif
    private:
        uint32_t batch_size_;
        uint32_t in_dim_;
        uint32_t out_dim_;
        bool has_bias_;
    };
}

#endif
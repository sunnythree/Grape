#ifndef __JAVERNN_FC_H__
#define __JAVERNN_FC_H__

#include "javernn/op.h"
namespace javernn{
    class Fc:public Op{
    public:
        Fc() = delete;
        explicit Fc(std::string name, uint32_t batch_size,uint32_t in_dim,uint32_t out_dim,bool has_bias = true);
        virtual ~Fc();
        void Setup();
        void ForwardCpu(); 
        void BackwardCpu();
        void UpdateWeightsCpu(Optimizer &opt);

#ifdef GPU
        void ForwardGpu(); 
        void BackwardGpu();
        void UpdateWeightsGpu(Optimizer &opt);
#endif
        void Save(std::string path, SERIALIZE_TYPE type);
        void Load(std::string path, SERIALIZE_TYPE type);

    private:
        uint32_t batch_size_;
        uint32_t in_dim_;
        uint32_t out_dim_;
        bool has_bias_;
    };
}

#endif
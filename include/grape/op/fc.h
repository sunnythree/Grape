#ifndef __GRAPE_FC_H__
#define __GRAPE_FC_H__

#include "grape/op.h"
#include "grape/util/activations.h"
#include "cereal/archives/json.hpp"


namespace Grape{
    class Fc : public Op{
    public:
        Fc() = delete;
        explicit Fc(
            std::string name, 
            uint32_t batch_size,
            uint32_t in_dim,
            uint32_t out_dim,
            bool has_bias = true,
            ACTIVATION activation = LEAKY
            );
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

        void Load(cereal::BinaryInputArchive &archive);
        void Load(cereal::JSONInputArchive &archive);
        void Load(cereal::XMLInputArchive &archive);
        void Save(cereal::BinaryOutputArchive &archive);
        void Save(cereal::JSONOutputArchive &archive);
        void Save(cereal::XMLOutputArchive &archive);


    private:
        uint32_t batch_size_;
        uint32_t in_dim_;
        uint32_t out_dim_;
        bool has_bias_ = true;
        ACTIVATION activation_ = LEAKY;
        bool setuped_ = false;
    };
}

#endif
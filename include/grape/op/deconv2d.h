#ifndef __GRAPE_DECONV2D_H__
#define __GRAPE_DECONV2D_H__

#include "grape/op.h"
#include "grape/util/activations.h"
#include "cereal/archives/json.hpp"


namespace Grape{
    class DeConv2d : public Op{
    public:
        DeConv2d() = delete;
        explicit DeConv2d(
            std::string name, 
            uint32_t batch_size,
            uint32_t in_c,
            uint32_t in_h,
            uint32_t in_w,
            uint32_t out_c,
            uint32_t group,
            uint32_t ksize,
            uint32_t stride,
            uint32_t padding,
            bool has_bias,
            ACTIVATION activation
            );
        virtual ~DeConv2d();
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
        uint32_t in_w_;
        uint32_t in_h_;
        uint32_t in_c_;
        uint32_t out_w_;
        uint32_t out_h_;
        uint32_t out_c_;
        uint32_t group_;
        uint32_t ksize_;
        uint32_t stride_;
        uint32_t padding_;
        bool has_bias_ = true;
        ACTIVATION activation_ = LEAKY;
        bool setuped_ = false;
        std::shared_ptr<Tensor> im_col_tensor_;
        uint32_t noutputs_;
        uint32_t nweights_;
    };
}

#endif
#include "grape/op/conv2d.h"

namespace Grape
{
    const static std::string TAG = "Conv2d";
    const static std::string CONV_TYPE = "Conv2d";
    Conv2d::Conv2d(
            std::string name, 
            uint32_t batch_size,
            uint32_t in_w,
            uint32_t in_h,
            uint32_t in_c,
            uint32_t knum,
            uint32_t ksize,
            uint32_t stride,
            uint32_t padding,
            bool has_bias = true,
            ACTIVATION activation = LEAKY
            ):
        Op({DATA,WEIGHTS,BIAS}, {DATA}),
        batch_size_(batch_size),
        in_w_(in_w),
        in_h_(in_h),
        in_c_(in_c),
        knum_(knum),
        ksize_(ksize),
        stride_(stride),
        padding_(padding),
        has_bias_(has_bias),
        activation_(activation)
    {
        
    }
    Conv2d::~Conv2d()
    {
        
    }
    void Conv2d::Setup()
    {

    }

    void Conv2d::ForwardCpu()
    {

    } 

    void Conv2d::BackwardCpu()
    {

    }

    void Conv2d::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void Conv2d::ForwardGpu()
    {

    } 

    void Conv2d::BackwardGpu()
    {

    }

    void Conv2d::UpdateWeightsGpu(Optimizer &opt)
    {

    }

#endif

    void Conv2d::Load(cereal::BinaryInputArchive &archive)
    {

    }

    void Conv2d::Load(cereal::JSONInputArchive &archive)
    {

    }

    void Conv2d::Load(cereal::XMLInputArchive &archive)
    {

    }

    void Conv2d::Save(cereal::BinaryOutputArchive &archive)
    {

    }

    void Conv2d::Save(cereal::JSONOutputArchive &archive)
    {

    }

    void Conv2d::Save(cereal::XMLOutputArchive &archive)
    {

    }
} // namespace Grape


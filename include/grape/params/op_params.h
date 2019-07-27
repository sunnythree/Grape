#ifndef __GRAPE_OP_PARAMS_H__
#define __GRAPE_OP_PARAMS_H__

#include <string>
#include <cstdint>
#include "cereal/cereal.hpp"
#include "cereal/archives/json.hpp"
namespace Grape{
    class OpParams{
    public:
        std::string name_;
        std::string type_;
        uint32_t batch_;
        uint32_t in_dim_;
        uint32_t out_dim_;
        std::string data_path_;
        std::string label_path_;
        bool has_bias_;
        bool random_;
        uint32_t sample_count_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("name_",name_));
            ar(cereal::make_nvp("type_",type_));
            ar(cereal::make_nvp("batch_",batch_));
            ar(cereal::make_nvp("in_dim_",in_dim_));
            ar(cereal::make_nvp("out_dim_",out_dim_));
            ar(cereal::make_nvp("data_path_",data_path_));
            ar(cereal::make_nvp("label_path_",label_path_));
            ar(cereal::make_nvp("has_bias_",has_bias_));
            ar(cereal::make_nvp("random_",random_));
            ar(cereal::make_nvp("sample_count_",sample_count_));
        }
    };
}

#endif
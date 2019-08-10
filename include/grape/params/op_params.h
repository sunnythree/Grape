#ifndef __GRAPE_OP_PARAMS_H__
#define __GRAPE_OP_PARAMS_H__

#include <string>
#include <cstdint>
#include "cereal/cereal.hpp"
#include "cereal/archives/json.hpp"
namespace Grape{

    static const std::string ACTIVATION_NONE = "none";
    static const std::string ACTIVATION_LOGISTIC = "logistic";
    static const std::string ACTIVATION_RELU = "relu";
    static const std::string ACTIVATION_RELIE = "relie";
    static const std::string ACTIVATION_LINEAR = "linear";
    static const std::string ACTIVATION_RAMP = "ramp";
    static const std::string ACTIVATION_TANH = "tanh";
    static const std::string ACTIVATION_PLSE = "plse";
    static const std::string ACTIVATION_LEAKY = "leaky";
    static const std::string ACTIVATION_ELU = "elu";
    static const std::string ACTIVATION_LOGGY = "loggy";
    static const std::string ACTIVATION_STAIR = "stair";
    static const std::string ACTIVATION_HARDTAN = "hardtan";
    static const std::string ACTIVATION_LHTAN = "lhtan";
    static const std::string ACTIVATION_SELU = "selu";

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
        std::string activation_;
        

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("name",name_));
            ar(cereal::make_nvp("type",type_));
            try{
                ar(cereal::make_nvp("batch",batch_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("in_dim",in_dim_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("out_dim",out_dim_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("data_path",data_path_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("label_path",label_path_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("random",random_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("sample_count",sample_count_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("sample_count",sample_count_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("activation",activation_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
        }
    };
}

#endif
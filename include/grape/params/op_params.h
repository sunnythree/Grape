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
        //conv2d and pool
        uint32_t in_c_;
        uint32_t in_h_;
        uint32_t in_w_;
        uint32_t out_c_;
        uint32_t out_h_;
        uint32_t out_w_;
        uint32_t group_;
        uint32_t ksize_;
        uint32_t stride_;
        uint32_t padding_;
        //dropout
        float probability_ = 0.5;

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
                ar(cereal::make_nvp("has_bias",has_bias_));
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
                ar(cereal::make_nvp("activation",activation_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("in_c",in_c_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("in_h",in_h_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("in_w",in_w_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("out_c",out_c_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("out_h",out_h_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("out_w",out_w_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("group",group_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("ksize",ksize_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("stride",stride_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("padding",padding_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("probability",probability_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
        }
    };
}

#endif
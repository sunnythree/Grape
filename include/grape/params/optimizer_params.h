#ifndef __GRAPE_OPTIMIZER_PARAMS_H__
#define __GRAPE_OPTIMIZER_PARAMS_H__

#include <string>
#include "grape/global_config.h"
#include "cereal/archives/json.hpp"
#include "cereal/types/vector.hpp"

namespace Grape{
    static const std::string POLICY_FIXED_STRING = "fixed";
    static const std::string POLICY_STEP_STRING = "step";
    static const std::string POLICY_MUTISTEP_STRING = "mutistep";
    static const std::string OPTIMIZER_TYPE_SGD = "sgd";
    static const std::string OPTIMIZER_TYPE_ADAM = "adam";
    class OptimizerParams{
    public:
        std::string graph_name_;
        std::string type_;
        float lr_;
        float decay_ = 0;  // weight decay
        std::string policy_ = POLICY_FIXED_STRING;
        uint32_t step_ = 1;
        std::vector<uint32_t> muti_step_;
        float gamma_ = 0;
        float momentum_ = 0;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( cereal::make_nvp("graph_name",graph_name_));

            try{
                ar( cereal::make_nvp("type",type_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("lr",lr_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("decay",decay_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("policy",policy_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("step",step_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("muti_step",muti_step_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("gamma",gamma_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar(cereal::make_nvp("momentum",momentum_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
        }
    };
}

#endif
#ifndef __GRAPE_GRAPH_PARAMS_H__
#define __GRAPE_GRAPH_PARAMS_H__

#include <string>
#include <cstdint>
#include <vector>
#include "op_params.h"
#include "optimizer_params.h"
#include "grape/global_config.h"

namespace Grape{
    static const std::string CAL_MODE_STRING = "cal_mode";
    static const std::string CAL_MODE_CPU_STRING = "cpu";
    static const std::string CAL_MODE_GPU_STRING = "gpu";
    static const std::string PHASE_STRING = "phase";
    static const std::string PHASE_TRAIN_STRING = "train";
    static const std::string PHASE_TEST_STRING = "test";
    static const std::string SERIALIZE_TYPE_STRING = "serialize_type";
    static const std::string SERIALIZE_TYPE_BINARY_STRING = "binary";
    static const std::string SERIALIZE_TYPE_JSON_STRING = "json";
    static const std::string SERIALIZE_TYPE_XML_STRING = "xml";
    class GraphParams{
    public:
        std::string name_ = "";
        uint32_t max_iter_ = 1;
        std::string cal_mode_ = CAL_MODE_CPU_STRING;
        std::string phase_ = PHASE_TRAIN_STRING;
        uint32_t device_id_ = 0;
        uint32_t display_iter_ = 100;
        uint32_t snapshot_iter_ = 10000;
        std::string save_path_ = "data/";
        std::string serialize_type_ = SERIALIZE_TYPE_BINARY_STRING;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( cereal::make_nvp("name",name_));
            try{
                ar( cereal::make_nvp("max_iter",max_iter_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("cal_mode",cal_mode_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("phase",phase_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("device_id",device_id_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("display_iter",display_iter_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("snapshot_iter",snapshot_iter_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("save_path",save_path_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("serialize_type",serialize_type_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
        }
    };
}

#endif
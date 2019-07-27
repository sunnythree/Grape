#ifndef __GRAPE_GRAPH_PARAMS_H__
#define __GRAPE_GRAPH_PARAMS_H__

#include <string>
#include <cstdint>
#include <vector>
#include "op_params.h"
#include "optimizer_params.h"
#include "grape/global_config.h"

namespace Grape{
    class GraphParams{
    public:
        std::string name_ = "";
        uint32_t max_iter_ = 1;
        CAL_MODE cal_mode_ = CPU_MODE;
        PHASE phase_ = TRAIN;
        uint32_t device_id_ = 0;
        std::string save_path_ = "data/";
        SERIALIZE_TYPE serialize_type_ = BINARY;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( cereal::make_nvp("name_",name_));
            try{
                ar( cereal::make_nvp("max_iter_",max_iter_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("cal_mode_",cal_mode_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("phase_",phase_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("device_id_",device_id_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
            try{
                ar( cereal::make_nvp("save_path_",save_path_));
            }  catch(cereal::Exception&){
                ar.setNextName(nullptr);
            }
        }
    };
}

#endif
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
        std::string name_;
        uint32_t max_iter_ = 1;
        CAL_MODE cal_mode_ = CPU_MODE;
        PHASE phase_ = TRAIN;
        uint32_t device_id_ = 0;
        OptimizerParams optimizer_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( cereal::make_nvp("name_",name_));
            ar( cereal::make_nvp("max_iter_",max_iter_));
            ar( cereal::make_nvp("cal_mode_",cal_mode_));
            ar( cereal::make_nvp("phase_",phase_));
            ar( cereal::make_nvp("device_id_",device_id_));
            ar( cereal::make_nvp("optimizer_",optimizer_));
        }
    };
}

#endif
#ifndef __GRAPE_OPTIMIZER_PARAMS_H__
#define __GRAPE_OPTIMIZER_PARAMS_H__

#include <string>
#include "grape/global_config.h"
#include "cereal/archives/json.hpp"

namespace Grape{
    class OptimizerParams{
    public:
        std::string graph_name_;
        OPTIMIZER_TYPE type_;
        float lr_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( cereal::make_nvp("graph_name",graph_name_));
            ar( cereal::make_nvp("type",type_));
            ar( cereal::make_nvp("lr",lr_));
        }
    };
}

#endif
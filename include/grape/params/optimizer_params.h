#ifndef __GRAPE_OPTIMIZER_PARAMS_H__
#define __GRAPE_OPTIMIZER_PARAMS_H__

#include <string>
#include "grape/global_config.h"
#include "cereal/archives/json.hpp"

namespace Grape{
    class OptimizerParams{
    public:
        OPTIMIZER_TYPE type_;
        float lr_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( cereal::make_nvp("type_",type_));
            ar( cereal::make_nvp("lr_",lr_));
        }
    };
}

#endif
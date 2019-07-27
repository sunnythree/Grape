#ifndef __GRAPE_OPTIMIZER_LIST_PARAMS_H__
#define __GRAPE_OPTIMIZER_LIST_PARAMS_H__

#include <string>
#include <map>
#include "optimizer_params.h"

namespace Grape{
    class OptimizerListParams{
    public:
        std::map<std::string,OptimizerParams> optimizer_mapL;
        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( cereal::make_nvp("optimizer_mapL",optimizer_mapL));
        }
    };
}

#endif
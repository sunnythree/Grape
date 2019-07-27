#ifndef __GRAPE_OPTIMIZER_LIST_PARAMS_H__
#define __GRAPE_OPTIMIZER_LIST_PARAMS_H__

#include <string>
#include <vector>
#include "optimizer_params.h"

namespace Grape{
    class OptimizerListParams{
    public:
        std::vector<OptimizerParams> optimizer_list_;
        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( cereal::make_nvp("optimizer_list",optimizer_list_));
        }
    };
}

#endif
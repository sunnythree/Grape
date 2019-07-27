#ifndef __GRAPE_OP_PATH_PARAMS_H__
#define __GRAPE_OP_PATH_PARAMS_H__

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include "op_params.h"
#include "cereal/archives/json.hpp"
#include "cereal/types/map.hpp"

namespace Grape{

    class OpPathParams{
    public:
        std::map<std::string,std::string> path_list_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("path_list",path_list_));
        }
    };
}

#endif
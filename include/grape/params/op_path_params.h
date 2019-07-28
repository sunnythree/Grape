#ifndef __GRAPE_OP_PATH_PARAMS_H__
#define __GRAPE_OP_PATH_PARAMS_H__

#include <cstdint>
#include <string>
#include <vector>
#include <tuple>
#include "op_params.h"
#include "cereal/archives/json.hpp"
#include "cereal/types/map.hpp"

namespace Grape{

    typedef struct name_path_t
    {
        std::string name;
        std::string path;
        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("name",name));
            ar(cereal::make_nvp("path",path));
        }
    }NamePathPair;
    

    class OpPathParams{
    public:
        std::vector<NamePathPair> path_list_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("path_list",path_list_));
        }
    };
}

#endif
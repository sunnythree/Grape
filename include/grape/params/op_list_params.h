#ifndef __GRAPE_OP_LIST_PARAMS_H__
#define __GRAPE_OP_LIST_PARAMS_H__

#include <cstdint>
#include <string>
#include <map>
#include "cereal/archives/json.hpp"
#include "cereal/types/map.hpp"

namespace Grape{

    class OpListParams{
    public:
        std::map<std::string,std::string> op_list_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("op_list_",op_list_));
        }
    };
}

#endif
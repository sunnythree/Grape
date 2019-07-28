#ifndef __GRAPE_OP_LIST_PARAMS_H__
#define __GRAPE_OP_LIST_PARAMS_H__

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include "op_params.h"
#include "cereal/archives/json.hpp"
#include "cereal/types/vector.hpp"

namespace Grape{

    class OpListParams{
    public:
        std::vector<OpParams> op_list_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("op_list",op_list_));
        }
    };
    
}

#endif
#ifndef __GRAPE_CONNECTION_LIST_PARAMS_H__
#define __GRAPE_CONNECTION_LIST_PARAMS_H__

#include <cstdint>
#include <string>
#include <vector>
#include "cereal/archives/json.hpp"
#include "cereal/types/map.hpp"
#include "grape/params/connection_params.h"


namespace Grape{

    class ConnectionListParams{
    public:
        std::vector<ConnectionParams> connection_list_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("connection_list",connection_list_));
        }
    };
}

#endif
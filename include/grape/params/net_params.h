#ifndef __GRAPE_NET_PARAMS_H__
#define __GRAPE_NET_PARAMS_H__

#include <cstdint>
#include <string>
#include <vector>
#include "grape/global_config.h"
#include "cereal/archives/json.hpp"
#include "graph_params.h"

namespace Grape{

    class NetParams{
    public:
        uint32_t max_iter_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("max_iter",max_iter_));
        }
    };
}

#endif
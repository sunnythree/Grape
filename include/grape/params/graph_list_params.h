#ifndef __GRAPE_GRAPH_LIST_PARAMS_H__
#define __GRAPE_GRAPH_LIST_PARAMS_H__

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include "cereal/archives/json.hpp"
#include "cereal/types/map.hpp"
#include "graph_params.h"

namespace Grape{

    class GraphListParams{
    public:
        std::vector<GraphParams> graph_list_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("graph_list",graph_list_));
        }
    };
}

#endif
#ifndef __GRAPE_CONNECTION_PARAMS_H__
#define __GRAPE_CONNECTION_PARAMS_H__

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include "cereal/types/tuple.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/types/map.hpp"


namespace Grape{

    class ConnectionParams{
    public:
        std::string op_list_name_ = "";
        std::string graph_name_  = "";
        std::vector<std::tuple<std::string,std::string>> cnnections_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("op_list_name_",op_list_name_));
            ar(cereal::make_nvp("graph_name_",graph_name_));
            ar(cereal::make_nvp("cnnections_",cnnections_));
        }
    };
}

#endif
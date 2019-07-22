#ifndef __GRAPE_CONNECTION_PARAMS_H__
#define __GRAPE_CONNECTION_PARAMS_H__

#include <cstdint>
#include <string>
#include <map>
#include "cereal/archives/json.hpp"
#include "cereal/types/map.hpp"


namespace Grape{

    class ConnectionParams{
    public:
        std::string name_ = "";
        std::string graph_ = "";
        std::map<std::string,std::string> cnnections_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("name_",name_));
            ar(cereal::make_nvp("cnnections_",cnnections_));
        }
    };
}

#endif
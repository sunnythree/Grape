#ifndef __JAVERNN_NET_PARAMS_H__
#define __JAVERNN_NET_PARAMS_H__

#include <cstdint>
#include <string>
#include "javernn/global_config.h"
#include "cereal/archives/json.hpp"

namespace javernn{

    class NetParams{
    public:
        std::string net_type_;
        std::string cal_type_;
        std::string optimizer_type_;
        uint32_t device_id_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( 
                cereal::make_nvp("net_type_",net_type_),
                cereal::make_nvp("cal_type_",cal_type_),
                cereal::make_nvp("optimizer_type_",optimizer_type_),
                cereal::make_nvp("device_id_",device_id_)
            );
        }
    };
}

#endif
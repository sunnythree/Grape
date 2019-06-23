#ifndef __JAVERNN_NET_PARAMS_TEST_H__
#define __JAVERNN_NET_PARAMS_TEST_H__

#include <string>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include "cereal/archives/json.hpp"
#include "javernn/params/net_params.h"

namespace javernn{

    static void parse_net_params()
    {
        std::ifstream in("src/test/net_params.json");
        NetParams net;
        {
            cereal::JSONInputArchive archive(in);
            archive(net);
        }
        std::cout<<net.cal_type_<<" : "<<net.net_type_<<" : "<<net.optimizer_type_<<" : "<<net.device_id_<<std::endl;
    }

    static void serialize_net_params()
    {
        std::ofstream out("net_params.json",std::ios::trunc);
        {
            cereal::JSONOutputArchive archive(out);
            NetParams net;
            net.cal_type_ = "gpu";
            net.net_type_ = "sequence";
            net.optimizer_type_ = "sgd";
            net.device_id_ = 100;
        
            archive(cereal::make_nvp("net",net));
        }
    }
}

#endif
#ifndef __JAVERNN_NET_PARAMS_H__
#define __JAVERNN_NET_PARAMS_H__

#include <cstdint>
#include <string>
#include "javernn/global_config.h"

namespace javernn{

    class NetParams{
    public:
        NET_TYPE net_type_;
        CAL_MODE cal_type_;
        OPTIMIZER_TYPE optimizer_type_;
        uint32_t device_id_;
    private:
        std::string str_net_type_;
        std::string str_cal_mode_;
        std::string str_optimizer_type_;
    };
}

#endif
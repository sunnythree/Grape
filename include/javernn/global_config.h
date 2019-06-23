#ifndef __JAVERNN_GLOBAL_CONFIG_H__
#define __JAVERNN_GLOBAL_CONFIG_H__
#include <cstdint>
namespace javernn{
    enum CAL_MODE{
        CPU_MODE,
        GPU_MODE
    };

    enum NET_TYPE{
        NET_SEQUENCE,
        NET_GRAPH
    };

    enum OPTIMIZER_TYPE{
        SGD,
        SGDM,
        ADAM
    };

    static CAL_MODE gNetMode = CPU_MODE;
    static uint32_t gDeviceId = 0;
}

#endif
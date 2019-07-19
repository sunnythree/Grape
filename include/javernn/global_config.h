#ifndef __JAVERNN_GLOBAL_CONFIG_H__
#define __JAVERNN_GLOBAL_CONFIG_H__
#include <cstdint>
namespace javernn{
    enum CAL_MODE{
        CPU_MODE,
        GPU_MODE
    };

    enum GRAPH_PHASE{
        GRAPH_TRAIN,
        GRAPH_TEST
    };

    enum OPTIMIZER_TYPE{
        SGD,
        SGDM,
        ADAM
    };

}

#endif
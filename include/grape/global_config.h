#ifndef __GRAPE_GLOBAL_CONFIG_H__
#define __GRAPE_GLOBAL_CONFIG_H__
#include <cstdint>
namespace Grape{
    enum CAL_MODE{
        CPU_MODE,
        GPU_MODE
    };

    enum PHASE{
        TRAIN,
        TEST
    };

    enum OPTIMIZER_TYPE{
        SGD,
        ADAM
    };

    enum SERIALIZE_TYPE{
        BINARY,
        JSON,
        XML
    };

}

#endif
#ifndef __GRAPE_GLOBAL_CONFIG_H__
#define __GRAPE_GLOBAL_CONFIG_H__
#include <cstdint>
#include <string>
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
    const static std::string STRING_ACCURACY_TEST_TYPE = "AccuracyTest";
    const static std::string STRING_BINARY_DATA_TYPE = "BinaryData";
    const static std::string STRING_CONV2D_TYPE = "Conv2d";
    const static std::string STRING_FC_TYPE = "Fc";
    const static std::string STRING_INPUT_TYPE = "Input";
    const static std::string STRING_MNIST_DATA_TYPE = "MnistData";
    const static std::string STRING_POOL_MAX_TYPE = "PoolMax";
    const static std::string STRING_POOL_AVG_TYPE = "PoolAvg";
    const static std::string STRING_SOFTMAX_WITH_LOSS_TYPE = "SoftmaxWithLoss";
    const static std::string STRING_SOFTMAX_TYPE = "Softmax";
    const static std::string STRING_DROPOUT_TYPE = "Dropout";
}

#endif
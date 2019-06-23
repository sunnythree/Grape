#ifndef __JAVERNN_OP_PARAMS_H__
#define __JAVERNN_OP_PARAMS_H__

#include <string>
#include <cstdint>

namespace javernn{
    class OpParams{
    public:
        std::string name;
        //fc
        uint32_t fc_in_dim;
        uint32_t fc_out_dim;
        //conv2d
        uint32_t conv2d_batch;
        uint32_t conv2d_channel;
        uint32_t conv2d_height;
        uint32_t conv2d_width;
    };
}

#endif
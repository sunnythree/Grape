#ifndef __JAVERNN_OP_PARAMS_H__
#define __JAVERNN_OP_PARAMS_H__

#include <string>
#include <cstdint>

namespace javernn{
    class FcParams{
    public:
        uint32_t in_dim;
        uint32_t out_dim;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( in_dim, out_dim );
        }
    };
}

#endif
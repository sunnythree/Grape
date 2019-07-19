#ifndef __GRAPE_OP_PARAMS_H__
#define __GRAPE_OP_PARAMS_H__

#include <string>
#include <cstdint>

namespace Grape{
    class OpParams{
    public:
        std::string name;
        std::string connect_to;
        std::string params;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( name, connect_to, params );
        }
    };
}

#endif
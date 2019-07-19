#ifndef __GRAPE_OPTIMIZER_PARAMS_H__
#define __GRAPE_OPTIMIZER_PARAMS_H__

#include <string>

namespace Grape{
    class OptimizerParams{
    public:
        std::string type_;
        float lr;
        template <class Archive>
        void serialize( Archive & ar )
        {
            ar( type_, lr);
        }
    };
}

#endif
#ifndef __GRAPE_OPTIMIZER_FACTORY_H__
#define __GRAPE_OPTIMIZER_FACTORY_H__

#include <memory>
#include <vector>
#include "grape/params/optimizer_params.h"
#include "grape/optimizer/optimizer.h"


namespace Grape{

    class OptimizerFactory{
    public:
        static std::shared_ptr<Optimizer> Build(OptimizerParams& gp);
        static std::vector<std::shared_ptr<Optimizer>> Build(std::vector<OptimizerParams> gps);
    };
}

#endif
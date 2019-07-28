#include <memory>
#include "grape/optimizer_factory.h"
#include "grape/optimizer/sgd.h"

namespace Grape
{
    std::shared_ptr<Optimizer> OptimizerFactory::Build(OptimizerParams& gp)
    {
        switch (gp.type_)
        {
        case SGD:
            return std::make_shared<SGDOptimizer>(gp.lr_);
            break;
        case SGDM:
            break;
        default:
            break;
        }
        return nullptr;
    }

    std::vector<std::shared_ptr<Optimizer>> OptimizerFactory::Build(std::vector<OptimizerParams> opts)
    {
        std::vector<std::shared_ptr<Optimizer>> optimizers;
        for(int i=0;i<opts.size();++i){
            std::shared_ptr<Optimizer> opt = Build(opts[i]);
            optimizers.emplace_back(opt);
        }
        return optimizers;
    }
} // namespace Grape        
#include <memory>
#include "grape/optimizer_factory.h"
#include "grape/optimizer/sgd.h"

namespace Grape
{
    std::shared_ptr<Optimizer> OptimizerFactory::Build(OptimizerParams& opt)
    {
        OPTIMIZER_TYPE opt_type;
        if(opt.type_ == "sgd"){
            opt_type = SGD;
        }else if(opt.type_ == "adam"){
            opt_type = ADAM;
        }
        switch (opt_type)
        {
        case SGD:
            return std::make_shared<SGDOptimizer>(opt);
            break;
        case ADAM:
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
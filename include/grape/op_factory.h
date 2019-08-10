#ifndef __GRAPE_OP_FACTORY_H__
#define __GRAPE_OP_FACTORY_H__

#include <memory>
#include <vector>
#include <map>
#include "grape/op.h"
#include "grape/params/op_params.h"
#include "grape/util/activations.h"


namespace Grape{

    class OpFactory{
    public:
        static ACTIVATION GetActivationByString(std::string activation);
        static std::shared_ptr<Op> Build(OpParams& opp);
        static std::map<std::string,std::shared_ptr<Op>> Build(std::vector<OpParams> opps);
    };
}

#endif
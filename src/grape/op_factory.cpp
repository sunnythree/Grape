#include <map>
#include "grape/op_factory.h"
#include "grape/op/fc.h"
#include "grape/op/softmax.h"
#include "grape/op/softmax_with_loss.h"
#include "grape/op/accuracy_test.h"

namespace Grape
{
    std::shared_ptr<Op> OpFactory::Build(OpParams& opp)
    {
        std::shared_ptr<Op> bop;
        if(opp.type_ == "Fc"){
            bop = std::make_shared<Fc>(opp.name_,opp.batch_,opp.in_dim_,opp.out_dim_,opp.has_bias_);
        }else if(opp.type_ == "Softmax"){
            bop = std::make_shared<Softmax>(opp.name_,opp.batch_,opp.in_dim_);
        }else if(opp.type_ == "SoftmaxWithLoss"){
            bop = std::make_shared<SoftmaxWithLoss>(opp.name_,opp.batch_,opp.in_dim_);
        }else if(opp.type_ == "AccuracyTest"){
            bop = std::make_shared<AccuracyTest>(opp.name_,opp.batch_,opp.in_dim_);
        }
        return bop;
    }

    std::map<std::string,std::shared_ptr<Op>> OpFactory::Build(std::vector<OpParams> opps)
    {
        std::map<std::string,std::shared_ptr<Op>> op_map;
        for(int i=0;i<opps.size();i++){
            std::shared_ptr<Op> op_tmp = Build(opps[i]);
            op_map.insert(std::pair<std::string,std::shared_ptr<Op>>(op_tmp->get_name(),op_tmp));
        }
        return op_map;
    }
} // namespace Grape

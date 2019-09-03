#include <map>
#include "grape/op_factory.h"
#include "grape/op/fc.h"
#include "grape/op/softmax.h"
#include "grape/op/softmax_with_loss.h"
#include "grape/op/accuracy_test.h"
#include "grape/op/mnist_data.h"
#include "grape/op/pool_max.h"
#include "grape/op/pool_avg.h"
#include "grape/op/conv2d.h"
#include "grape/op/dropout.h"
#include "grape/log.h"
#include "grape/error.h"
#include "grape/global_config.h"

namespace Grape
{
    static std::string TAG = "OpFactory";
    ACTIVATION OpFactory::GetActivationByString(std::string activation)
    {
        if(activation.empty()){
            return NONE;
        }

        if(activation == ACTIVATION_NONE){
            return NONE;
        }else if(activation == ACTIVATION_LOGISTIC){
            return LOGISTIC;
        }else if(activation == ACTIVATION_RELU){
            return RELU;
        }else if(activation == ACTIVATION_RELIE){
            return RELIE;
        }else if(activation == ACTIVATION_LINEAR){
            return LINEAR;
        }else if(activation == ACTIVATION_RAMP){
            return RAMP;
        }else if(activation == ACTIVATION_TANH){
            return TANH;
        }else if(activation == ACTIVATION_PLSE){
            return PLSE;
        }else if(activation == ACTIVATION_LEAKY){
            return LEAKY;
        }else if(activation == ACTIVATION_ELU){
            return ELU;
        }else if(activation == ACTIVATION_LOGGY){
            return LOGGY;
        }else if(activation == ACTIVATION_STAIR){
            return STAIR;
        }else if(activation == ACTIVATION_HARDTAN){
            return HARDTAN;
        }else if(activation == ACTIVATION_LHTAN){
            return LHTAN;
        }else if(activation == ACTIVATION_SELU){
            return SELU;
        }else{
            throw NotimplementedError();
        }
    }

    std::shared_ptr<Op> OpFactory::Build(OpParams& opp)
    {
        std::shared_ptr<Op> bop;
        if(opp.type_ == "Fc"){
            bop = std::make_shared<Fc>(
                opp.name_,
                opp.batch_,
                opp.in_dim_,
                opp.out_dim_,
                opp.has_bias_,
                GetActivationByString(opp.activation_)
            );
        }else if(opp.type_ == STRING_SOFTMAX_TYPE){
            bop = std::make_shared<Softmax>(
                opp.name_,
                opp.batch_,
                opp.in_dim_
            );
        }else if(opp.type_ == STRING_SOFTMAX_WITH_LOSS_TYPE){
            bop = std::make_shared<SoftmaxWithLoss>(
                opp.name_,
                opp.batch_,
                opp.in_dim_
            );
        }else if(opp.type_ == STRING_ACCURACY_TEST_TYPE){
            bop = std::make_shared<AccuracyTest>(
                opp.name_,
                opp.batch_,
                opp.in_dim_
            );
        }else if(opp.type_ == STRING_MNIST_DATA_TYPE){
            bop = std::make_shared<MnistData>(
                opp.name_,
                opp.data_path_,
                opp.label_path_,
                opp.batch_,
                opp.random_,
                opp.sample_count_
            );
        }else if(opp.type_ == STRING_POOL_MAX_TYPE){
            bop = std::make_shared<PoolMax>(
                opp.name_,
                opp.batch_,
                opp.in_c_,
                opp.in_h_,
                opp.in_w_,
                opp.ksize_,
                opp.stride_,
                opp.padding_
            );
        }else if(opp.type_ == STRING_POOL_AVG_TYPE){
            bop = std::make_shared<PoolAvg>(
                opp.name_,
                opp.batch_,
                opp.in_c_,
                opp.in_h_,
                opp.in_w_
            );
        }else if(opp.type_ == STRING_CONV2D_TYPE){
            bop = std::make_shared<Conv2d>(
                opp.name_,
                opp.batch_,
                opp.in_c_,
                opp.in_h_,
                opp.in_w_,
                opp.out_c_,
                opp.group_,
                opp.ksize_,
                opp.stride_,
                opp.padding_,
                opp.has_bias_,
                GetActivationByString(opp.activation_)
            );
        }else if(opp.type_ == STRING_DROPOUT_TYPE){
            bop = std::make_shared<Dropout>(
                opp.name_,
                opp.batch_,
                opp.in_dim_,
                opp.probability_
            );
        }
        else{
            Log::e(TAG,"Op type not consist, you input is "+opp.type_);
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

#include "grape/optimizer/sgd.h"
#include "grape/util/blas.h"

namespace Grape
{
    SGDOptimizer::SGDOptimizer(float lr):
    lr_(lr)
    {

    }

    SGDOptimizer::SGDOptimizer(float lr, float decay):
    lr_(lr),
    decay_(decay)
    {
        
    }

    SGDOptimizer::SGDOptimizer(float lr,LR_POLICY policy,uint32_t step):
    lr_(lr),
    policy_(policy),
    step_(step)
    {

    }

    SGDOptimizer::SGDOptimizer(float lr,LR_POLICY policy,std::vector<uint32_t> muti_step):
    lr_(lr),
    policy_(policy),
    muti_step_(muti_step)
    {

    }

    SGDOptimizer::SGDOptimizer(OptimizerParams &optimizer_params)
    {
        lr_ = optimizer_params.lr_;
        if(optimizer_params.policy_==POLICY_FIXED_STRING){
            policy_ = POLICY_FIXED;
        }else if(optimizer_params.policy_==POLICY_STEP_STRING){
            policy_ = POLICY_STEP;
        }else if(optimizer_params.policy_==POLICY_STEP_STRING){
            policy_ = POLICY_MUTISTEP;
        }
        decay_ = optimizer_params.decay_;
        momentum_ = optimizer_params.momentum_;
        gamma_ = optimizer_params.gamma_;
        step_ = optimizer_params.step_;
        muti_step_ = optimizer_params.muti_step_;
    }

    SGDOptimizer::~SGDOptimizer()
    {

    }

    void SGDOptimizer::reset() 
    {

    }

    void SGDOptimizer::UpdateLr(uint32_t iter_cout)
    {
        switch (policy_)
        {
        case POLICY_FIXED:
            break;
        case POLICY_STEP:
            if((iter_cout+1)%step_){
                lr_ *= gamma_;
            }
            break;
        case POLICY_MUTISTEP:
            for(auto tmp:muti_step_){
                if(tmp == (iter_cout+1)){
                    lr_ *= gamma_;
                }
            }
            break;
        default:
            break;
        }
    }
    
    void SGDOptimizer::CheckLrUpdate(uint32_t iter_cout)
    {
        UpdateLr(iter_cout);
    }

    void SGDOptimizer::UpdateCpu(Tensor *weights, uint32_t batch)
    {
        float *W = (float *)weights->mutable_cpu_data();
        float *dW = (float *)weights->mutable_cpu_diff(); 
        uint32_t n = weights->shape().count(); 
        if(weights->vtype() == WEIGHTS){
            axpy_cpu(n, -decay_*batch, W, 1, dW, 1);
            axpy_cpu(n, lr_/batch, dW, 1, W, 1);
            scal_cpu(n, momentum_, dW, 1);
        }else{
            axpy_cpu(n, lr_/batch, dW, 1, W, 1);
            scal_cpu(n, momentum_, dW, 1);
        }
    }

#ifdef GPU
    void SGDOptimizer::UpdateGpu(Tensor *weights, uint32_t iter_cout, uint32_t batch)
    {

    }
#endif
} // namespace Grape



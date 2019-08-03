
#ifndef __GRAPE_SGD_H__
#define __GRAPE_SGD_H__

#include "grape/optimizer/optimizer.h"
#include "grape/tensor.h"
#include "grape/params/optimizer_params.h"
namespace Grape
{
    class SGDOptimizer:public Optimizer{
    public:
        SGDOptimizer(float lr);
        SGDOptimizer(float lr, float decay);
        SGDOptimizer(float lr,LR_POLICY policy,uint32_t step);
        SGDOptimizer(float lr,LR_POLICY policy,std::vector<uint32_t> muti_step);
        SGDOptimizer(OptimizerParams &optimizer_params);
        virtual ~SGDOptimizer();
        virtual void reset(); // override to implement pre-learning action
        virtual void UpdateCpu(Tensor *weights, uint32_t batch);
    #ifdef GPU
        virtual void UpdateGpu(Tensor *weights, uint32_t batch);
    #endif
        virtual void CheckLrUpdate(uint32_t iter_cout);
        inline float get_lr(){return lr_;};
        inline void set_lr(float lr){lr_=lr;};
        inline float get_decay(){return decay_;};
        inline void set_decay(float decay){decay_=decay;};
        inline float set_gamma(){return gamma_;};
        inline void set_gamma(float gamma){gamma_=gamma;};
        inline float get_momentum(){return momentum_;};
        inline void set_momentum(float momentum){momentum_=momentum;};
        inline uint32_t get_step(){return step_;};
        inline void set_step(uint32_t step){step_ = step;};
        inline LR_POLICY get_policy(){return policy_;};
        inline void set_policy(LR_POLICY policy){policy_=policy;};
        inline std::vector<uint32_t> &get_muti_step(){return muti_step_;};
        inline void set_muti_step(std::vector<uint32_t> &muti_step){muti_step_ = muti_step;};

    private:
        void UpdateLr(uint32_t iter_cout);
        float lr_ = 0;   // learning rate
        float decay_ = 0;  // weight decay
        LR_POLICY policy_ = POLICY_FIXED;
        uint32_t step_ = 1;
        std::vector<uint32_t> muti_step_;
        float gamma_ = 0;
        float momentum_ = 0;
    };
} // namespace Grape

#endif
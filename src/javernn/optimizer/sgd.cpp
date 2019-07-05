#include "javernn/optimizer/sgd.h"
#include "javernn/util/blas.h"

namespace javernn
{
    SGDOptimizer::SGDOptimizer(float lr):alpha_(lr), lambda_(float(0))
    {

    }

    SGDOptimizer::~SGDOptimizer()
    {

    }

    void SGDOptimizer::reset() 
    {

    }

    void SGDOptimizer::UpdateCpu(Tensor *weights)
    {
        float *W = (float *)weights->mutable_cpu_data();
        float *dW = (float *)weights->mutable_cpu_diff(); 
        uint32_t n = weights->shape().count(); 
        // if(n<1000){
        // std::cout<<"dW update before"<<std::endl;
        // for(int i=0;i<n;i++){
        //     std::cout<<dW[i]<<" ";
        // }
        // std::cout<<std::endl;
        // }
        // if(n<1000){
        // std::cout<<"W update before"<<std::endl;
        // for(int i=0;i<n;i++){
        //     std::cout<<W[i]<<" ";
        // }
        // std::cout<<std::endl;
        // }
        axpy_cpu(n, -lambda_, W, 1, dW, 1);
        axpy_cpu(n, alpha_, dW, 1, W, 1);
        // if(n<1000){
        // std::cout<<"w update after"<<std::endl;
        // for(int i=0;i<n;i++){
        //     std::cout<<W[i]<<" ";
        // }
        // std::cout<<std::endl;
        // }

    }

#ifdef GPU
    void SGDOptimizer::UpdateGpu(Tensor *weights)
    {

    }
#endif
} // namespace javernn



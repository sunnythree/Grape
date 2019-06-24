#ifndef __JAVERNN_NODE_H__
#define __JAVERNN_NODE_H__

#include <memory>
#include <vector>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "javernn/optimizers/optimizers.h"
#include "javernn/error.h"


namespace javernn{

class Tensor;
typedef std::shared_ptr<Tensor> tensorptr_t;

class Op{
public:
    Op() = delete;
    explicit Op(const std::vector<TENSOR_TYPE> &in_type,
        const std::vector<TENSOR_TYPE> &out_type);
    virtual ~Op();

    inline const std::vector<tensorptr_t> &prev() const { return prev_; }
    inline const std::vector<tensorptr_t> &next() const { return next_; }

    int32_t PrevPort(const Tensor &e) const;
    int32_t NextPort(const Tensor &e) const;

    std::vector<Op *> PrevOps() const;
    std::vector<Op *> NextOps() const;

    inline int32_t in_size() const { return in_size_; }
    inline int32_t out_size() const { return out_size_; }

    virtual void Setup() = 0;
    
    virtual std::vector<Tensor> ForwardCpu() = 0; 
    virtual void BackwardCpu() = 0;
    virtual void UpdateWeightsCpu(Optimizer &opt) = 0;

#ifdef GPU
    virtual std::vector<Tensor> ForwardGpu() = 0; 
    virtual void BackwardGpu() = 0;
    virtual void UpdateWeightsGpu(Optimizer &opt) = 0;
#endif


protected:
    friend void connect_op(Op *head,Op *tail,int32_t head_index,int32_t tail_index);
    mutable std::vector<tensorptr_t> prev_;
    mutable std::vector<tensorptr_t> next_;
    std::vector<TENSOR_TYPE> in_type_;
    std::vector<TENSOR_TYPE> out_type_;
    bool initialized_;
    int32_t in_size_;
    int32_t out_size_;
};

void connection_mismatch(const Op &from, const Op &to);
void connect_op(Op *head,Op *tail,int32_t head_index = 0,int32_t tail_index = 0);
Op &operator<<(Op &lhs, Op &rhs);

}

#endif
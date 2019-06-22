#ifndef __JAVERNN_EDGE_H__
#define __JAVERNN_EDGE_H__

#include <vector>
#include <memory>
#include "javernn/synced_memory.h"

namespace javernn{
    enum  TENSOR_TYPE{
        DATA,
        WEIGHTS,
        BIAS,
        LABEL,
        AUX,
    };
    class Op;
    class Tensor {
    public:
        Tensor(Op *prev,std::vector<uint32_t> shape,TENSOR_TYPE type)
        : prev_(prev),shape_(shape),type_(type){}
        virtual ~Tensor() {};

        inline const std::vector<Op *> &next() const { return next_; }
        inline Op *prev() { return prev_; }
        inline const Op *prev() const { return prev_; }
        inline void add_next_op(Op *next) { next_.push_back(next); }

    private:
        std::shared_ptr<SyncedMemory> data_;
        std::shared_ptr<SyncedMemory> diff_;
        Op *prev_;                // previous node, "producer" of this tensor
        std::vector<uint32_t> shape_;
        TENSOR_TYPE type_;
        std::vector<Op *> next_;  // next nodes, "consumers" of this tensor
    };
}

#endif
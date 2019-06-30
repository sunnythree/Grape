#ifndef __JAVERNN_TENSOR_H__
#define __JAVERNN_TENSOR_H__

#include <vector>
#include <memory>
#include <iostream>
#include "javernn/synced_memory.h"
#include "javernn/shape.h"

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
        Tensor(Op *prev,Shape shape,TENSOR_TYPE type,CAL_MODE mode)
        : prev_(prev),shape_(shape),type_(type),mode_(mode){
            //std::cout<<" Tensor "<<shape_.count()*sizeof(float)<<std::endl;
            data_ = std::make_shared<SyncedMemory>(shape_.count()*sizeof(float),mode_);
            diff_ = std::make_shared<SyncedMemory>(shape_.count()*sizeof(float),mode_);
        }
        virtual ~Tensor() {};

        inline const std::vector<Op *> &next() const { return next_; }
        inline Op *prev() { return prev_; }
        inline const Op *prev() const { return prev_; }
        inline void add_next_op(Op *next) { next_.push_back(next); }
        const Shape &shape() const { return shape_; }
        TENSOR_TYPE vtype() const { return type_; }
        inline const void *cpu_data(){return data_->cpu_data();};
        inline const void *cpu_diff(){return diff_->cpu_data();};
        inline const void *gpu_data(){return data_->gpu_data();};
        inline const void *gpu_diff(){return diff_->gpu_data();};
        inline void *mutable_cpu_data(){return data_->mutable_cpu_data();};
        inline void *mutable_cpu_diff(){return diff_->mutable_cpu_data();};
        inline void *mutable_gpu_data(){return data_->mutable_gpu_data();};
        inline void *mutable_gpu_diff(){return diff_->mutable_gpu_data();};
        inline void data_to_cpu(){data_->to_cpu();};
        inline void data_to_gpu(){data_->to_gpu();};
        inline void diff_to_cpu(){diff_->to_cpu();};
        inline void diff_to_gpu(){diff_->to_gpu();};

    private:
        std::shared_ptr<SyncedMemory> data_;
        std::shared_ptr<SyncedMemory> diff_;
        Op *prev_;                // previous node, "producer" of this tensor
        Shape shape_;
        TENSOR_TYPE type_;
        CAL_MODE mode_;
        std::vector<Op *> next_;  // next nodes, "consumers" of this tensor
    };
}

#endif
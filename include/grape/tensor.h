#ifndef __GRAPE_TENSOR_H__
#define __GRAPE_TENSOR_H__

#include <vector>
#include <memory>
#include <iostream>
#include "grape/synced_memory.h"
#include "grape/shape.h"

namespace Grape{
    enum  TENSOR_TYPE{
        DATA,
        WEIGHTS,
        BIAS,
        LABEL,
        AUX,
        INDEX
    };
    class Op;
    class Tensor {
    public:
        Tensor(Op *prev,Shape shape,TENSOR_TYPE type,int type_size): 
        prev_(prev),
        shape_(shape),
        type_(type)
        {
            //std::cout<<" Tensor "<<shape_.count()*sizeof(float)<<std::endl;
            data_ = std::make_shared<SyncedMemory>(shape_.count()*type_size);
            diff_ = std::make_shared<SyncedMemory>(shape_.count()*type_size);
        }
        virtual ~Tensor() {};

        inline const std::vector<Op *> &next() const { return next_; }
        inline Op *prev() { return prev_; }
        inline const Op *prev() const { return prev_; }
        inline void add_next_op(Op *next) { next_.push_back(next); }
        inline const Shape &shape() const { return shape_; }
        inline TENSOR_TYPE vtype() const { return type_; }
        inline const void *cpu_data(){return data_->cpu_data();};
        inline const void *cpu_diff(){return diff_->cpu_data();};
        inline const void *gpu_data(){return data_->gpu_data();};
        inline const void *gpu_diff(){return diff_->gpu_data();};
        inline void *mutable_cpu_data(){return data_->mutable_cpu_data();};
        inline void *mutable_cpu_diff(){return diff_->mutable_cpu_data();};
        inline void *mutable_gpu_data(){return data_->mutable_gpu_data();};
        inline void *mutable_gpu_diff(){return diff_->mutable_gpu_data();};

        template <class Archive>
        void serialize(Archive & ar)
        {
            float *data = (float *)cpu_data();
            for(int i=0;i<shape_.count();++i){
                ar(data[i]);
            }
        }
    private:
        std::shared_ptr<SyncedMemory> data_;
        std::shared_ptr<SyncedMemory> diff_;
        Op *prev_;                // previous node, "producer" of this tensor
        Shape shape_;
        TENSOR_TYPE type_;
        std::vector<Op *> next_;  // next nodes, "consumers" of this tensor
    };
}

#endif
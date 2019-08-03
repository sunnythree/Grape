#include <algorithm>
#include <string>
#include "grape/op.h"
#include "grape/tensor.h"
#include "grape/util/util.h"
#include "grape/log.h"


namespace Grape{
    static std::string TAG = "Op";

    Op::Op(const std::vector<TENSOR_TYPE> &in_type,
        const std::vector<TENSOR_TYPE> &out_type) 
        : in_type_(in_type),out_type_(out_type),
        prev_(in_type.size()), next_(out_type.size()) 
    {

    }

    Op::~Op() 
    {

    }

    int32_t Op::PrevPort(const Tensor &e) const {
        auto it = std::find_if(prev_.begin(), prev_.end(),
                                [&](tensorptr_t ep) { return ep.get() == &e; });
        return (int32_t)std::distance(prev_.begin(), it);
    }

    int32_t Op::NextPort(const Tensor &e) const {
        auto it = std::find_if(next_.begin(), next_.end(),
                                [&](tensorptr_t ep) { return ep.get() == &e; });
        return (int32_t)std::distance(next_.begin(), it);
    }

    std::vector<Op *> Op::PrevOps() const 
    {
        std::vector<Op *> vecs;
        for (auto &e : prev_) {
            if (e && e->prev()) {
                vecs.insert(vecs.end(), e->prev());
            }
        }
    return vecs;
    }

    std::vector<Op *> Op::PrevDataOps() const 
    {
        std::vector<Op *> vecs;
        for (auto &e : prev_) {
            if (e && e->prev() && e->vtype() == DATA) {
                vecs.insert(vecs.end(), e->prev());
            }
        }
    return vecs;
    }

    std::vector<Op *> Op::NextOps() const 
    {
        std::vector<Op *> vecs;
        for (auto &e : next_) {
            if (e) {
                auto n = e->next();
                vecs.insert(vecs.end(), n.begin(), n.end());
            }
        }
        return vecs;
    }

    bool exist_op_check(const std::vector<Op *> &op_list,Op *op)
    {
        for(auto tmp_op:op_list){
            if(tmp_op->get_name() == op->get_name()){
                return true;
            }
        }
        return false;
    }

    void connect_op(Op *head,
                        Op *tail,
                        int32_t head_index,
                        int32_t tail_index) 
    {

        if (!head->next_[head_index]) {
            throw Error("output edge must not be null");
        }
        Log::v(TAG,head->name_+"["+std::to_string(head_index)+"] << "
        +tail->name_+"["+std::to_string(tail_index)+"]");
        tail->prev_[tail_index] = head->next_[head_index];
        if(!exist_op_check(tail->prev_[tail_index]->next(),tail)){
            tail->prev_[tail_index]->add_next_op(tail);
        }
    }

    std::vector<Op *> operator,(Op &lhs, Op &rhs)
    {
        Log::v(TAG,"operator ,");
        std::vector<Op *> vec;
        vec.push_back(&lhs);
        vec.push_back(&rhs);
        Log::v(TAG,"operator ,");
        return vec;
    }

    std::vector<Op *> &operator,(std::vector<Op *> &lhs, Op &rhs)
    {
        lhs.push_back(&rhs);
        return lhs;
    }

    std::vector<Op *> &operator,(Op &lhs, std::vector<Op *> &rhs)
    {
        rhs.push_back(&lhs);
        return rhs;
    }

    std::vector<Op *> &operator,(std::vector<Op *> &lhs, std::vector<Op *> &rhs)
    {
        for(auto tmp:lhs){
            rhs.push_back(tmp);
        }
        return rhs;
    }

    Op &operator<<(Op &lhs, Op &rhs) {
        connect_op(&lhs, &rhs);
        return rhs;
    }

    Op &operator<<(std::vector<Op *> &lhs, Op &rhs)
    {
        Log::v(TAG,"operator << ll lhs size is "+std::to_string(lhs.size()));
        for (int32_t i = 0; i < lhs.size(); ++i) {
            Log::v(TAG,"i is "+std::to_string(i));
            connect_op(lhs[i], &rhs, 0, i);
        }
        return rhs;
    }

    std::vector<Op *> &operator<<(Op &lhs, std::vector<Op *> &rhs)
    {
        for (size_t i = 0; i < rhs.size(); i++) {
            connect_op(&lhs, rhs[i], i, 0);
        }
        return rhs;
    }

}

#include <algorithm>
#include "javernn/op.h"
#include "javernn/tensor.h"
#include "javernn/util/util.h"

namespace javernn{
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

    void connect_op(Op *head,
                        Op *tail,
                        int32_t head_index,
                        int32_t tail_index) {

        if (!head->next_[head_index]) {
            throw Error("output edge must not be null");
        }

        tail->prev_[tail_index] = head->next_[head_index];
        tail->prev_[tail_index]->add_next_op(tail);
    }

    inline Op &operator<<(Op &lhs, Op &rhs) {
        connect_op(&lhs, &rhs);
        return rhs;
    }
}

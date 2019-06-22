#include <algorithm>
#include "javernn/op.h"
#include "javernn/tensor.h"

namespace javernn{
    Op::Op(size_t in_size, size_t out_size) : prev_(in_size), next_(out_size) 
    {

    }
    Op::~Op() 
    {

    }

    size_t Op::PrevPort(const Tensor &e) const {
        auto it = std::find_if(prev_.begin(), prev_.end(),
                                [&](tensorptr_t ep) { return ep.get() == &e; });
        return (size_t)std::distance(prev_.begin(), it);
    }

    size_t Op::NextPort(const Tensor &e) const {
        auto it = std::find_if(next_.begin(), next_.end(),
                                [&](tensorptr_t ep) { return ep.get() == &e; });
        return (size_t)std::distance(next_.begin(), it);
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
    void Op::Backward()
    {

    }

    std::vector<Tensor> Op::Forward()
    {
        std::vector<Tensor> cost;
        return cost;
    }

    void Op::UpdateWeights(Optimizer *opt)
    {

    }

    void Op::Setup(bool reset_weight)
    {

    }
}

#ifndef __JAVERNN_EDGE_H__
#define __JAVERNN_EDGE_H__

#include <vector>

namespace javernn{
    class Op;
    class Tensor {
    public:
        Tensor(Op *prev)
        : prev_(prev) {}
        virtual ~Tensor() {};

        inline const std::vector<Op *> &Next() const { return next_; }
        inline Op *Prev() { return prev_; }
        inline const Op *Prev() const { return prev_; }
        inline void AddNextNode(Op *next) { next_.push_back(next); }

    private:
        Op *prev_;                // previous node, "producer" of this tensor
        std::vector<Op *> next_;  // next nodes, "consumers" of this tensor
    };
}

#endif
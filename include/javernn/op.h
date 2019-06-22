#ifndef __javernn_node_h__
#define __javernn_node_h__

#include <memory>
#include <vector>
#include <cstdlib>
#include "javernn/optimizers/optimizers.h"

namespace javernn{

class Tensor;
typedef std::shared_ptr<Tensor> tensorptr_t;

class Op{
public:
    Op(size_t in_size, size_t out_size);
    virtual ~Op();

    inline const std::vector<tensorptr_t> &prev() const { return prev_; }
    inline const std::vector<tensorptr_t> &next() const { return next_; }

    size_t PrevPort(const Tensor &e) const;
    size_t NextPort(const Tensor &e) const;

    std::vector<Op *> PrevOps() const;
    std::vector<Op *> NextOps() const;


    void Backward();
    std::vector<Tensor> Forward();  // NOLINT
    void UpdateWeights(Optimizer *opt);
    void Setup(bool reset_weight);

protected:
    Op() = delete;

    friend void connect(Op *head,
                        Op *tail,
                        size_t head_index,
                        size_t tail_index);

    mutable std::vector<tensorptr_t> prev_;
    mutable std::vector<tensorptr_t> next_;
};

inline void connect(Op *head,
                    Op *tail,
                    size_t head_index = 0,
                    size_t tail_index = 0) {
  tail->prev_[tail_index] = head->next_[head_index];
  tail->prev_[tail_index]->add_next_op(tail);
}

inline Op &operator<<(Op &lhs, Op &rhs) {
  connect(&lhs, &rhs);
  return rhs;
}

}

#endif
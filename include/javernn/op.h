#ifndef __javernn_node_h__
#define __javernn_node_h__

#include <memory>
#include <vector>


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

protected:
    Op() = delete;

    friend void connect(Op *head,
                        Op *tail,
                        size_t head_index,
                        size_t tail_index);

    mutable std::vector<tensorptr_t> prev_;
    mutable std::vector<tensorptr_t> next_;
};


}

#endif
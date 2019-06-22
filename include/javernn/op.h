#ifndef __javernn_node_h__
#define __javernn_node_h__

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
    Op(int32_t in_size, int32_t out_size);
    virtual ~Op();

    inline const std::vector<tensorptr_t> &prev() const { return prev_; }
    inline const std::vector<tensorptr_t> &next() const { return next_; }

    int32_t PrevPort(const Tensor &e) const;
    int32_t NextPort(const Tensor &e) const;

    std::vector<Op *> PrevOps() const;
    std::vector<Op *> NextOps() const;


    void Backward();
    std::vector<Tensor> Forward(); 
    virtual void ForwardPropagation(const std::vector<Tensor *> &in_data,
                                   std::vector<Tensor *> &out_data) = 0;
    virtual void BackPropagation(const std::vector<Tensor *> &in_data,
                                const std::vector<Tensor *> &out_data,
                                std::vector<Tensor *> &out_grad,
                                std::vector<Tensor *> &in_grad) = 0;
    void UpdateWeights(Optimizer *opt);
    void Setup(bool reset_weight);


    virtual std::vector<Shape> in_shape() const = 0;
    virtual std::vector<Shape> out_shape() const = 0;
    virtual std::string layer_type() const = 0;
    virtual void SetInShape(const Shape &in_shape);

    inline int32_t in_size() const { return in_size_; }
    ///< number of outgoing edges in this layer
    inline int32_t out_size() const { return out_size_; }

    int32_t InDataSize() const;
    int32_t OutDataSize() const;
    std::vector<Shape> InDataShape();
    std::vector<Shape> OutDataShape();
protected:
    Op() = delete;

    friend void connect_op(Op *head,
                        Op *tail,
                        int32_t head_index,
                        int32_t tail_index);

    mutable std::vector<tensorptr_t> prev_;
    mutable std::vector<tensorptr_t> next_;
    bool initialized_;
    size_t in_size_;
    size_t out_size_;
    std::vector<TENSOR_TYPE> in_type_;
    std::vector<TENSOR_TYPE> out_type_;
};

void connection_mismatch(const Op &from, const Op &to);
void connect_op(Op *head,Op *tail,int32_t head_index = 0,int32_t tail_index = 0);
Op &operator<<(Op &lhs, Op &rhs);

}

#endif
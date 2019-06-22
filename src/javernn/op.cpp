#include <algorithm>
#include "javernn/op.h"
#include "javernn/tensor.h"
#include "javernn/util/util.h"

namespace javernn{
    Op::Op(int32_t in_size, int32_t out_size) : prev_(in_size), next_(out_size) 
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
        if (in_shape().size() != in_size_ ||
            out_shape().size() != out_size_) {
        throw Error("Connection mismatch at setup layer");
        }

        for (size_t i = 0; i < out_size_; i++) {
            if (!next_[i]) {
                next_[i] = std::make_shared<Tensor>(this, out_shape()[i], out_type_[i]);
            }
        }

        if (reset_weight || !initialized_) {
        //init_weight();
        }
    }

    void Op::SetInShape(const Shape &in_shape) {
        throw Error(
        "Can't set shape. Shape inferring not applicable for this "
        "layer (yet).");
    }

    int32_t Op::InDataSize() const {
        return sumif(in_shape(),
                    [&](size_t i) {  // NOLINT
                        return in_type_[i] == DATA;
                    },
                    [](const Shape &s) { return s.count(); });
    }

    int32_t Op::OutDataSize() const {
    return sumif(out_shape(),
                 [&](size_t i) {  // NOLINT
                   return out_type_[i] == DATA;
                 },
                 [](const Shape &s) { return s.count(); });
    }

    std::vector<Shape> Op::InDataShape() {
        return filter(in_shape(), [&](size_t i) {  // NOLINT
        return in_type_[i] == DATA;
        });
    }

    std::vector<Shape> Op::OutDataShape() {
        return filter(out_shape(), [&](size_t i) {  // NOLINT
        return out_type_[i] == DATA;
        });
    }

    void connection_mismatch(const Op &from, const Op &to) {
        std::ostringstream os;

        os << std::endl;
        os << "output size of Nth layer must be equal to input of (N+1)th layer\n";

        os << "OpN:   " << std::setw(12) << from.layer_type()
            << " in:" << from.InDataSize() << "(" << from.in_shape().size() << "), "
            << "out:" << from.OutDataSize() << "(" << from.out_shape().size() << ")\n";

        os << "OpN+1: " << std::setw(12) << to.layer_type()
            << " in:" << to.InDataSize() << "(" << to.in_shape() .size()<< "), "
            << "out:" << to.OutDataSize() << "(" << to.out_shape().size() << ")\n";

        os << from.OutDataSize() << " != " << to.InDataSize() << std::endl;
        std::string detail_info = os.str();

        throw Error("layer dimension mismatch!" + detail_info);
    }

    void connect_op(Op *head,
                        Op *tail,
                        int32_t head_index,
                        int32_t tail_index) {
        Shape out_shape = head->out_shape()[head_index];
        Shape in_shape  = tail->in_shape()[tail_index];

        head->Setup(false);

        // todo (karandesai) enable shape inferring for all layers
        // currently only possible for activation layers.
        if (in_shape.count() == 0) {
            tail->SetInShape(out_shape);
            in_shape = out_shape;
        }

        if (out_shape.count() != in_shape.count()) {
            connection_mismatch(*head, *tail);
        }

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

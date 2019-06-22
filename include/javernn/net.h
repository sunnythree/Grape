#ifndef __JAVERNN_NET_H__
#define __JAVERNN_NET_H__

#include "javernn/optimizers/optimizers.h"
#include "javernn/ops.h"
namespace javernn{
    enum NET_TYPE{
        NET_SEQUENCE,
        NET_GRAPH
    };
    class Net{
    public:
        explicit Net(NET_TYPE net_type);
        ~Net();
        void Train(Optimizer &optimizer,
            const std::vector<Tensor> &inputs,
            const std::vector<Tensor> &class_labels,
            int32_t batch_size,
            int32_t epoch);
        void Test(const std::vector<Tensor> &inputs,
            const std::vector<Tensor> &class_labels);
    private:
        std::shared_ptr<Ops> ops_;
    };
}

#endif
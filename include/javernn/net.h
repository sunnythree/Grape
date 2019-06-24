#ifndef __JAVERNN_NET_H__
#define __JAVERNN_NET_H__

#include "javernn/optimizers/optimizers.h"
#include "javernn/ops.h"
#include "javernn/params/net_params.h"


namespace javernn{

    class Net{
    public:
        explicit Net(NetParams &net_params);
        ~Net();
        void Construct();
        void Construct(const std::vector<Op *> &input,
                 const std::vector<Op *> &output);
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
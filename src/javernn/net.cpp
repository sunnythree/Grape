#include "javernn/net.h"
#include "javernn/graph.h"
#include "javernn/sequence.h"

namespace javernn{
    Net::Net(NET_TYPE net_type)
    {
        if(net_type == NET_GRAPH){
            ops_ = std::make_shared<Graph>();
        }else{
            ops_ = std::make_shared<Sequence>();
        }
    }

    Net::~Net()
    {

    }

    void Net::Train(Optimizer &optimizer,
        const std::vector<Tensor> &inputs,
        const std::vector<Tensor> &class_labels,
        int32_t batch_size,
        int32_t epoch)
    {

    }

    void Net::Test(const std::vector<Tensor> &in,
        const std::vector<Tensor> &t)
    {

    }
}
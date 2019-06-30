#include <memory>
#include "javernn/net.h"
#include "javernn/graph.h"

namespace javernn{
    Net::Net(NetParams &net_params)
    {
        ops_ = std::make_shared<Graph>();
    }

    Net::~Net()
    {

    }

    //build graph
    void Net::Construct(const std::vector<Op *> &input,
                const std::vector<Op *> &output)
    {
        dynamic_cast<Graph *>(ops_.get())->Construct(input,output);
    }

    void Net::Train()
    {
        ops_->Backward();
        ops_->UpdateWeights(*optimizer_);
    }

    void Net::Test()
    {
        ops_->Forward();
    }

    
}
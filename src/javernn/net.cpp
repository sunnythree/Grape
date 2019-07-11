#include <memory>
#include "javernn/optimizer/sgd.h"
#include "javernn/net.h"
#include "javernn/graph.h"

namespace javernn{
    Net::Net(NetParams &net_params)
    {
        ops_ = std::make_shared<Graph>();
        max_train_iters_ = net_params.max_train_iters_;
        optimizer_ = std::make_shared<SGDOptimizer>(0.1f);
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
        for(int i=0;i<max_train_iters_;++i){
            ops_->Forward();
            ops_->Backward();
            ops_->UpdateWeights(*optimizer_);
        }

    }

    void Net::Test()
    {
        ops_->Forward();
    }

    
}
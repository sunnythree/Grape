#include <memory>
#include "javernn/net.h"
#include "javernn/graph.h"
#include "javernn/sequence.h"

namespace javernn{
    Net::Net(NetParams &net_params)
    {
        if(net_params.net_type_ == "graph"){
            ops_ = std::make_shared<Graph>();
        }else{
            ops_ = std::make_shared<Sequence>();
        }
    }

    Net::~Net()
    {

    }

    //build sequence 
    void Net::Construct()
    {
        dynamic_cast<Sequence *>(ops_.get())->Construct();
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
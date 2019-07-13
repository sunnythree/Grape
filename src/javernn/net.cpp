#include <memory>
#include "javernn/optimizer/sgd.h"
#include "javernn/net.h"
#include "javernn/graph.h"

namespace javernn{
    Net::Net(NetParams &net_params)
    {
        ops_ = std::make_shared<Graph>("data/test",JSON,SGD,0.005f);
        max_train_iters_ = net_params.max_train_iters_;
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
            ops_->UpdateWeights();
        }

    }

    void Net::Test()
    {
        for(int i=0;i<max_train_iters_;++i){
            ops_->Forward();
        }
    }

    void Net::Save()
    {
        ops_->Save();
    }
    
    void Net::Load()
    {
        ops_->Load();
    }
}
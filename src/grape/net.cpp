#include <memory>
#include "grape/optimizer/sgd.h"
#include "grape/net.h"
#include "grape/graph.h"

namespace Grape{
    Net::Net(NetParams &net_params)
    {
        max_iter_ = net_params.max_iter_;
    }

    Net::~Net()
    {

    }

    void Net::Run()
    {
        for(uint32_t i = 0;i<max_iter_;++i){
            for(auto iter : ops_){
                iter->Run();
            }
        }
    }

    void Net::AddOps(Ops *ops)
    {
        ops_.push_back(ops);
    }
}
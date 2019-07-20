#include <memory>
#include "grape/optimizer/sgd.h"
#include "grape/net.h"
#include "grape/graph.h"

namespace Grape{
    Net::Net(NetParams &net_params)
    {

    }

    Net::~Net()
    {

    }

    void Net::Run()
    {
        if(iter_mode_ == SEQUENCE){
            for(auto iter : ops_){
                iter->Run();
            }
        }else if(iter_mode_ == STAGGER){
            std::vector<int> iter_count(the_ops_iters.size());
            for(auto tmp:iter_count) tmp = 0;
            bool finish = true;
            while(true){
                for(int i=0;i<the_ops_iters.size();++i){
                    int iters = the_ops_iters[i];
                    iter_count[i] += iters;
                    if(iter_count[i] < ops_[i]->GetMaxIter()){
                        for(int j=0; j<iters ;++j){
                            ops_[j]->RunOnce();
                        }
                    }
                }
                //check break
                finish = true;
                for(int i=0;i<iter_count.size();++i){
                    if(iter_count[i] < ops_[i]->GetMaxIter()){
                        finish = false;
                    }
                }
                if(finish){
                    break;
                }
            }
        }
    }

    void Net::AddOps(Ops *ops)
    {
        ops_.push_back(ops);
    }
}
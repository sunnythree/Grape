#ifndef __GRAPE_NET_H__
#define __GRAPE_NET_H__

#include "grape/optimizer/optimizer.h"
#include "grape/ops.h"
#include "grape/params/net_params.h"
#include "grape/graph.h"


namespace Grape{

    class Net{
    public:
        explicit Net(NetParams &net_params);
        ~Net();
        void Run();
        void AddOps(Ops *ops);
        inline uint32_t get_max_iter(){return max_iter_;};
        inline void set_max_iter(uint32_t iter){max_iter_ = iter;};
    private:
        std::vector<Ops *> ops_;
        uint32_t max_iter_ = 1;
    };
}

#endif
#ifndef __GRAPE_NET_H__
#define __GRAPE_NET_H__

#include "grape/optimizer/optimizer.h"
#include "grape/ops.h"
#include "grape/params/net_params.h"
#include "grape/graph.h"


namespace Grape{
    enum ITER_MODE{
        STAGGER,
        SEQUENCE
    };
    class Net{
    public:
        explicit Net(NetParams &net_params);
        ~Net();
        void Run();
        void AddOps(Ops *ops);
    private:
        std::vector<Ops *> ops_;
        std::vector<int32_t> the_ops_iters;
        ITER_MODE iter_mode_ = SEQUENCE;
    };
}

#endif
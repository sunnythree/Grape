#ifndef __JAVERNN_NET_H__
#define __JAVERNN_NET_H__

#include "javernn/optimizer/optimizer.h"
#include "javernn/ops.h"
#include "javernn/params/net_params.h"
#include "javernn/graph.h"


namespace javernn{
    typedef std::shared_ptr<Ops> ops_t;
    enum ITER_MODE{
        STAGGER,
        SEQUENCE
    };
    class Net{
    public:
        explicit Net(NetParams &net_params);
        ~Net();
        void Run();
    private:
        std::vector<ops_t> ops_;
        std::vector<int32_t> the_ops_iters;
        ITER_MODE iter_mode_ = SEQUENCE;
    };
}

#endif
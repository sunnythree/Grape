#ifndef __javernn_nodes_h__
#define __javernn_nodes_h__
#include <vector>
#include "javernn/op.h"
#include "javernn/global_config.h"

namespace javernn{
    class Ops{
    public:
        virtual ~Ops(){};
        virtual void Backward() = 0;
        virtual void Forward() = 0;  
        virtual void UpdateWeights() = 0;
        virtual void Setup() = 0;
        virtual void TrainOnce() = 0;
        virtual void Train() = 0;
        virtual void TestOnce() = 0;
        virtual void Test() = 0;
        virtual void Run() = 0;
        virtual void RunOnce() = 0;
        virtual uint32_t GetMaxIter() = 0;
        virtual GRAPH_PHASE GetGraphPhase() = 0;
        virtual void Save();
        virtual void Load();
    };
}

#endif
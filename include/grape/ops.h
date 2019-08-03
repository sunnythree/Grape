#ifndef __GRAPE_nodes_h__
#define __GRAPE_nodes_h__
#include <vector>
#include "grape/op.h"
#include "grape/global_config.h"

namespace Grape{
    class Ops{
    public:
        virtual ~Ops(){};
        virtual void Backward() = 0;
        virtual void Forward() = 0;  
        virtual void UpdateWeights() = 0;
        virtual void Setup(bool load) = 0;
        virtual void TrainOnce() = 0;
        virtual void Train() = 0;
        virtual void TestOnce() = 0;
        virtual void Test() = 0;
        virtual void Run() = 0;
        virtual void RunOnce() = 0;
        virtual PHASE GetPhase() = 0;
        virtual void SetPhase(PHASE phase) = 0;
        virtual uint32_t GetMaxIter() = 0;
        virtual void SetMaxIter(uint32_t iter) = 0;
        virtual void Save();
        virtual void Load();
        virtual void OnNetRunBegin();
        virtual void OnNetRunEnd();
    };
}

#endif
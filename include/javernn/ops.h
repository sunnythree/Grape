#ifndef __javernn_nodes_h__
#define __javernn_nodes_h__
#include <vector>
#include "javernn/op.h"

namespace javernn{
    class Ops{
    public:
        virtual ~Ops(){};
        virtual void Backward() = 0;
        virtual void Forward() = 0;  
        virtual void UpdateWeights() = 0;
        virtual void Setup() = 0;
        virtual void Save();
        virtual void Load();
    };
}

#endif
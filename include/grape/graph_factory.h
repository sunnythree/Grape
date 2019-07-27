#ifndef __GRAPE_GRAPH_FACTORY_H__
#define __GRAPE_GRAPH_FACTORY_H__

#include <memory>
#include <vector>
#include "grape/graph.h"
#include "grape/params/graph_params.h"


namespace Grape{

    class GraphFactory{
    public:
        static std::shared_ptr<Graph> Build(GraphParams& gp);
        static std::vector<std::shared_ptr<Graph>> Build(std::vector<GraphParams> gps);
    };
}

#endif
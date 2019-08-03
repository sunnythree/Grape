#include "grape/graph_factory.h"

namespace Grape
{
            // std::string save_path,
            // SERIALIZE_TYPE serialize_type,
            // int32_t max_iter,
            // PHASE graph_phase,
            // CAL_MODE cal_mode
    std::shared_ptr<Graph> GraphFactory::Build(GraphParams& gp)
    {
        return std::make_shared<Graph>(gp);
    }

    std::vector<std::shared_ptr<Graph>> GraphFactory::Build(std::vector<GraphParams> gps)
    {
        std::vector<std::shared_ptr<Graph>> graphs;
        for(int i=0;i<gps.size();i++){
            std::shared_ptr<Graph> graph = Build(gps[i]);
            graphs.emplace_back(graph);
        }
        return graphs;
    }
} // namespace Grape        
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
        return std::make_shared<Graph>(
            gp.save_path_,
            gp.serialize_type_,
            gp.max_iter_,
            gp.phase_,
            gp.cal_mode_
            );
    }

    std::vector<std::shared_ptr<Graph>> GraphFactory::Build(std::vector<GraphParams> gps)
    {

    }
} // namespace Grape        
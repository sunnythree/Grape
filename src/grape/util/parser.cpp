#include <fstream>
#include "grape/util/parser.h"
#include "cereal/types/vector.hpp"
#include "cereal/types/map.hpp"

namespace Grape
{
    static std::string NET = "net";
    static std::string GRAPHS = "graphs";
    static std::string CONNECTIONS = "connections";
    static std::string OP_LIST = "op_list";
    void Parser::Parse(std::string path,OpListParams& op_list,GraphListParams &graph_list,
            ConnectionParams &connections,NetParams &net)
    {
        std::ifstream is(path);
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp(OP_LIST,op_list));
        archive(cereal::make_nvp(GRAPHS,graph_list));
        archive(cereal::make_nvp(CONNECTIONS,connections));
        archive(cereal::make_nvp(NET,net));
    }

    void Parser::Serialize(std::string path,OpListParams& op_list,GraphListParams &graph_list,
            ConnectionParams &connections,NetParams &net)
    {
        std::ofstream os(path);
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp(OP_LIST,op_list));
        archive(cereal::make_nvp(GRAPHS,graph_list));
        archive(cereal::make_nvp(CONNECTIONS,connections));
        archive(cereal::make_nvp(NET,net));
    }
} // namespace Grape

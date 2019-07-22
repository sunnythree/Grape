#ifndef __GRAPE_PARSER_H__
#define __GRAPE_PARSER_H__

#include "grape/params/net_params.h"
#include "grape/params/connection_params.h"
#include "grape/params/graph_list_params.h"
#include "grape/params/op_list_params.h"

namespace Grape
{
    class Parser
    {
    public:
        static void Parse(std::string path,OpListParams& op_list,GraphListParams &graph_list,
            ConnectionParams &connections,NetParams &net);
        static void Serialize(std::string path,OpListParams& op_list,GraphListParams &graph_list,
            ConnectionParams &connections,NetParams &net);
    };
}


#endif
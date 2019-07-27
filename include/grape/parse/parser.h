#ifndef __GRAPE_PARSER_H__
#define __GRAPE_PARSER_H__

#include "grape/params/net_params.h"
#include "grape/params/connection_list_params.h"
#include "grape/params/graph_list_params.h"
#include "grape/params/op_list_params.h"
#include "grape/params/op_path_params.h"
#include "grape/params/optimizer_list_params.h"
#include "grape/op.h"
#include "grape/graph.h"

namespace Grape
{

    class Parser
    {
    public:
        static void Parse(std::string path,
            OpPathParams& op_path);

        static void Parse(std::string path,
            OptimizerListParams& optimizer_list);

        static void Parse(std::string path,
            GraphListParams &graph_list);

        static void Parse(std::string path,
            ConnectionListParams &connection_list);

        static void Parse(std::string path,
            NetParams &net);

        static void Parse(std::string path,
            OpListParams& op_list);

        static void Parse(std::string path,
            OpPathParams& op_path,
            OptimizerListParams &optimizer_list,
            GraphListParams &graph_list,
            ConnectionListParams &connection_list,
            NetParams &net);

        static void Serialize(std::string path,
            OpPathParams& op_path);

        static void Serialize(std::string path,
            OptimizerListParams& optimizer_list);

        static void Serialize(std::string path,
            GraphListParams &graph_list);

        static void Serialize(std::string path,
            ConnectionListParams &connection_list);

        static void Serialize(std::string path,
            NetParams &net);

        static void Serialize(std::string path,
            OpListParams& op_list);

        static void Serialize(std::string path,
            OpPathParams& op_path,
            OptimizerListParams &optimizer_list,
            GraphListParams &graph_list,
            ConnectionListParams &connection_list,
            NetParams &net);

        static void ParseOpList(std::string path,OpListParams &op_list);

        static void SerializeOpList(std::string path,OpListParams &op_list);

        void Parse(std::string path);

        void ParseOpConnection(std::string &prev,std::string &next,OpConnection &conn);

        void CombineOpAndGraph(
            ConnectionParams &conn,
            std::shared_ptr<Graph> &graph,
            std::map<std::string,std::shared_ptr<Op>> &ops
        );

        void ParseInputAndOutput(
            std::vector<OpConnection> &conn,
            std::vector<std::string> &inputs,
            std::vector<std::string> &outputs
        );

    private:
        std::map<std::string,std::map<std::string,std::shared_ptr<Op>>> op_map_;
        std::map<std::string,std::shared_ptr<Graph>> graph_map_;

    };
}


#endif
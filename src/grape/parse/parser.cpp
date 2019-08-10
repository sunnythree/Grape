#include <fstream>
#include <set>
#include <map>
#include "assert.h"
#include "grape/parse/parser.h"
#include "cereal/types/vector.hpp"
#include "cereal/types/map.hpp"
#include "grape/op_factory.h"
#include "grape/graph_factory.h"
#include "grape/optimizer_factory.h"
#include "grape/net.h"
#include "grape/util/util.h"
#include "grape/log.h"

namespace Grape
{
    static const std::string NET = "net";
    static const std::string GRAPHS = "graphs";
    static const std::string CONNECTIONS = "connections";
    static const std::string OP_PATH = "op_paths";
    static const std::string OP_LIST = "ops";
    static const std::string OPTIMIZER_LIST = "optimizers";
    static const std::string TAG = "Parser";
    void Parser::Parse(std::string path,
        OpPathParams& op_path)
    {
        std::ifstream is(path);
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp(OP_PATH,op_path));
    }

    void Parser::Parse(std::string path,
        OptimizerListParams& optimizer_list)
    {
        std::ifstream is(path);
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp(OPTIMIZER_LIST,optimizer_list));
    }

    void Parser::Parse(std::string path,
        GraphListParams &graph_list)
    {
        std::ifstream is(path);
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp(GRAPHS,graph_list));
    }

    void Parser::Parse(std::string path,
        ConnectionListParams &connection_list)
    {
        std::ifstream is(path);
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp(CONNECTIONS,connection_list));
    }

    void Parser::Parse(std::string path,
        NetParams &net)
    {
        std::ifstream is(path);
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp(NET,net));
    }

    void Parser::Parse(std::string path,
        OpListParams &op_list)
    {
        std::ifstream is(path);
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp(OP_LIST,op_list));
    }

    void Parser::Parse(std::string path,
            OpPathParams& op_path,
            OptimizerListParams &optimizer_list,
            GraphListParams &graph_list,
            ConnectionListParams &connection_list,
            NetParams &net)
    {
        std::ifstream is(path);
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp(OP_PATH,op_path));
        archive(cereal::make_nvp(OPTIMIZER_LIST,optimizer_list));
        archive(cereal::make_nvp(GRAPHS,graph_list));
        archive(cereal::make_nvp(CONNECTIONS,connection_list));
        archive(cereal::make_nvp(NET,net));
    }

    void Parser::Serialize(std::string path,
        OpPathParams& op_path)
    {
        std::ofstream os(path);
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp(OP_PATH,op_path));
    }

    void Parser::Serialize(std::string path,
        OpListParams& op_list)
    {
        std::ofstream os(path);
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp(OP_LIST,op_list));
    }

    void Parser::Serialize(std::string path,
        OptimizerListParams& optimizer_list)
    {
        std::ofstream os(path);
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp(OPTIMIZER_LIST,optimizer_list));
    }

    void Parser::Serialize(std::string path,
        GraphListParams &graph_list)
    {
        std::ofstream os(path);
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp(GRAPHS,graph_list));
    }

    void Parser::Serialize(std::string path,
        ConnectionListParams &connection_list)
    {
        std::ofstream os(path);
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp(CONNECTIONS,connection_list));
    }

    void Parser::Serialize(std::string path,
        NetParams &net)
    {
        std::ofstream os(path);
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp(NET,net));
    }

    void Parser::Serialize(std::string path,
            OpPathParams& op_path,
            OptimizerListParams &optimizer_list,
            GraphListParams &graph_list,
            ConnectionListParams &connection_list,
            NetParams &net)
    {
        std::ofstream os(path);
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp(OP_PATH,op_path));
        archive(cereal::make_nvp(OPTIMIZER_LIST,optimizer_list));
        archive(cereal::make_nvp(GRAPHS,graph_list));
        archive(cereal::make_nvp(CONNECTIONS,connection_list));
        archive(cereal::make_nvp(NET,net));
    }

    void Parser::ParseOpList(std::string path,OpListParams &op_list)
    {
        std::ifstream is(path);
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp(OP_LIST,op_list));
    }

    void Parser::SerializeOpList(std::string path,OpListParams &op_list)
    {
        std::ofstream os(path);
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp(OP_LIST,op_list));
    }

    void Parser::ParseOpConnection(std::string &prev,std::string &next,OpConnection &conn)
    {
        trim(prev);
        trim(next);
        std::vector<std::string> split_f;
        split(prev,":",split_f);
        std::vector<std::string> split_s;
        split(next,":",split_s);
        assert(split_f.size()==2);
        assert(split_s.size()==2);
        conn.head = split_f[0];
        conn.head_index = std::atoi(split_f[1].c_str());
        conn.tail = split_s[0];
        conn.tail_index = std::atoi(split_s[1].c_str());
    }

    void Parser::ParseInputAndOutput(
        std::vector<OpConnection> &conn,
        std::vector<std::string> &inputs,
        std::vector<std::string> &outputs)
    {
        std::map<std::string,std::vector<std::string>> prev_map;
        std::map<std::string,std::vector<std::string>> next_map;
        std::set<std::string> all_op_name;
        for(int i=0;i<conn.size();i++){
            next_map[conn[i].head].emplace_back(conn[i].tail);
            prev_map[conn[i].tail].emplace_back(conn[i].head);
            all_op_name.insert(conn[i].head);
            all_op_name.insert(conn[i].tail);
        }
        //check input and output
        //no next is ouput
        for(auto tmp:all_op_name){
            auto iter = next_map.find(tmp);
            if(iter == next_map.end()){
                outputs.emplace_back(tmp);
            }
        }
        //no prev is input
        for(auto tmp:all_op_name){
            auto iter = prev_map.find(tmp);
            if(iter == prev_map.end()){
                inputs.emplace_back(tmp);
            }
        }
    }

    void Parser::CombineOpAndGraph(ConnectionParams &conn,
        std::shared_ptr<Graph> &graph,
        std::map<std::string,std::shared_ptr<Op>> &ops)
    {
        std::vector<OpConnection> connnections;
        for(int i=0;i<conn.connections_.size();i++){
            OpConnection ocn;
            ParseOpConnection(
                conn.connections_[i].from,
                conn.connections_[i].to,
                ocn
                );
            connnections.emplace_back(ocn);
            //ensure head and tail exist
            auto iter = ops.find(ocn.head);
            assert(iter != ops.end());
            iter = ops.find(ocn.tail);
            assert(iter != ops.end());
            //connect op
            std::shared_ptr<Op> head = ops[ocn.head];
            std::shared_ptr<Op> tail = ops[ocn.tail];
            connect_op(head.get(),tail.get(),ocn.head_index,ocn.tail_index);
        }
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        ParseInputAndOutput(connnections,inputs,outputs);
        std::vector<Op *> inputs_op;
        for(int i=0;i<inputs.size();i++){
            inputs_op.emplace_back(ops[inputs[i]].get());
        }
        std::vector<Op *> outputs_op;
        for(int i=0;i<inputs.size();i++){
            outputs_op.emplace_back(ops[outputs[i]].get());
        }
        graph->Construct(inputs_op,outputs_op);
    }

    void Parser::Parse(std::string path)
    {
        NetParams net_params;
        GraphListParams graph_list;
        ConnectionListParams connection_list;
        OpPathParams op_path;
        OptimizerListParams optimizer_list;
        Parser::Parse(path,op_path,optimizer_list,graph_list,connection_list,net_params);
        
        //get op_params_map
        std::map<std::string,std::vector<OpParams>> op_params_map;
        for(auto tmp:op_path.path_list_){
            std::string name = tmp.name;
            std::string path = tmp.path;
            OpListParams op_list;
            Parser::ParseOpList(path,op_list);
            op_params_map.insert(std::pair<std::string,std::vector<OpParams>>(name,op_list.op_list_));
        }
        //to this,all parameters parse over
        //we start build obj by parameters

        //get op_map
        for(auto tmp:op_params_map){
            auto tmp_op_list = OpFactory::Build(tmp.second);
            op_map_.insert(std::pair<std::string,std::map<std::string,std::shared_ptr<Op>>>(tmp.first,tmp_op_list));
        }

        //get graph_map
        for(auto tmp:graph_list.graph_list_){
            auto tmp_graph = GraphFactory::Build(tmp);
            graph_map_.insert(std::pair<std::string,std::shared_ptr<Graph>>(tmp.name_,tmp_graph));
        }
        //combine op_map and graph_map
        for(int i=0;i<connection_list.connection_list_.size();i++){
            ConnectionParams &conn = connection_list.connection_list_[i];
            std::shared_ptr<Graph> &graph = graph_map_[conn.graph_name_];
            std::map<std::string,std::shared_ptr<Op>> &ops = op_map_[conn.op_list_name_];
            CombineOpAndGraph(conn,graph,ops);
        }
        //get optimizer_map
        for(int i=0;i<optimizer_list.optimizer_list_.size();++i){
            OptimizerParams &optp = optimizer_list.optimizer_list_[i];
            std::shared_ptr<Optimizer> opt = OptimizerFactory::Build(optp); 
            opt_map_.insert(std::pair<std::string,std::shared_ptr<Optimizer>>(optp.graph_name_,opt));
        }

        //set optimizer for graph
        for(auto tmp:graph_map_){
            if(opt_map_.find(tmp.first) == opt_map_.end()){
                Log::v(TAG,"not set optimizer for "+tmp.first);
                continue;
            }
            tmp.second->set_optimizer(opt_map_[tmp.first].get());
        }
        //set up
        for(auto tmp:graph_map_){
            tmp.second->Setup(false);
        }
        //build net
        net_ = std::make_shared<Net>(net_params);
        for(auto tmp:graph_list.graph_list_){
            net_->AddOps(graph_map_[tmp.name_].get());
        }
        Log::i(TAG,"parse over");
    }
    
} // namespace Grape

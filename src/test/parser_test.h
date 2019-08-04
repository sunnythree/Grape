#include "gtest/gtest.h"
#include "grape/parse/parser.h"
#include "grape/params/net_params.h"
#include "grape/params/connection_list_params.h"
#include "grape/params/graph_list_params.h"
#include "grape/params/op_list_params.h"
#include "grape/params/op_path_params.h"
#include "grape/params/optimizer_list_params.h"

using namespace Grape;

TEST(paser,net)
{
    NetParams net_params;
    net_params.max_iter_ = 1000;
    Parser::Serialize("obj/test_net_params.json",net_params);

    NetParams net_params1;
    Parser::Parse("obj/test_net_params.json",net_params1);
    EXPECT_EQ(net_params1.max_iter_,1000);
}

TEST(paser,op_list_only_parse)
{
    OpListParams ops;
    Parser::Parse("src/test/data/test_op_list_params.json",ops);
    for(int i=0;i<3;i++){
        OpParams &opp = ops.op_list_[i];
        EXPECT_EQ(opp.batch_,10);
        EXPECT_EQ(opp.in_dim_,i);
        EXPECT_EQ(opp.out_dim_,i);
        EXPECT_EQ(opp.name_,"name"+std::to_string(i));
        EXPECT_EQ(opp.has_bias_,true);
    }
}

TEST(paser,op_list)
{
    OpListParams ops;
    for(int i=0;i<3;i++){
        OpParams opp;
        opp.batch_ = 10;
        opp.in_dim_ = i;
        opp.out_dim_ = i;
        opp.name_ = "name"+std::to_string(i);
        opp.has_bias_ = true;
        ops.op_list_.push_back(opp);
    }
    Parser::Serialize("obj/test_op_list_params.json",ops);

    OpListParams ops1;
    Parser::Parse("obj/test_op_list_params.json",ops1);
    for(int i=0;i<3;i++){
        OpParams &opp = ops1.op_list_[i];
        EXPECT_EQ(opp.batch_,10);
        EXPECT_EQ(opp.in_dim_,i);
        EXPECT_EQ(opp.out_dim_,i);
        EXPECT_EQ(opp.name_,"name"+std::to_string(i));
        EXPECT_EQ(opp.has_bias_,true);
    }
}

TEST(paser,graph_list)
{
    GraphListParams graph_list;
    for(int i=0;i<3;i++){
        GraphParams graph_param;
        graph_param.cal_mode_ = "cpu";
        graph_param.device_id_ = i;
        graph_param.max_iter_ = i*100;
        graph_param.name_ = "name"+std::to_string(i);
        graph_param.phase_ = "train";
        graph_param.save_path_ = "test";
        graph_list.graph_list_.emplace_back(graph_param);
    }
    Parser::Serialize("obj/test_graph_list_params.json",graph_list);

    GraphListParams graph_list1;
    Parser::Parse("obj/test_graph_list_params.json",graph_list1);
    for(int i=0;i<3;i++){
        GraphParams &graph_param = graph_list1.graph_list_[i];
        EXPECT_EQ(graph_param.cal_mode_ , "cpu");
        EXPECT_EQ(graph_param.device_id_ , i);
        EXPECT_EQ(graph_param.max_iter_ , i*100);
        EXPECT_EQ(graph_param.name_ , "name"+std::to_string(i));
        EXPECT_EQ(graph_param.phase_ , "train");
        EXPECT_EQ(graph_param.save_path_ , "test");
    }
}

TEST(paser,connection_list)
{
    ConnectionListParams connection_list;
    for(int i=0;i<3;i++){
        ConnectionParams connection;
        connection.graph_name_ = "graph"+std::to_string(i);
        connection.op_list_name_ = "op_list"+std::to_string(i);
        for(int i=0;i<3;i++){
            Conn conn;
            conn.from = "from:"+std::to_string(i);
            conn.to = "to:"+std::to_string(i);
            connection.connections_.push_back(conn);
        }
        connection_list.connection_list_.emplace_back(connection);
    }
    Parser::Serialize("obj/test_connection_list_params.json",connection_list);

    ConnectionListParams connection_list1;
    Parser::Parse("obj/test_connection_list_params.json",connection_list1);
    for(int i=0;i<3;i++){
        ConnectionParams &connection = connection_list1.connection_list_[i];
        EXPECT_EQ(connection.graph_name_ , "graph"+std::to_string(i));
        EXPECT_EQ(connection.op_list_name_ , "op_list"+std::to_string(i));
        for(int i=0;i<3;i++){
            Conn &conn = connection.connections_[i];
            EXPECT_EQ(conn.from , "from:"+std::to_string(i));
            EXPECT_EQ(conn.to , "to:"+std::to_string(i));
        }
    }
}

TEST(paser,optimizer_list)
{
    OptimizerListParams optimizer_list;
    for(int i=0;i<3;i++){
        OptimizerParams op;
        op.type_ = "sgd",
        op.lr_ = 0.01f;
        optimizer_list.optimizer_list_.emplace_back(op);
    }
    Parser::Serialize("obj/test_optimizer_list_params.json",optimizer_list);

    OptimizerListParams optimizer_list1;
    Parser::Parse("obj/test_optimizer_list_params.json",optimizer_list1);
    for(int i=0;i<3;i++){
        OptimizerParams &op = optimizer_list1.optimizer_list_[i];
        EXPECT_EQ(op.type_ , "sgd");
        EXPECT_EQ(op.lr_,0.01f);
    }
}

TEST(paser,op_path)
{
    OpPathParams oppath;
    for(int i=0;i<3;i++){
        NamePathPair npp;
        npp.name = "name:"+std::to_string(i);
        npp.path = "path:"+std::to_string(i);
        oppath.path_list_.emplace_back(npp);
    }
    Parser::Serialize("obj/test_oppath_params.json",oppath);

    OpPathParams oppath1;
    Parser::Parse("obj/test_oppath_params.json",oppath1);
    for(int i=0;i<3;i++){
        NamePathPair &npp = oppath1.path_list_[i];
        EXPECT_EQ(npp.name , "name:"+std::to_string(i));
        EXPECT_EQ(npp.path , "path:"+std::to_string(i));
    }
}

TEST(paser,parse_all)
{
    NetParams net_params;
    GraphListParams graph_list;
    ConnectionListParams connection_list;
    OpPathParams op_path;
    OptimizerListParams optimizer_list;
    Parser::Parse("src/test/data/test.json",op_path,optimizer_list,graph_list,connection_list,net_params);
    
    //path
    EXPECT_EQ(op_path.path_list_[0].name,"mnist");
    EXPECT_EQ(op_path.path_list_[0].path,"src/test/data/test_op.json");

    //graph
    EXPECT_EQ(graph_list.graph_list_[0].name_,"graph0");
    EXPECT_EQ(graph_list.graph_list_[0].max_iter_,100);
    EXPECT_EQ(graph_list.graph_list_[0].cal_mode_,"cpu");
    EXPECT_EQ(graph_list.graph_list_[0].phase_,"train");
    EXPECT_EQ(graph_list.graph_list_[0].save_path_,"test");

    //connections
    EXPECT_EQ(connection_list.connection_list_[0].graph_name_,"graph0");
    EXPECT_EQ(connection_list.connection_list_[0].op_list_name_,"mnist");
    EXPECT_EQ(connection_list.connection_list_[0].connections_[0].from,"name0:0");
    EXPECT_EQ(connection_list.connection_list_[0].connections_[0].to,"name1:0");
    EXPECT_EQ(connection_list.connection_list_[0].connections_[1].from,"name1:0");
    EXPECT_EQ(connection_list.connection_list_[0].connections_[1].to,"name2:0");

    //optimizer
    EXPECT_EQ(optimizer_list.optimizer_list_[0].lr_,0.1f);
    EXPECT_EQ(optimizer_list.optimizer_list_[0].type_,"sgd");

    //net
    EXPECT_EQ(net_params.max_iter_,1000);
}

TEST(paser,json_to_code)
{
    Parser parser;
    parser.Parse("src/test/data/test.json");
}

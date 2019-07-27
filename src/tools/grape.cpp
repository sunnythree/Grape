#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <assert.h>
#include "grape/op/fc.h"
#include "grape/tensor.h"
#include "grape/op/mnist_data.h"
#include "grape/op/softmax_with_loss.h"
#include "grape/op/softmax.h"
#include "grape/graph.h"
#include "grape/optimizer/sgd.h"
#include "grape/net.h"
#include "grape/op/accuracy_test.h"
#include "cereal/types/vector.hpp"
#include "grape/parse/parser.h"
#include "grape/op_factory.h"
#include "grape/graph_factory.h"

using namespace std;
using namespace Grape;

void code_net(){
   int batch = 100;
    MnistData input("input","data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",batch,false,50000);      
    Fc fc1("fc1",batch,784,100);
    Fc fc2("fc2",batch,100,30);
    Fc fc3("fc3",batch,30,10);
    SoftmaxWithLoss sm("sm",batch,10);
    
    
    input<<fc1<<fc2<<fc3<<sm;
    connect_op(&input,&sm,1,1);
    
    Graph graph("data/test",JSON,1000,TRAIN,CPU_MODE);
    graph.set_phase(TRAIN);
    graph.SetMaxIter(500);
    graph.Construct({&input},{&sm});
    graph.Setup(false);

    NetParams params;
    Net net(params);
    net.AddOps(&graph);

    MnistData input_test("input","data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",batch,false,10000);      
    Fc fc1_t("fc1",batch,784,100);
    Fc fc2_t("fc2",batch,100,30);
    Fc fc3_t("fc3",batch,30,10);
    Softmax sm1("sm",batch,10);
    AccuracyTest accuracy("test",batch,10);

    input_test<<fc1_t<<fc2_t<<fc3_t<<sm1<<accuracy;
    connect_op(&input_test,&accuracy,1,1);

    Graph graph1("data/test",JSON,100,TEST,CPU_MODE);
    graph1.set_phase(TEST);
    graph1.SetMaxIter(100);
    graph1.Construct({&input_test},{&accuracy});
    
    net.AddOps(&graph1);
    
    net.Run();
}

void combine_oplist_graph(
    std::map<std::string,std::vector<std::shared_ptr<Op>>> &op_map,
    std::map<std::string,std::shared_ptr<Graph>> &graph_map,
    ConnectionListParams &connection_list
)
{
    assert(op_map.size()==graph_map.size());
    
}

void json_net(std::string path)
{
  
}

int main(int argc,char **argv)
{
    if(argc != 2){
        std::cout<<"usage: ./Grape cfb_file"<<std::endl;
    }
    json_net(argv[1]);
    
    return 0;
}
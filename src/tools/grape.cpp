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
#include "grape/log.h"

using namespace std;
using namespace Grape;

void code_net(){
   int batch = 20;
    MnistData input_train("input_train","data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",batch,true,50000);  
    MnistData input_test("input_test","data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",batch,false,10000);      
    Fc fc1("fc1",batch,784,100);
    Fc fc2("fc2",batch,100,30);
    Fc fc3("fc3",batch,30,10);
    AccuracyTest accuracy("test",batch,10);
    SoftmaxWithLoss sml("sml",batch,10);
    Softmax sm("sm",batch,10);
    
    
    input_train<<fc1<<fc2<<fc3<<sml;
    connect_op(&input_train,&sml,1,1);
    
    Graph graph("data/test",JSON,2500,100,500,TRAIN,GPU_MODE);
    graph.set_phase(TRAIN);
    graph.Construct({&input_train},{&sml});
    graph.Setup(false);
    SGDOptimizer sgd(0.01);
    sgd.set_momentum(0.9);
    sgd.set_policy(POLICY_STEP);
    sgd.set_step(5000);
    sgd.set_gamma(0.9);
    graph.set_optimizer(&sgd);

    NetParams params;
    params.max_iter_ = 30;
    Net net(params);
    net.AddOps(&graph);
    

    input_test<<fc1<<fc2<<fc3<<sm<<accuracy;
    connect_op(&input_test,&accuracy,1,1);
    Graph graph1("data/test",JSON,500,500,0,TEST,GPU_MODE);
    graph1.set_phase(TEST);
    graph1.Construct({&input_test},{&accuracy});
    net.AddOps(&graph1);
    net.Run();
}


int main(int argc,char **argv)
{
    if(argc != 2){
        std::cout<<"usage: ./Grape cfb_file"<<std::endl;
        return -1;
    }
    Log::set_log_level(VERBOSE);
    Parser parser;
    parser.Parse(argv[1]);
    
    Net *net =  parser.get_net().get();
    net->Run();
    //code_net();
    return 0;
}
#include <iostream>
#include <string>
#include "grape/op/fc.h"
#include "grape/tensor.h"
#include "grape/op/mnist_data.h"
#include "grape/op/softmax_with_loss.h"
#include "grape/op/softmax.h"
#include "grape/graph.h"
#include "grape/optimizer/sgd.h"
#include "grape/net.h"
#include "grape/op/accuracy_test.h"

using namespace std;
using namespace Grape;

int main()
{
    int batch = 100;
    MnistData input("input","data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",batch,false,50000);      
    Fc fc1("fc1",batch,784,100);
    Fc fc2("fc2",batch,100,30);
    Fc fc3("fc3",batch,30,10);
    SoftmaxWithLoss sm("sm",batch,10);
    
    
    input<<fc1<<fc2<<fc3<<sm;
    connect_op(&input,&sm,1,1);
    
    Graph graph("data/test",JSON,SGD,0.1f);
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

    Graph graph1("data/test",JSON,SGD,0.1f);
    graph1.set_phase(TEST);
    graph1.SetMaxIter(100);
    graph1.Construct({&input_test},{&accuracy});
    
    net.AddOps(&graph1);
    
    net.Run();

    return 0;
}
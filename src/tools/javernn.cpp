#include <iostream>
#include <string>
#include "javernn/graph.h"
#include "javernn/op/fc.h"
#include "javernn/tensor.h"
#include "javernn/op/mnist_data.h"
#include "javernn/op/softmax_with_loss.h"
#include "javernn/op/softmax.h"
#include "javernn/graph.h"
#include "javernn/optimizer/sgd.h"
#include "javernn/net.h"
#include "javernn/op/accuracy_test.h"

using namespace std;
using namespace javernn;

int main()
{
    int batch = 100;
    MnistData input("input","data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",batch,true,50000);      
    Fc fc1("fc1",batch,784,100);
    Fc fc2("fc2",batch,100,30);
    Fc fc3("fc3",batch,30,10);
    SoftmaxWithLoss sm("sm",batch,10);
    
    
    input<<fc1<<fc2<<fc3<<sm;
    connect_op(&input,&sm,1,1);
    

    NetParams params;
    params.max_train_iters_ = 5000;
    Net net(params);
    net.Construct({&input},{&sm});
    net.Train();
    net.Save();

    MnistData input_test("input","data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",batch,false,10000);      
    Fc fc1_t("fc1",batch,784,100);
    Fc fc2_t("fc2",batch,100,30);
    Fc fc3_t("fc3",batch,30,10);
    Softmax sm1("sm",batch,10);
    AccuracyTest accuracy("test",batch,10);

    input_test<<fc1_t<<fc2_t<<fc3_t<<sm1<<accuracy;
    connect_op(&input_test,&accuracy,1,1);

    NetParams params1;
    params1.max_train_iters_ = 100;
    Net net1(params1);
    net1.Construct({&input_test},{&accuracy});
    net1.Load();
    net1.Test();

    return 0;
}
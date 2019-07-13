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
    //std::cout<<"hello world"<<std::endl;
    // int batch = 20;
    // MnistData input("input","data/train-images-idx3-ubyte","data/train-labels-idx1-ubyte",batch);      
    // Fc fc1("fc1",batch,784,64),fc2("fc2",batch,64,10);
    // SoftmaxWithLoss sm("sm",batch,10);
    
    
    // input<<fc1<<fc2<<sm;
    // connect_op(&input,&sm,1,1);
    

    // NetParams params;
    // params.max_train_iters_ = 1000;
    // Net net(params);
    // net.Construct({&input},{&sm});
    // net.Train();
    // net.Save();

    int batch = 20;
    MnistData input("input","data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte",batch);      
    Fc fc1("fc1",batch,784,64),fc2("fc2",batch,64,10);
    Softmax sm("sm",batch,10);
    AccuracyTest accuracy("test",batch,10);

    input<<fc1<<fc2<<sm<<accuracy;
    connect_op(&input,&accuracy,1,1);

    NetParams params;
    params.max_train_iters_ = 500;
    Net net(params);
    net.Construct({&input},{&sm});
    net.Load();
    net.Test();
    
    return 0;
}
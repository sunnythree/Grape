#include <iostream>
#include <string>
#include "javernn/graph.h"
#include "javernn/op/fc.h"
#include "javernn/tensor.h"
#include "javernn/op/mnist_data.h"
#include "javernn/op/softmax_with_loss.h"
#include "javernn/graph.h"
#include "javernn/optimizer/sgd.h"
#include "javernn/net.h"

using namespace std;
using namespace javernn;

int main()
{
    //std::cout<<"hello world"<<std::endl;
    int batch = 30;
    MnistData input("data/train-images-idx3-ubyte","data/train-labels-idx1-ubyte",batch);      
    Fc fc1(batch,784,64),fc2(batch,64,10);
    SoftmaxWithLoss sm(batch,10);
    
    
    input<<fc1<<fc2<<sm;
    connect_op(&input,&sm,1,1);
    

    NetParams params;
    params.max_train_iters_ = 10000;
    Net net(params);
    net.Construct({&input},{&sm});
    net.Train();

    return 0;
}
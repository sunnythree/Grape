#include <iostream>
#include <string>
#include "javernn/graph.h"
#include "javernn/op/fc.h"
#include "javernn/tensor.h"
#include "javernn/op/binary_data.h"
#include "javernn/op/softmax_with_loss.h"
#include "javernn/graph.h"
#include "javernn/optimizer/sgd.h"
#include "javernn/net.h"

using namespace std;
using namespace javernn;

int main()
{
    //std::cout<<"hello world"<<std::endl;
    int batch = 100;
    BinaryData input("data/train-images-idx3-ubyte",batch,784,784,16);
    BinaryData label("data/train-labels-idx1-ubyte",batch,1,10,8,true);
    Fc fc1(batch,784,64),fc2(batch,64,10);
    SoftmaxWithLoss sm(batch,10);
    std::vector<Op *> tuple1 = (fc2,label);


    input<<fc1<<fc2;
    tuple1<<sm;

    NetParams params;
    params.max_train_iters_ = 100;
    Net net(params);
    net.Construct({&input},{&sm});
    net.Train();

    return 0;
}
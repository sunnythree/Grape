#include <iostream>
#include <string>
#include "javernn/graph.h"
#include "javernn/op/fc.h"
#include "javernn/tensor.h"
#include "javernn/op/binary_data.h"
#include "javernn/op/softmax_with_loss.h"
#include "javernn/graph.h"
#include "javernn/optimizer/sgd.h"

using namespace std;
using namespace javernn;

int main()
{
    //std::cout<<"hello world"<<std::endl;
    int batch = 100;
    BinaryData input("data/train-images-idx3-ubyte",batch,784,784,16);
    BinaryData label("data/train-labels-idx1-ubyte",batch,1,10,8,true);
    Fc fc1(batch,784,10),fc2(batch,10,10);
    SoftmaxWithLoss sm(batch,10);
    std::vector<Op *> tuple1 = (fc2,label);


    input<<fc1<<fc2;
    tuple1<<sm;

    Graph graph;
    graph.Construct({&input,&label},{&sm});
    int i=10;
    SGDOptimizer opt(0.01);
    while(--i>0){
        graph.Forward();
        graph.Backward();
        graph.UpdateWeights(opt);
    }

    return 0;
}
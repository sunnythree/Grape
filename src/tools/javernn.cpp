#include <iostream>
#include <string>
#include "javernn/graph.h"
#include "javernn/op/fc.h"
#include "javernn/tensor.h"
#include "javernn/op/binary_data.h"
#include "javernn/op/softmax_with_loss.h"
#include "javernn/graph.h"

using namespace std;
using namespace javernn;

int main()
{
    std::cout<<"hello world"<<std::endl;
    BinaryData input("data/train-images-idx3-ubyte",20,784,16);
    BinaryData label("data/train-labels-idx1-ubyte",20,10,8);
    Fc fc1(20,100,30),fc2(20,30,10);
    SoftmaxWithLoss sm(20,10);
    std::vector<Op *> tuple1 = (fc2,label);


    input<<fc1<<fc2;
    tuple1<<sm;
    //std::cout<<"2222 "<<std::endl;
    Graph graph;
    graph.Construct({&input,&label},{&sm});
    //std::cout<<"3333 "<<std::endl;
    graph.Forward();
    //std::cout<<"4444 "<<std::endl;
    graph.Backward();
    //std::cout<<"5555 "<<std::endl;
    return 0;
}
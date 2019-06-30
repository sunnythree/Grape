#include <iostream>
#include "javernn/graph.h"
#include "javernn/op/fc.h"
#include "javernn/tensor.h"
#include "javernn/op/input.h"
#include "javernn/op/softmax_with_loss.h"
#include "javernn/graph.h"

using namespace std;
using namespace javernn;

int main()
{
    std::cout<<"hello world"<<std::endl;
    Fc fc1(1,1,1),fc2(1,1,1);
    Input input(1,1);
    Input label(1,1);
    SoftmaxWithLoss sm(1,1);
    std::vector<Op *> tuple1 = (fc2,label);


    std::cout<<"11111"<<std::endl;
    input<<fc1<<fc2;
    tuple1<<sm;
    std::cout<<"2222 "<<std::endl;
    Graph graph;
    graph.Construct({&input,&label},{&sm});
    graph.Forward();
    graph.Backward();
    return 0;
}
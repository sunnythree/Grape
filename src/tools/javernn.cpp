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
    input<<fc1<<fc2<<sm;
    connect_op(&label,&sm,0,1);

    Graph graph;
    graph.Construct({&input,&label},{&sm});
    graph.Forward();
    graph.Backward();
    return 0;
}
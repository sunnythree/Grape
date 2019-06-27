#include <iostream>
#include "javernn/graph.h"
#include "javernn/sequence.h"
#include "javernn/op/fc.h"
#include "javernn/tensor.h"
#include "javernn/op/input.h"
#include "javernn/op/softmax_with_loss.h"

using namespace std;
using namespace javernn;
int main()
{
    std::cout<<"hello world"<<std::endl;

    Fc fc1(1,1,1),fc2(1,1,1);
    Input input(1,1);
    SoftmaxWithLoss sm(1,1);
    Sequence s;
    s.Add(&input);
    s.Add(&fc1);
    s.Add(&fc2);
    s.Add(&sm);
    s.Construct();
    std::vector<Tensor> in;
    s.Forward(in);
    std::cout<<"end"<<std::endl;
    //s.Backward(in);
    return 0;
}
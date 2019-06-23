#include <iostream>
#include "javernn/graph.h"
#include "javernn/sequence.h"
#include "javernn/op/fc.h"
#include "javernn/tensor.h"

using namespace std;
using namespace javernn;
int main()
{
    std::cout<<"hello world"<<std::endl;
    Fc fc1(1,1,1),fc2(1,1,1),fc3(1,1,1);
    Sequence s;
    s.Add(&fc1);
    s.Add(&fc2);
    s.Add(&fc3);
    std::vector<Tensor> in;
    s.Forward(in);
    s.Backward(in);
    return 0;
}
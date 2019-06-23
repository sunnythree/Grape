#include <iostream>
#include "javernn/graph.h"
#include "javernn/op/fc.h"
#include "javernn/tensor.h"
#include "net_params_test.h"

using namespace std;
using namespace javernn;
int main()
{
    // std::cout<<"hello world"<<std::endl;
    // Fc fc1(1,1,1),fc2(1,1,1),fc3(1,1,1);
    // fc1<<fc2<<fc3;
    // Graph graph;
    // graph.Construct({&fc1},{&fc3});
    // std::vector<Tensor> in;
    // graph.Forward(in);
    // graph.Backward(in);
    
    //serialize_net_params();
    parse_net_params();
    return 0;
}
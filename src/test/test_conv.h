#include "gtest/gtest.h"
#include "grape/op/conv2d.h"

using namespace Grape;

TEST(conv,conv2d)
{
    Conv2d conv("conv2d",1,1,4,4,1,1,3,1,0,true,LEAKY);
    std::shared_ptr<Tensor> tmp = std::make_shared<Tensor>(static_cast<Op *>(&conv),
            Shape({1,1,4,4}),DATA,sizeof(float));
    conv.Setup();
    std::vector<tensorptr_t> & tttt = (std::vector<tensorptr_t> &)conv.prev();
    tttt[0] = tmp;
    float a[16] = {
        1,1,1,-1,
        0,0,0,-1,
        1,1,1,-1,
        2,2,2,-1
    };
    float *in_cpu_data = (float *)tmp->mutable_cpu_data();
    for(int i=0;i<16;i++){
        in_cpu_data[i] = a[i];
    }
    printf("in_cpu_data\n");
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            printf("%f ",in_cpu_data[i*4+j]);
        }
        printf("\n");
    }
    printf("\n");
    conv.ForwardCpu();
    const std::vector<tensorptr_t> & ttttt = conv.next();
    Tensor* output = ttttt[0].get();
    float *out_cpu_data = (float *)output->cpu_data();
    printf("out_cpu_data\n");
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            printf("%f ",out_cpu_data[i*2+j]);
        }
        printf("\n");
    }
    printf("\n");
}

#include "gtest/gtest.h"
#include "grape/op/pool_max.h"

using namespace Grape;

TEST(pool,max)
{
    PoolMax pm("poolmax",1,1,4,4,2,2,0);
    std::shared_ptr<Tensor> tmp = std::make_shared<Tensor>(static_cast<Op *>(&pm),
            Shape({1,1,4,4}),DATA,sizeof(float));
    std::vector<tensorptr_t> & tttt = (std::vector<tensorptr_t> &)pm.prev();
    tttt[0] = tmp;
    float a[16] = {
        1,2,3,4,
        5,6,7,8,
        9,0,1,2,
        3,4,5,6
    };
    float *in_cpu_data = (float *)tmp->mutable_cpu_data();
    for(int i=0;i<16;i++){
        in_cpu_data[i] = i;
    }
    printf("in_cpu_data\n");
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            printf("%f ",in_cpu_data[i*4+j]);
        }
        printf("\n");
    }
    printf("\n");
    pm.ForwardCpu();
    const std::vector<tensorptr_t> & ttttt = pm.next();
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

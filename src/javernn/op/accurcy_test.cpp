 #include "javernn/op/accuracy_test.h"
 #include "javernn/log.h"
 #include "javernn/util/util.h"

 namespace javernn{
    static std::string TAG = "AccuracyTest";

    AccuracyTest::AccuracyTest(std::string name, uint32_t batch_size, uint32_t in_dim):
    Op({DATA,DATA},{}),
    batch_size_(batch_size),
    in_dim_(in_dim)
    {
        type_ = "AccuracyTest";
        name_ = name;
    }
    AccuracyTest::~AccuracyTest()
    {

    }
    void AccuracyTest::Setup()
    {

    }
    void AccuracyTest::ForwardCpu()
    {
        Log::v(TAG,"ForwardCpu");
        Tensor *intput_tensor = prev_[0].get();
        Tensor *label_tensor = prev_[1].get();
        const float *intput_data = (const float *)intput_tensor->cpu_data();
        const float *labal_data = (const float *)label_tensor->cpu_data();
        int data_index = 0;
        int label_index = 0;
        for(int i=0;i<batch_size_;i++){
            data_index = max_index(intput_data+i*in_dim_,in_dim_);
            label_index = max_index(labal_data+i*in_dim_,in_dim_);
            if(data_index == label_index)accuracy_count_++;
            all_count++;
        }
        accuracy_ = accuracy_count_/(float)all_count;
    }
    void AccuracyTest::BackwardCpu()
    {

    }
    void AccuracyTest::UpdateWeightsCpu(Optimizer &opt)
    {

    }

#ifdef GPU
    void AccuracyTest::ForwardGpu()
    {

    } 
    void AccuracyTest::BackwardGpu()
    {

    }
    void AccuracyTest::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
 }
 
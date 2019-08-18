 #include "grape/op/accuracy_test.h"
 #include "grape/log.h"
 #include "grape/util/util.h"
 #include "grape/global_config.h"

 namespace Grape{
    const static std::string TAG = "AccuracyTest";

    AccuracyTest::AccuracyTest(std::string name, uint32_t batch_size, uint32_t in_dim):
    Op({DATA,DATA},{}),
    batch_size_(batch_size),
    in_dim_(in_dim)
    {
        type_ = STRING_ACCURACY_TEST_TYPE;
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
        Tensor *intput_tensor = prev_[0].get();
        Tensor *label_tensor = prev_[1].get();
        const float *intput_data = (const float *)intput_tensor->cpu_data();
        const float *labal_data = (const float *)label_tensor->cpu_data();
        int data_index = 0;
        int label_index = 0;
        for(int i=0;i<batch_size_;i++){
            data_index = max_index(intput_data+i*in_dim_,in_dim_);
            label_index = max_index(labal_data+i*in_dim_,in_dim_);
            if(data_index == label_index){
                accuracy_count_++;
            }
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

    void AccuracyTest::OnTestBegin()
    {
        accuracy_count_ = 0;
        all_count = 0;
        accuracy_ = 0;
    }
    
    void AccuracyTest::Display()
    {
        Log::v(TAG,"accuracy_ "+std::to_string(accuracy_)
            +" accuracy_count_ "+std::to_string(accuracy_count_)
            +" all_count "+std::to_string(all_count)
            );
    };

#ifdef GPU
    void AccuracyTest::ForwardGpu()
    {
        ForwardCpu();
    } 
    void AccuracyTest::BackwardGpu()
    {

    }
    void AccuracyTest::UpdateWeightsGpu(Optimizer &opt)
    {

    }
#endif
 }
 
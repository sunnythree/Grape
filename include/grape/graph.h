#ifndef __GRAPE_Graph_H__
#define __GRAPE_Graph_H__

#include "grape/ops.h"
#include "grape/global_config.h"

namespace Grape{
    class Graph:public Ops{
    public:
        explicit Graph(std::string save_path, SERIALIZE_TYPE serialize_type,
        OPTIMIZER_TYPE optimizer_type, float lr);
        virtual ~Graph();
        void Backward(void);
        void Forward();  
        void UpdateWeights();
        void Setup(bool load);
        void Construct(const std::vector<Op *> &input,
                 const std::vector<Op *> &output);
        int32_t FindIndex(const std::vector<Op *> &ops, Op *target);
        void TrainOnce();
        void Train();
        void TestOnce();
        void Test();
        void Run();
        void RunOnce();
        void Save();
        void Load();
        PHASE GetPhase();
        void SetPhase(PHASE phase);
        uint32_t GetMaxIter();
        void SetMaxIter(uint32_t iter);
        inline CAL_MODE get_cal_mode(){return cal_mode_;};
        inline void set_cal_mode(CAL_MODE mode){cal_mode_ = mode;};
        inline PHASE get_phase(){return graph_phase_;};
        inline void set_phase(PHASE phase){graph_phase_ = phase;};
    private:
        std::vector<Op *> ops_;
        std::vector<Op *> input_ops_;
        std::vector<Op *> output_ops_;
        std::string save_path_ = ".";
        SERIALIZE_TYPE serialize_type_ = BINARY;
        OPTIMIZER_TYPE optimizer_type_ = SGD;
        CAL_MODE cal_mode_ = CPU_MODE;
        std::shared_ptr<Optimizer> optimizer_;
        float lr_ = 0.1f;
        uint32_t max_iter_ = 0;
        PHASE graph_phase_ = TRAIN;
    };
}

#endif
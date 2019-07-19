#ifndef __JAVERNN_GRAPH_H__
#define __JAVERNN_GRAPH_H__

#include "javernn/ops.h"
#include "javernn/global_config.h"

namespace javernn{
    class Graph:public Ops{
    public:
        explicit Graph(std::string save_path, SERIALIZE_TYPE serialize_type,
        OPTIMIZER_TYPE optimizer_type, float lr);
        virtual ~Graph();
        void Backward(void);
        void Forward();  
        void UpdateWeights();
        void Setup();
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
        uint32_t GetMaxIter();
        GRAPH_PHASE GetGraphPhase();
    private:
        std::vector<Op *> ops_;
        std::vector<Op *> input_ops_;
        std::vector<Op *> output_ops_;
        std::string save_path_ = ".";
        SERIALIZE_TYPE serialize_type_ = BINARY;
        OPTIMIZER_TYPE optimizer_type_ = SGD;
        std::shared_ptr<Optimizer> optimizer_;
        float lr_ = 0.1f;
        uint32_t max_iter_ = 0;
        GRAPH_PHASE graph_phase_ = GRAPH_TRAIN;
    };
}

#endif
#ifndef __GRAPE_Graph_H__
#define __GRAPE_Graph_H__

#include "grape/ops.h"
#include "grape/global_config.h"
#include "grape/params/graph_params.h"

namespace Grape{
    class Graph:public Ops{
    public:
        explicit Graph(
            std::string save_path,
            SERIALIZE_TYPE serialize_type,
            int32_t max_iter,
            int32_t display_iter,
            int32_t snapshot_iter,
            PHASE graph_phase,
            CAL_MODE cal_mode
        );
        explicit Graph(
            GraphParams &graph_params
        );
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
        void Save(std::string path);
        void Load(std::string path);
        void OnNetRunBegin();
        void OnNetRunEnd();
        PHASE GetPhase();
        void SetPhase(PHASE phase);
        uint32_t GetMaxIter();
        void SetMaxIter(uint32_t iter);
        inline CAL_MODE get_cal_mode(){return cal_mode_;};
        inline void set_cal_mode(CAL_MODE mode){cal_mode_ = mode;};
        inline PHASE get_phase(){return graph_phase_;};
        inline void set_phase(PHASE phase){graph_phase_ = phase;};
        inline Optimizer *get_optimizer(){return optimizer_;};
        inline void set_optimizer(Optimizer *optimizer){optimizer_=optimizer;};
        
    private:
        std::vector<Op *> ops_;
        std::vector<Op *> input_ops_;
        std::vector<Op *> output_ops_;
        std::string save_path_ = "";
        SERIALIZE_TYPE serialize_type_ = BINARY;
        CAL_MODE cal_mode_ = CPU_MODE;
        uint32_t max_iter_ = 1;
        uint32_t display_iter_ = 1;
        uint32_t snapshot_iter_ = 10000;
        uint32_t device_id_ = 0;
        PHASE graph_phase_ = TRAIN;
        Optimizer *optimizer_;
        std::vector<OpConnectionPoint> connections_;
        void SnapShotConnections();
        void GetConnection(Op *op,std::vector<OpConnectionPoint> &connections);
        void ReConnection();
        uint32_t run_iter_ = 0;
    };
}

#endif
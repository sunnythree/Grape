#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>
#include "grape/graph.h"
#include "grape/error.h"
#include "grape/global_config.h"
#include "grape/log.h"
#include "grape/optimizer/sgd.h"

namespace Grape{
    static std::string TAG = "Grape";

    Grape::Grape(std::string save_path,SERIALIZE_TYPE serialize_type,OPTIMIZER_TYPE optimizer_type,float lr):
    save_path_(save_path),
    serialize_type_(serialize_type),
    optimizer_type_(optimizer_type),
    lr_(lr)
    {
        switch (optimizer_type_)
        {
        case SGD:
            optimizer_ = std::make_shared<SGDOptimizer>(0.1f);
            break;
        
        default:
            break;
        }
    }

    Grape::~Grape()
    {

    }


    void Grape::Backward()
    {
        for(int i=ops_.size()-1;i>-1;--i){
            if(cal_mode_ == CPU_MODE){
                ops_[i]->BackwardCpu();
            }else{
#ifdef GPU
                ops_[i]->BackwardGpu();
#endif
            }
        }
    }
    void Grape::Forward()
    {
        Log::v(TAG,"Forward");
        for(auto o:ops_){
            if(cal_mode_ == CPU_MODE){
                o->ForwardCpu();
            }else{
#ifdef GPU
                o->ForwardGpu();
#endif
            }
        }
    } 
    void Grape::UpdateWeights()
    {
        for(auto o:ops_){
            if(cal_mode_ == CPU_MODE){
                o->UpdateWeightsCpu(*optimizer_.get());
            }else{
#ifdef GPU
                o->UpdateWeightsGpu(*optimizer_.get());
#endif
            }
        }
    }
    void Grape::Setup()
    {
        for(auto o:ops_){
            o->Setup();
        }
    }

    void Grape::TrainOnce()
    {
        Forward();
        Backward();
        UpdateWeights();
    }

    void Grape::Train()
    {
        for(uint32_t i=0;i<max_iter_;i++){
            TrainOnce();
        }
    }

    void Grape::TestOnce()
    {
        Forward();
    }

    void Grape::Test()
    {
        for(uint32_t i=0;i<max_iter_;i++){
            Forward();
        }
    }

    void Grape::Run()
    {
        if(grape_phase_ == GRAPH_TRAIN){
            Train();
        }else{
            Test();
        }
    }

    void Grape::RunOnce()
    {
        if(grape_phase_ == GRAPH_TRAIN){
            TrainOnce();
        }else{
            TestOnce();
        }
    }

    void Grape::Construct(const std::vector<Op *> &inputs,
                    const std::vector<Op *> &outputs) {
        std::vector<Op *> sorted;
        std::vector<Op *> input_nodes(inputs.begin(), inputs.end());
        std::unordered_map<Op *, std::vector<uint8_t>> removed_edge;

        // topological-sorting
        while (!input_nodes.empty()) {
            sorted.push_back(dynamic_cast<Op *>(input_nodes.back()));
            input_nodes.pop_back();

            Op *curr              = sorted.back();
            std::vector<Op *> next = curr->NextOps();

            for (size_t i = 0; i < next.size(); i++) {
                if (!next[i]) continue;
                // remove edge between next[i] and current
                if (removed_edge.find(next[i]) == removed_edge.end()) {
                    removed_edge[next[i]] =
                        std::vector<uint8_t>(next[i]->PrevOps().size(), 0);
                }

                std::vector<uint8_t> &removed = removed_edge[next[i]];
                removed[FindIndex(next[i]->PrevOps(), curr)] = 1;

                if (std::all_of(removed.begin(), removed.end(),
                                [](uint8_t x) { return x == 1; })) {
                    input_nodes.push_back(next[i]);
                }
            }
        }
        int num = 0;
        for (auto &n : sorted) {
            ops_.push_back(n);
        }

        input_ops_  = inputs;
        output_ops_ = outputs;
        Setup();
    }

    int32_t Grape::FindIndex(const std::vector<Op *> &ops, Op *target) 
    {
        for (int32_t i = 0; i < ops.size(); i++) {
            if (ops[i] == static_cast<Op *>(&*target)) return i;
        }
        throw Error("invalid connection");
    }

    void Grape::Save()
    {
        for(auto o:ops_){
            o->Save(save_path_,serialize_type_);
        }
    }

    void Grape::Load()
    {
        for(auto o:ops_){
            o->Load(save_path_,serialize_type_);
        }
    }

    uint32_t Grape::GetMaxIter()
    {
        return max_iter_;
    }

    GRAPH_PHASE Grape::GetGraphPhase()
    {
        return grape_phase_;
    }

}

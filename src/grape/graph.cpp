#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fstream>
#include <algorithm>
#include "grape/graph.h"
#include "grape/error.h"
#include "grape/global_config.h"
#include "grape/log.h"
#include "grape/optimizer/sgd.h"
#include "grape/util/cuda.h"

namespace Grape{
    static std::string TAG = "Grape";

    Graph::Graph(
        std::string save_path,
        SERIALIZE_TYPE serialize_type,
        int32_t max_iter,
        int32_t display_iter,
        int32_t snapshot_iter,
        PHASE graph_phase,
        CAL_MODE cal_mode
    ):
    save_path_(save_path),
    serialize_type_(serialize_type),
    max_iter_(max_iter),
    display_iter_(display_iter),
    snapshot_iter_(snapshot_iter),
    graph_phase_(graph_phase),
    cal_mode_(cal_mode)
    {

    }

    Graph::Graph(GraphParams &graph_params)
    {
        save_path_ = graph_params.save_path_;
        max_iter_ = graph_params.max_iter_;
        display_iter_ = graph_params.display_iter_;
        snapshot_iter_ = graph_params.snapshot_iter_;
        device_id_ = graph_params.device_id_;
        //seiralize type
        if(graph_params.serialize_type_ == SERIALIZE_TYPE_BINARY_STRING){
            serialize_type_ = BINARY;
        }else if(graph_params.serialize_type_== SERIALIZE_TYPE_JSON_STRING){
            serialize_type_ = JSON;
        }else if(graph_params.serialize_type_== SERIALIZE_TYPE_XML_STRING){
            serialize_type_ = XML;
        }
        //phase
        if(graph_params.phase_ == PHASE_TRAIN_STRING){
            graph_phase_ = TRAIN;
        }else if(graph_params.phase_== PHASE_TEST_STRING){
            graph_phase_ = TEST;
        }
        //cal mode
        if(graph_params.cal_mode_ == CAL_MODE_CPU_STRING){
            cal_mode_ = CPU_MODE;
        }else if(graph_params.cal_mode_== CAL_MODE_GPU_STRING){
            cal_mode_ = GPU_MODE;
        }

    }

    Graph::~Graph()
    {

    }


    void Graph::Backward()
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
    void Graph::Forward()
    {
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
    void Graph::UpdateWeights()
    {
        for(auto o:ops_){
            if(cal_mode_ == CPU_MODE){
                o->UpdateWeightsCpu(*optimizer_);
            }else{
#ifdef GPU
                o->UpdateWeightsGpu(*optimizer_);
#endif
            }
        }
    }
    void Graph::Setup(bool load)
    {
        //std::cout<<ops_.size()<<std::endl;
        #ifdef GPU
            cuda_set_device(device_id_);
        #endif
        if(load){
            Load(save_path_);
        }else
        {
            for(auto o:ops_){
                o->Setup();
            }
        }
        

    }

    void Graph::TrainOnce()
    {
        Forward();
        Backward();
        UpdateWeights();
    }

    void Graph::Train()
    {
        for(auto o:ops_){
            o->OnTrainBegin();
        }
        for(uint32_t i=0;i<max_iter_;i++){
            TrainOnce();
            optimizer_->CheckLrUpdate(run_iter_*max_iter_+i);
            if((run_iter_*max_iter_+i+1)%display_iter_==0){
                for(auto o:ops_){
                    o->Display();
                }
            }
            if(snapshot_iter_>0 && (run_iter_*max_iter_+i+1)%snapshot_iter_==0){
                Save(save_path_+std::to_string((run_iter_*max_iter_+i+1)/snapshot_iter_));
            }
        }
        for(auto o:ops_){
            o->OnTrainEnd();
        }
    }

    void Graph::TestOnce()
    {
        Forward();
    }

    void Graph::Test()
    {
        for(auto o:ops_){
            o->OnTestBegin();
        }
        for(uint32_t i=0;i<max_iter_;i++){
            Forward();
            if((i+1)%display_iter_==0){
                for(auto o:ops_){
                    o->Display();
                }
            }
        }
        for(auto o:ops_){
            o->OnTestEnd();
        }
    }

    void Graph::Run()
    {
        #ifdef GPU
            cuda_set_device(device_id_);
        #endif
        ReConnection();
        if(graph_phase_ == TRAIN){
            Train();
            Save(save_path_);
        }else{
            Test();
        }
        run_iter_++;
    }

    void Graph::RunOnce()
    {
        if(graph_phase_ == TRAIN){
            TrainOnce();
        }else{
            TestOnce();
        }
    }

    void Graph::GetConnection(Op *op,std::vector<OpConnectionPoint> &connections)
    {
        std::vector<tensorptr_t> output_tensors = op->next();
        for(int i=0;i<output_tensors.size();++i){
            std::vector<Op *> consumer_op = output_tensors[i]->next();
            for(int j=0;j<consumer_op.size();j++){
                uint32_t index = consumer_op[j]->PrevPort(*output_tensors[i].get());
                OpConnectionPoint opconnp;
                opconnp.head = op;
                opconnp.head_index = i;
                opconnp.tail = consumer_op[j];
                opconnp.tail_index = index;
                connections.emplace_back(opconnp);
            }
        }
    }

    void Graph::SnapShotConnections()
    {
        connections_.clear();
        for(auto tmp:ops_){
            GetConnection(tmp,connections_);
        }
    }

    void Graph::ReConnection()
    {
        for(auto tmp:connections_){
            connect_op(tmp.head,tmp.tail,tmp.head_index,tmp.tail_index);
        }
    }

    void Graph::Construct(
        const std::vector<Op *> &inputs,
        const std::vector<Op *> &outputs) 
    {
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
                        std::vector<uint8_t>(next[i]->PrevDataOps().size(), 0);
                }

                auto &removed = removed_edge[next[i]];
                removed[FindIndex(next[i]->PrevDataOps(), curr)] = 1;

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
        SnapShotConnections();
    }

    int32_t Graph::FindIndex(const std::vector<Op *> &ops, Op *target) 
    {
        for (int32_t i = 0; i < ops.size(); i++) {
            if (ops[i] == static_cast<Op *>(&*target)) return i;
        }
        throw Error("invalid connection "+target->get_name());
    }

    void Graph::Save(std::string path)
    {
        
        switch (serialize_type_)
        {
        case BINARY:{
            std::ofstream save_stream(path+".binary");
            cereal::BinaryOutputArchive archive(save_stream);
            for(auto o:ops_){
                o->Save(archive);
            }
        }
        break;
        case JSON:{
            std::ofstream save_stream(path+".json");
            cereal::JSONOutputArchive archive(save_stream);
            for(auto o:ops_){
                o->Save(archive);
            }
        }
        break;
        case XML:{
            std::ofstream save_stream(path+".xml");
            cereal::XMLOutputArchive archive(save_stream);
            for(auto o:ops_){
                o->Save(archive);
            }
        }
        break;
        default:
            break;
        }
    }

    void Graph::Load(std::string path)
    {
      
        switch (serialize_type_)
        {
        case BINARY:{
            std::ifstream load_stream(path+".binary");
            cereal::BinaryInputArchive archive(load_stream);
            for(auto o:ops_){
                o->Load(archive);
            }
        }
        break;
        case JSON:{
            std::ifstream load_stream(path+".json");
            cereal::JSONInputArchive archive(load_stream);
            for(auto o:ops_){
                o->Load(archive);
            }
        }
        break;
        case XML:{
            std::ifstream load_stream(path+".xml");
            cereal::XMLInputArchive archive(load_stream);
            for(auto o:ops_){
                o->Load(archive);
            }
        }
        break;
        default:
            break;
        }
    }

    void Graph::OnNetRunBegin()
    {
        run_iter_ = 0;
    }

    void Graph::OnNetRunEnd()
    {
        run_iter_ = 0;
    }

    PHASE Graph::GetPhase()
    {
        return graph_phase_;
    }

    void Graph::SetPhase(PHASE phase)
    {
        graph_phase_ = phase;
    }

    uint32_t Graph::GetMaxIter()
    {
        return max_iter_;
    }

    void Graph::SetMaxIter(uint32_t iter)
    {
        max_iter_ = iter;
    }
}

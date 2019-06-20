#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>
#include "javernn/graph.h"
#include "javernn/nn_error.h"

namespace javernn{
    
void Graph::Backward()
{
    for(auto o:ops_){
        o->Backward();
    }
}
std::vector<Tensor> Graph::Forward()
{
    for(auto o:ops_){
        o->Forward();
    }
} 
void Graph::UpdateWeights(Optimizer *opt)
{
    for(auto o:ops_){
        o->UpdateWeights(opt);
    }
}
void Graph::Setup(bool reset_weight)
{
    for(auto o:ops_){
        o->Setup(reset_weight);
    }
}
void Graph::Construct(const std::vector<Op *> &input,
                const std::vector<Op *> &output) {
    std::vector<Op *> sorted;
    std::vector<Op *> input_nodes(input.begin(), input.end());
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

    for (auto &n : sorted) {
      ops_.push_back(n);
    }

    input_layers_  = input;
    output_layers_ = output;

    Setup(false);
}

size_t FindIndex(const std::vector<Op *> &ops, Op *target) 
{
    for (size_t i = 0; i < ops.size(); i++) {
        if (ops[i] == static_cast<Op *>(&*target)) return i;
    }
    throw Error("invalid connection");
}

}

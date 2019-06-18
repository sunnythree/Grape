#include <algorithm>
#include "javernn/node.h"
#include "javernn/edge.h"

namespace javernn{

    Node::Node(size_t in_size, size_t out_size) : prev_(in_size), next_(out_size) 
    {

    }
    Node::~Node() 
    {

    }

    size_t Node::PrevPort(const Edge &e) const {
        auto it = std::find_if(prev_.begin(), prev_.end(),
                                [&](edgeptr_t ep) { return ep.get() == &e; });
        return (size_t)std::distance(prev_.begin(), it);
    }

    size_t Node::NextPort(const Edge &e) const {
        auto it = std::find_if(next_.begin(), next_.end(),
                                [&](edgeptr_t ep) { return ep.get() == &e; });
        return (size_t)std::distance(next_.begin(), it);
    }

    std::vector<Node *> Node::PrevNodes() const 
    {
        std::vector<Node *> vecs;
        for (auto &e : prev_) {
            if (e && e->Prev()) {
                vecs.insert(vecs.end(), e->Prev());
            }
        }
    return vecs;
    }

    std::vector<Node *> Node::NextNodes() const 
    {
        std::vector<Node *> vecs;
        for (auto &e : next_) {
            if (e) {
                auto n = e->Next();
                vecs.insert(vecs.end(), n.begin(), n.end());
            }
        }
        return vecs;
    }
}

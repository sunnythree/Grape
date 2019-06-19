#ifndef __javernn_edge_h__
#define __javernn_edge_h__

#include <vector>

namespace javernn{
    class Node;
    class Edge {
    public:
        Edge(Node *prev)
        : prev_(prev) {}
        virtual ~Edge() {};

        inline const std::vector<Node *> &Next() const { return next_; }
        inline Node *Prev() { return prev_; }
        inline const Node *Prev() const { return prev_; }
        inline void AddNextNode(Node *next) { next_.push_back(next); }

    private:
        Node *prev_;                // previous node, "producer" of this tensor
        std::vector<Node *> next_;  // next nodes, "consumers" of this tensor
    };
}

#endif
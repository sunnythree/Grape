#ifndef __node_h__
#define __node_h__

#include <memory>
#include <vector>


namespace javernn{

class Edge;
typedef std::shared_ptr<Edge> edgeptr_t;

class Node{
public:
    Node(size_t in_size, size_t out_size);
    virtual ~Node();

    inline const std::vector<edgeptr_t> &prev() const { return prev_; }
    inline const std::vector<edgeptr_t> &next() const { return next_; }

    size_t PrevPort(const Edge &e) const;
    size_t NextPort(const Edge &e) const;

    std::vector<Node *> PrevNodes() const;
    std::vector<Node *> NextNodes() const;

protected:
    Node() = delete;

    friend void connect(Node *head,
                        Node *tail,
                        size_t head_index,
                        size_t tail_index);

    mutable std::vector<edgeptr_t> prev_;
    mutable std::vector<edgeptr_t> next_;
};


}

#endif
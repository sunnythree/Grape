#ifndef __GRAPE_NODE_H__
#define __GRAPE_NODE_H__

#include <memory>
#include <vector>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "grape/optimizer/optimizer.h"
#include "grape/error.h"
#include "cereal/archives/binary.hpp"
#include "cereal/archives/xml.hpp"
#include "cereal/archives/json.hpp"


namespace Grape{

class Tensor;
typedef std::shared_ptr<Tensor> tensorptr_t;

typedef struct op_connection_t{
    std::string head;
    std::string tail;
    uint32_t head_index;
    uint32_t tail_index;
}OpConnection;

class Op;
typedef struct op_connection_point_t{
    Op *head;
    Op *tail;
    uint32_t head_index;
    uint32_t tail_index;
}OpConnectionPoint;

class Op{
public:
    Op() = delete;
    explicit Op(const std::vector<TENSOR_TYPE> &in_type,
        const std::vector<TENSOR_TYPE> &out_type);
    virtual ~Op();

    inline const std::vector<tensorptr_t> &prev() const { return prev_; }
    inline const std::vector<tensorptr_t> &next() const { return next_; }

    int32_t PrevPort(const Tensor &e) const;
    int32_t NextPort(const Tensor &e) const;

    std::vector<Op *> PrevOps() const;
    std::vector<Op *> PrevDataOps() const;
    std::vector<Op *> NextOps() const;

    inline int32_t in_size() const { return in_size_; }
    inline int32_t out_size() const { return out_size_; }

    virtual void Setup() = 0;
    
    virtual void ForwardCpu() = 0; 
    virtual void BackwardCpu() = 0;
    virtual void UpdateWeightsCpu(Optimizer &opt) = 0;
    std::string type_;
#ifdef GPU
    virtual void ForwardGpu() = 0; 
    virtual void BackwardGpu() = 0;
    virtual void UpdateWeightsGpu(Optimizer &opt) = 0;
#endif

    virtual void Load(cereal::BinaryInputArchive &archive){};
    virtual void Load(cereal::JSONInputArchive &archive){};
    virtual void Load(cereal::XMLInputArchive &archive){};
    virtual void Save(cereal::BinaryOutputArchive &archive){};
    virtual void Save(cereal::JSONOutputArchive &archive){};
    virtual void Save(cereal::XMLOutputArchive &archive){};

    virtual void OnTrainBegin(){};
    virtual void OnTrainEnd(){};
    virtual void OnTestBegin(){};
    virtual void OnTestEnd(){};
    virtual void Display(){};



    inline void set_name(std::string name){name_ = name;};
    inline std::string get_name(){return name_;};
protected:
    friend void connect_op(Op *head,Op *tail,int32_t head_index,int32_t tail_index);
    mutable std::vector<tensorptr_t> prev_;
    mutable std::vector<tensorptr_t> next_;
    std::vector<TENSOR_TYPE> in_type_;
    std::vector<TENSOR_TYPE> out_type_;
    bool initialized_ = false;
    int32_t in_size_ = 0;
    int32_t out_size_ = 0;
    std::string name_ = "";
};

void connection_mismatch(const Op &from, const Op &to);
void connect_op(Op *head,Op *tail,int32_t head_index = 0,int32_t tail_index = 0);

std::vector<Op *> operator,(Op &lhs, Op &rhs);
std::vector<Op *> &operator,(std::vector<Op *> &lhs, Op &rhs);
std::vector<Op *> &operator,(Op &lhs, std::vector<Op *> &rhs);
std::vector<Op *> &operator,(std::vector<Op *> &lhs, std::vector<Op *> &rhs);
Op &operator<<(Op &lhs, Op &rhs);
Op &operator<<(std::vector<Op *> &lhs, Op &rhs);
std::vector<Op *> &operator<<(Op &rhs, std::vector<Op *> &lhs);
}

#endif
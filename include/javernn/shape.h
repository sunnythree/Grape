#ifndef __JAVERNN_SHAPE_H__
#define __JAVERNN_SHAPE_H__
#include <vector>
#include <cstdint>

namespace javernn{
    class Shape{
    public:
        Shape(std::vector<uint32_t> shape_size):shape_size_(shape_size){};
        
        inline int count(uint32_t start_axis, uint32_t end_axis) const {
            int count = 1;
            for (int i = start_axis; i < end_axis; ++i) {
                count *= shape_size_[i];
            }
            return count;
        }
        inline uint32_t num_axes() const { return shape_size_.size(); }
        inline uint32_t count() const { return count(0, num_axes());; }
        inline uint32_t count(uint32_t start_axis) const {
            return count(start_axis, num_axes());
        }
        inline uint32_t index(uint32_t i) const {return shape_size_[i];};
    private:
        std::vector<uint32_t> shape_size_;

    };
}

#endif
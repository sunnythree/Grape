#ifndef __GRAPE_SHAPE_H__
#define __GRAPE_SHAPE_H__
#include <vector>
#include <cstdint>

namespace Grape{
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
        //for image
        inline uint32_t get_width() const {return shape_size_[0];};
        inline uint32_t get_height() const {return shape_size_[1];};
        inline uint32_t get_channel() const {return shape_size_[2];};
        inline uint32_t get_batch() const {return shape_size_[3];};
    private:
        std::vector<uint32_t> shape_size_;

    };
}

#endif
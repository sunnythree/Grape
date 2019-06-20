#ifndef __JAVERNN_SHAPE_H__
#define __JAVERNN_SHAPE_H__
#include <vector>
namespace javernn{
    class Shape{
    public:
        Shape(std::vector<int> shape_size):shape_size_(shape_size){};
        
        inline int count(int start_axis, int end_axis) const {
            int count = 1;
            for (int i = start_axis; i < end_axis; ++i) {
                count *= shape_size_(i);
            }
            return count;
        }
        inline int num_axes() const { return shape_size_.size(); }
        inline int count() const { return count(0, num_axes());; }
        inline int count(int start_axis) const {
            return count(start_axis, num_axes());
        }
        inline int index(int i) const {return shape_size_[i]};
    private:
        std::vector<int> shape_size_;

    };
}

#endif
#ifndef __JAVERNN_UTIL_H__
#define __JAVERNN_UTIL_H__

#include <cstdint>
#include <vector>
namespace javernn
{
    template <typename T, typename Pred, typename Sum>
    int32_t sumif(const std::vector<T> &vec, Pred p, Sum s) {
    size_t sum = 0;
    for (size_t i = 0; i < vec.size(); i++) {
        if (p(i)) sum += s(vec[i]);
    }
    return sum;
    }

    template <typename T, typename Pred>
    std::vector<T> filter(const std::vector<T> &vec, Pred p) {
    std::vector<T> res;
    for (size_t i = 0; i < vec.size(); i++) {
        if (p(i)) res.push_back(vec[i]);
    }
    return res;
    }

    float sum_array(float *a, int n);

} // namespace javernn





#endif
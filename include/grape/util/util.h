#ifndef __GRAPE_UTIL_H__
#define __GRAPE_UTIL_H__

#include <cstdint>
#include <vector>
#include <string>

namespace Grape
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

    uint32_t max_index(const float *a, int n);

    void split(
        const std::string& src,
        const std::string& separator, 
        std::vector<std::string>& dest
    );

    void trim(std::string &s);
} // namespace Grape





#endif
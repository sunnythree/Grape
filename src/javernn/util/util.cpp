#include <float.h>
#include "javernn/util/util.h"

namespace javernn
{
    float sum_array(float *a, int n)
    {
        float sum = 0;
        for(uint32_t i = 0; i < n; ++i) sum += a[i];
        return sum;
    }
    
    uint32_t max_index(const float *a, int n)
    {
        uint32_t index;
        float max = -FLT_MAX;
        for(uint32_t i = 0;i < n; ++i){
            if(a[i] > max){
                index = i;
                max = a[i];
            }
        }
        return index;
    }
} // namespace javernn

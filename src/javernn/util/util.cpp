#include "javernn/util/util.h"

namespace javernn
{
    float sum_array(float *a, int n)
    {
        float sum = 0;
        for(int i = 0; i < n; ++i) sum += a[i];
        return sum;
    }
} // namespace javernn

#include <float.h>
#include "grape/util/util.h"

namespace Grape
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

    void split(const std::string& src, const std::string& separator, std::vector<std::string>& dest)
    {
        std::string str = src;
        std::string substring;
        std::string::size_type start = 0, index;

        do
        {
            index = str.find_first_of(separator,start);
            if (index != std::string::npos)
            {    
                substring = str.substr(start,index-start);
                dest.push_back(substring);
                start = str.find_first_not_of(separator,index);
                if (start == std::string::npos) return;
            }
        }while(index != std::string::npos);
        
        //the last token
        substring = str.substr(start);
        dest.push_back(substring);
    }

     void trim(std::string &s)
     {
         if( !s.empty() ){
             s.erase(0,s.find_first_not_of(" "));
             s.erase(s.find_last_not_of(" ") + 1);
         }
     }
} // namespace Grape

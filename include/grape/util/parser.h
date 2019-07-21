#ifndef __GRAPE_PARSER_H__
#define __GRAPE_PARSER_H__

#include "grape/params/net_params.h"

namespace Grape{
    class Parser
    {
    public:
        static void Parse(NetParams &params,std::string path);
        static void Serialize(NetParams &params,std::string path);
    }
    
}


#endif
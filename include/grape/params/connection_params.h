#ifndef __GRAPE_CONNECTION_PARAMS_H__
#define __GRAPE_CONNECTION_PARAMS_H__

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include "cereal/archives/json.hpp"



namespace Grape{
    typedef struct conn_t
    {
        std::string from;
        std::string to;
        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("from",from));
            ar(cereal::make_nvp("to",to));
        }
    }Conn;
    

    class ConnectionParams{
    public:
        std::string op_list_name_ = "";
        std::string graph_name_  = "";
        std::vector<Conn> connections_;

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(cereal::make_nvp("op_list_name",op_list_name_));
            ar(cereal::make_nvp("graph_name",graph_name_));
            ar(cereal::make_nvp("cnnections",connections_));
        }
    };
}

#endif
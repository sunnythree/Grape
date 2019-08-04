#include <iostream>
#include "grape/log.h"

namespace Grape{

    LOG_LEVEL Log::log_level_= VERBOSE;

    void Log::log_print(std::string tag,std::string log)
    {
        std::cout<<tag<<" "<<log<<std::endl;
    }

    void Log::v(std::string tag,std::string log)
    {
        if(log_level_ >= VERBOSE){
            log_print(tag,log);
        }
    }

    void Log::d(std::string tag,std::string log)
    {
        if(log_level_ >= DEBUG){
            log_print(tag,log);
        }
    }

    void Log::i(std::string tag,std::string log)
    {
        if(log_level_ >= INFO){
            log_print(tag,log);
        }
    }

    void Log::e(std::string tag,std::string log)
    {
        if(log_level_ >= ERROR){
            log_print(tag,log);
        }
    }
    
    void Log::set_log_level(LOG_LEVEL level)
    {
        log_level_ = level;
    }
}
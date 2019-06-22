#ifndef __JAVERNN_LOG_H__
#define __JAVERNN_LOG_H__

#include <cstdint>
#include <string>

namespace javernn{
    enum LOG_LEVEL{
        VERBOSE,
        DEBUG,
        INFO,
        ERROR
    };
    class Log{
    public:
        static void log_print(std::string tag,std::string log);
        static void v(std::string tag,std::string log);
        static void d(std::string tag,std::string log);
        static void i(std::string tag,std::string log);
        static void e(std::string tag,std::string log);
        static void set_log_level(LOG_LEVEL level);
    private:
        static LOG_LEVEL log_level_;
    };
}

#endif
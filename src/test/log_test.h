#include "gtest/gtest.h"
#include "grape/log.h"

using namespace Grape;
TEST(log,level)
{
    Log::set_log_level(VERBOSE);
    Log::v("test","VERBOSE");
    Log::set_log_level(DEBUG);
    Log::d("test","DEBUG");
    Log::set_log_level(INFO);
    Log::i("test","INFO");
    Log::set_log_level(ERROR);
    Log::v("test","ERROR");
}

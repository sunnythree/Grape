#include "gtest/gtest.h"
#include "grape/util/util.h"

TEST(util,split)
{
    std::string tmp = "a b c d e f g";
    std::vector<std::string> results;
    Grape::split(tmp," ",results);
    EXPECT_EQ(results.size(),7);
    EXPECT_EQ(results[0],"a");
    EXPECT_EQ(results[1],"b");
    EXPECT_EQ(results[2],"c");
    EXPECT_EQ(results[3],"d");
    EXPECT_EQ(results[4],"e");
    EXPECT_EQ(results[5],"f");
    EXPECT_EQ(results[6],"g");
}


#include <iostream>
#include "gtest/gtest.h"
#include "util_test.h"
#include "parser_test.h"
#include "log_test.h"
#include "test_pool.h"
#include "test_conv.h"


using namespace std;

int main(int argc,char **argv)
{
    testing::InitGoogleTest(&argc, argv);//将命令行参数传递给gtest
    return RUN_ALL_TESTS();   //RUN_ALL_TESTS()运行所有测试案例
}